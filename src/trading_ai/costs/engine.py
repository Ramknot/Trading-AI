"""Balanced transaction-cost engine with point-in-time estimates and reconciliation."""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

from trading_ai.backtesting.models import Fill
from trading_ai.core.hashing import stable_hash
from trading_ai.core.models import OrderSide, TradingProfile
from trading_ai.costs.base import TransactionCostEngine
from trading_ai.costs.commission import commission_amount
from trading_ai.costs.config import CostConfigurationBundle, load_balanced_cost_config
from trading_ai.costs.exchange_fees import exchange_fee_amount
from trading_ai.costs.financing import balanced_financing_component
from trading_ai.costs.fx import convert_component, fx_cost_component
from trading_ai.costs.models import (
    ActualTradingCost,
    BPS,
    CostComponent,
    CostCoverage,
    CostReconciliation,
    CostStatus,
    EstimatedCashRequirement,
    InstrumentCostMetadata,
    PreTradeCostEstimate,
    PreTradeCostRequest,
    TariffStatus,
    TradingCostBreakdown,
    ZERO,
)
from trading_ai.costs.operating import build_operating_costs
from trading_ai.costs.spread_slippage import estimated_spread_slippage
from trading_ai.costs.taxes import transaction_tax_component
from trading_ai.portfolio.currency import CurrencyConverter, SameCurrencyConverter


QUANTUM = Decimal("0.00000001")


def _q(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


def _available_amount(component: CostComponent) -> Decimal:
    if component.amount is None:
        raise AssertionError("UNAVAILABLE cost reached a complete-cost aggregation")
    return component.amount


def _total_component(
    components: Iterable[CostComponent], currency: str
) -> CostComponent:
    items = tuple(components)
    if any(item.status is CostStatus.UNAVAILABLE for item in items):
        return CostComponent.unavailable(
            "total_variable_cost",
            currency,
            "BalancedTransactionCostEngine",
            "one or more variable-cost components are unavailable",
        )
    amount = sum((_available_amount(item) for item in items), ZERO)
    if all(item.status is CostStatus.NOT_APPLICABLE for item in items):
        return CostComponent.not_applicable(
            "total_variable_cost",
            currency,
            "BalancedTransactionCostEngine",
            "all variable costs are explicitly not applicable",
        )
    status = (
        CostStatus.ESTIMATED
        if any(item.status is CostStatus.ESTIMATED for item in items)
        else CostStatus.KNOWN
    )
    return CostComponent(
        "total_variable_cost",
        status,
        _q(amount),
        currency,
        "sum of explicit variable-cost components",
    )


def _breakdown(
    *,
    commission: CostComponent,
    spread: CostComponent,
    slippage: CostComponent,
    exchange_fees: CostComponent,
    transaction_tax: CostComponent,
    fx_cost: CostComponent,
    financing_cost: CostComponent,
    other_variable_cost: CostComponent,
    currency: str,
) -> TradingCostBreakdown:
    components = (
        commission,
        spread,
        slippage,
        exchange_fees,
        transaction_tax,
        fx_cost,
        financing_cost,
        other_variable_cost,
    )
    return TradingCostBreakdown(*components, _total_component(components, currency))


def _combine_component(
    name: str, entry: CostComponent, exit_: CostComponent, currency: str
) -> CostComponent:
    if CostStatus.UNAVAILABLE in {entry.status, exit_.status}:
        return CostComponent.unavailable(
            name, currency, "round-trip estimate",
            "entry or exit component is unavailable",
        )
    amount = _available_amount(entry) + _available_amount(exit_)
    statuses = {entry.status, exit_.status}
    if statuses == {CostStatus.NOT_APPLICABLE}:
        return CostComponent.not_applicable(
            name, currency, "round-trip estimate", "not applicable on either leg"
        )
    status = CostStatus.ESTIMATED if CostStatus.ESTIMATED in statuses else CostStatus.KNOWN
    return CostComponent(name, status, _q(amount), currency, "entry plus estimated exit")


def combine_round_trip(
    entry: TradingCostBreakdown, exit_: TradingCostBreakdown, currency: str
) -> TradingCostBreakdown:
    names = (
        "commission", "spread", "slippage", "exchange_fees",
        "transaction_tax", "fx_cost", "financing_cost", "other_variable_cost",
    )
    values = tuple(
        _combine_component(name, getattr(entry, name), getattr(exit_, name), currency)
        for name in names
    )
    return TradingCostBreakdown(*values, _total_component(values, currency))


class BalancedTransactionCostEngine(TransactionCostEngine):
    """Deterministic offline costs; no broker, network, strategy, or Risk authority."""

    def __init__(
        self,
        bundle: CostConfigurationBundle,
        *,
        currency_converter: CurrencyConverter | None = None,
    ) -> None:
        if not bundle.config.enabled:
            raise ValueError("transaction-cost configuration is disabled")
        self.bundle = bundle
        self.converter = currency_converter or SameCurrencyConverter()

    @classmethod
    def from_profile(
        cls,
        profile: TradingProfile,
        *,
        tariff_profile: str | None = None,
        currency_converter: CurrencyConverter | None = None,
    ) -> BalancedTransactionCostEngine:
        return cls(
            load_balanced_cost_config(profile, tariff_profile=tariff_profile),
            currency_converter=currency_converter,
        )

    @property
    def engine_name(self) -> str:
        return self.bundle.config.engine_name

    @property
    def engine_version(self) -> str:
        return self.bundle.config.engine_version

    @property
    def config_hash(self) -> str:
        return self.bundle.config_hash

    @property
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        return self.bundle.config.to_parameters()

    @property
    def tariff(self):
        return self.bundle.tariff

    @property
    def tariff_profile_id(self) -> str:
        return self.bundle.tariff.profile_id

    @property
    def tariff_status(self) -> TariffStatus:
        return self.bundle.tariff.status

    def operating_costs(self, period_start, period_end):
        return build_operating_costs(
            self.bundle.operating, period_start, period_end
        )

    def _instrument(self, symbol: str) -> InstrumentCostMetadata | None:
        return self.bundle.instrument_for(symbol)

    def _tariff_status(
        self, request: PreTradeCostRequest, metadata: InstrumentCostMetadata
    ) -> tuple[CostStatus, bool, str | None]:
        tariff = self.bundle.tariff
        if metadata.market not in tariff.markets:
            return CostStatus.UNAVAILABLE, False, "tariff does not cover instrument market"
        covered = tariff.covers(request.timestamp, metadata.market)
        if covered and tariff.status is TariffStatus.VERIFIED:
            return CostStatus.KNOWN, True, None
        if self.bundle.config.allow_retrospective_tariff:
            return CostStatus.ESTIMATED, False, "CURRENT_TARIFF_APPLIED_RETROSPECTIVELY"
        return CostStatus.UNAVAILABLE, False, "tariff is unverified, expired, or outside effective dates"

    def _leg(
        self,
        request: PreTradeCostRequest,
        *,
        actual_spread: Decimal | None = None,
        actual_slippage: Decimal | None = None,
    ) -> tuple[TradingCostBreakdown, bool, tuple[str, ...]]:
        config = self.bundle.config
        tariff = self.bundle.tariff
        metadata = self._instrument(request.symbol)
        if metadata is None:
            unavailable = CostComponent.unavailable(
                "commission", config.base_currency, "instrument cost metadata",
                "symbol metadata is unavailable",
            )
            generic = tuple(
                CostComponent.unavailable(
                    name, config.base_currency, "instrument cost metadata",
                    "symbol metadata is unavailable",
                )
                for name in (
                    "spread", "slippage", "exchange_fees", "transaction_tax",
                    "fx_cost", "financing_cost", "other_variable_cost",
                )
            )
            breakdown = _breakdown(
                commission=unavailable,
                spread=generic[0], slippage=generic[1], exchange_fees=generic[2],
                transaction_tax=generic[3], fx_cost=generic[4],
                financing_cost=generic[5], other_variable_cost=generic[6],
                currency=config.base_currency,
            )
            return breakdown, False, ("UNKNOWN_INSTRUMENT_COST_METADATA",)

        notional = request.reference_price * request.quantity
        status, period_covered, tariff_warning = self._tariff_status(request, metadata)
        if status is CostStatus.UNAVAILABLE:
            commission = CostComponent.unavailable(
                "commission", config.base_currency, tariff.source_reference,
                tariff_warning or "tariff unavailable",
            )
        else:
            amount = commission_amount(
                tariff,
                quantity=request.quantity,
                notional=notional,
                monthly_volume_before=request.monthly_volume_before,
            )
            commission = CostComponent(
                "commission", status, _q(amount), tariff.currency,
                tariff.source_reference, tariff_warning,
            )
            commission = convert_component(
                commission,
                target_currency=config.base_currency,
                timestamp=request.timestamp,
                converter=self.converter,
            )

        spread_amount, slippage_amount = estimated_spread_slippage(
            notional, request.spread_bps, request.slippage_bps
        )
        spread = CostComponent(
            "spread",
            CostStatus.KNOWN if actual_spread is not None else CostStatus.ESTIMATED,
            _q(actual_spread if actual_spread is not None else spread_amount),
            metadata.currency,
            "BarExecutionModel spread_bps convention",
        )
        slippage = CostComponent(
            "slippage",
            CostStatus.KNOWN if actual_slippage is not None else CostStatus.ESTIMATED,
            _q(actual_slippage if actual_slippage is not None else slippage_amount),
            metadata.currency,
            "BarExecutionModel slippage_bps convention",
        )
        spread = convert_component(
            spread, target_currency=config.base_currency,
            timestamp=request.timestamp, converter=self.converter,
        )
        slippage = convert_component(
            slippage, target_currency=config.base_currency,
            timestamp=request.timestamp, converter=self.converter,
        )

        if tariff.exchange_fees_included_in_commission:
            exchange = CostComponent.not_applicable(
                "exchange_fees", config.base_currency, tariff.source_reference,
                "fixed tariff explicitly includes exchange/pass-through fees",
            )
        elif tariff.exchange_fee_status is CostStatus.UNAVAILABLE:
            exchange = CostComponent.unavailable(
                "exchange_fees", config.base_currency, tariff.source_reference,
                "venue/pass-through fee is unavailable",
            )
        else:
            amount = exchange_fee_amount(
                tariff, quantity=request.quantity, notional=notional
            )
            exchange = CostComponent(
                "exchange_fees", tariff.exchange_fee_status, _q(amount),
                tariff.currency, tariff.source_reference,
            )
            exchange = convert_component(
                exchange, target_currency=config.base_currency,
                timestamp=request.timestamp, converter=self.converter,
            )

        rule = (
            self.bundle.tax_rule(metadata.transaction_tax_rule_id)
            if metadata.transaction_tax_rule_id is not None else None
        )
        tax = transaction_tax_component(
            metadata=metadata,
            rule=rule,
            side=request.side,
            notional=notional,
            timestamp=request.timestamp,
            allow_retrospective=config.allow_retrospective_tariff,
        )
        tax = convert_component(
            tax, target_currency=config.base_currency,
            timestamp=request.timestamp, converter=self.converter,
        )
        fx = fx_cost_component(
            notional=notional,
            from_currency=metadata.currency,
            base_currency=config.base_currency,
            timestamp=request.timestamp,
            converter=self.converter,
            fx_cost_bps=config.fx_cost_bps,
        )
        financing = balanced_financing_component(config.base_currency)
        other = CostComponent.not_applicable(
            "other_variable_cost", config.base_currency, "Balanced cost configuration",
            "no other variable cost declared",
        )
        breakdown = _breakdown(
            commission=commission,
            spread=spread,
            slippage=slippage,
            exchange_fees=exchange,
            transaction_tax=tax,
            fx_cost=fx,
            financing_cost=financing,
            other_variable_cost=other,
            currency=config.base_currency,
        )
        warnings = tuple(
            item for item in (
                tariff_warning,
                "INSTRUMENT_METADATA_UNVERIFIED"
                if metadata.metadata_status is not TariffStatus.VERIFIED else None,
            ) if item is not None
        )
        return breakdown, period_covered, warnings

    def estimate(self, request: PreTradeCostRequest) -> PreTradeCostEstimate:
        entry, period_covered, warnings = self._leg(request)
        exit_request = PreTradeCostRequest(
            timestamp=request.timestamp,
            symbol=request.symbol,
            side=OrderSide.SELL if request.side is OrderSide.BUY else OrderSide.BUY,
            quantity=request.quantity,
            reference_price=request.reference_price,
            timeframe=request.timeframe,
            spread_bps=request.spread_bps,
            slippage_bps=request.slippage_bps,
            order_id=request.order_id,
            signal_id=request.signal_id,
            portfolio_plan_id=request.portfolio_plan_id,
            monthly_volume_before=request.monthly_volume_before,
        )
        exit_costs, exit_period_covered, exit_warnings = self._leg(exit_request)
        round_trip = combine_round_trip(
            entry, exit_costs, self.bundle.config.base_currency
        )
        metadata = self._instrument(request.symbol)
        base_notional: Decimal | None = None
        if metadata is not None and self.converter.has_rate(
            metadata.currency, self.bundle.config.base_currency, request.timestamp
        ):
            base_notional = self.converter.convert(
                request.reference_price * request.quantity,
                metadata.currency,
                self.bundle.config.base_currency,
                request.timestamp,
            )
        entry_total = entry.amount_if_complete
        buffer = (
            max(
                base_notional * self.bundle.config.cash_buffer_bps / BPS,
                self.bundle.config.cash_buffer_absolute,
            )
            if base_notional is not None else ZERO
        )
        if base_notional is None or entry_total is None:
            cash_requirement = EstimatedCashRequirement(
                request.reference_price * request.quantity,
                None, ZERO, None, None,
                self.bundle.config.base_currency,
                CostCoverage.INCOMPLETE,
            )
        else:
            total_cash = base_notional + entry_total + buffer
            cash_requirement = EstimatedCashRequirement(
                _q(base_notional), _q(entry_total), _q(buffer), _q(total_cash),
                _q(total_cash / request.quantity),
                self.bundle.config.base_currency,
                CostCoverage.COMPLETE,
            )
        payload = {
            "request": request,
            "entry": entry,
            "exit": exit_costs,
            "round_trip": round_trip,
            "config_hash": self.config_hash,
            "tariff_hash": self.bundle.tariff.config_hash,
        }
        digest = stable_hash(payload)
        lineage = tuple(sorted(
            item for item in (
                ("signal_id", request.signal_id) if request.signal_id else None,
                ("portfolio_plan_id", request.portfolio_plan_id) if request.portfolio_plan_id else None,
            ) if item is not None
        ))
        return PreTradeCostEstimate(
            estimate_id=f"cost-estimate-{digest[:24]}",
            timestamp=request.timestamp,
            order_id=request.order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            reference_price=request.reference_price,
            entry_costs=entry,
            estimated_exit_costs=exit_costs,
            round_trip_costs=round_trip,
            cash_requirement=cash_requirement,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            config_hash=self.config_hash,
            tariff_profile_id=self.bundle.tariff.profile_id,
            tariff_status=self.bundle.tariff.status,
            tariff_config_hash=self.bundle.tariff.config_hash,
            tariff_period_covered=period_covered and exit_period_covered,
            lineage=lineage,
            warnings=tuple(dict.fromkeys((*warnings, *exit_warnings))),
        )

    def actualize(
        self, fill: Fill, estimate: PreTradeCostEstimate
    ) -> ActualTradingCost:
        if fill.order_id != estimate.order_id or fill.symbol != estimate.symbol:
            raise ValueError("fill and cost estimate lineage do not match")
        request = PreTradeCostRequest(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            # Actual notional-sensitive charges use the realized execution
            # price.  Spread/slippage themselves are injected separately from
            # the Fill so their economic impact is reported, not debited twice.
            reference_price=fill.price,
            timeframe="actual-fill",
            spread_bps=ZERO,
            slippage_bps=ZERO,
            order_id=fill.order_id,
        )
        breakdown, _, _ = self._leg(
            request,
            actual_spread=fill.spread_cost,
            actual_slippage=fill.slippage_cost,
        )
        digest = stable_hash((estimate.estimate_id, fill.fill_id, breakdown))
        return ActualTradingCost(
            actual_cost_id=f"cost-actual-{digest[:24]}",
            estimate_id=estimate.estimate_id,
            order_id=fill.order_id,
            fill_id=fill.fill_id,
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            quantity=fill.quantity,
            reference_price=fill.reference_price,
            execution_price=fill.price,
            breakdown=breakdown,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            config_hash=self.config_hash,
        )

    def reconcile(
        self, estimate: PreTradeCostEstimate, actual: ActualTradingCost
    ) -> CostReconciliation:
        if actual.estimate_id != estimate.estimate_id:
            raise ValueError("actual cost does not reference estimate")
        errors = []
        for name in (
            "commission", "spread", "slippage", "exchange_fees",
            "transaction_tax", "fx_cost", "financing_cost", "other_variable_cost",
        ):
            estimated = getattr(estimate.entry_costs, name).amount
            realized = getattr(actual.breakdown, name).amount
            errors.append((name, realized - estimated if estimated is not None and realized is not None else None))
        estimated_total = estimate.entry_costs.amount_if_complete
        actual_total = actual.breakdown.amount_if_complete
        error = (
            actual_total - estimated_total
            if actual_total is not None and estimated_total is not None else None
        )
        digest = stable_hash((estimate.estimate_id, actual.actual_cost_id, errors))
        return CostReconciliation(
            reconciliation_id=f"cost-reconciliation-{digest[:24]}",
            estimate_id=estimate.estimate_id,
            actual_cost_id=actual.actual_cost_id,
            order_id=actual.order_id,
            fill_id=actual.fill_id,
            timestamp=actual.timestamp,
            estimated_total=estimated_total,
            actual_total=actual_total,
            estimate_error=error,
            component_errors=tuple(sorted(errors)),
            coverage=(
                CostCoverage.COMPLETE
                if estimated_total is not None and actual_total is not None
                else CostCoverage.INCOMPLETE
            ),
        )
