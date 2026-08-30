"""Immutable contracts for cost estimation, actual costs, and economic gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from trading_ai.core.models import OrderSide


ZERO = Decimal("0")
BPS = Decimal("10000")


def _text(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be normalized to UTC")


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")


class CostStatus(str, Enum):
    KNOWN = "KNOWN"
    ESTIMATED = "ESTIMATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class CostCoverage(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class TariffStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    EXPIRED = "EXPIRED"


class EconomicDecisionStatus(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    INCOMPLETE = "INCOMPLETE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EdgeStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class CostComponent:
    """One cost component whose unknown state can never masquerade as zero."""

    name: str
    status: CostStatus
    amount: Decimal | None
    currency: str
    source: str
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "cost component name")
        _text(self.currency, "cost currency")
        _text(self.source, "cost source")
        if self.status is CostStatus.UNAVAILABLE:
            if self.amount is not None:
                raise ValueError("UNAVAILABLE cost must not contain zero or an amount")
        elif self.status is CostStatus.NOT_APPLICABLE:
            if self.amount != ZERO:
                raise ValueError("NOT_APPLICABLE cost must carry explicit zero")
        elif self.amount is None or not self.amount.is_finite() or self.amount < ZERO:
            raise ValueError("KNOWN/ESTIMATED cost requires a finite non-negative amount")
        if self.reason is not None:
            _text(self.reason, "cost reason")

    @classmethod
    def known(
        cls, name: str, amount: Decimal, currency: str, source: str
    ) -> CostComponent:
        return cls(name, CostStatus.KNOWN, amount, currency, source)

    @classmethod
    def estimated(
        cls,
        name: str,
        amount: Decimal,
        currency: str,
        source: str,
        reason: str | None = None,
    ) -> CostComponent:
        return cls(name, CostStatus.ESTIMATED, amount, currency, source, reason)

    @classmethod
    def not_applicable(
        cls, name: str, currency: str, source: str, reason: str
    ) -> CostComponent:
        return cls(name, CostStatus.NOT_APPLICABLE, ZERO, currency, source, reason)

    @classmethod
    def unavailable(
        cls, name: str, currency: str, source: str, reason: str
    ) -> CostComponent:
        return cls(name, CostStatus.UNAVAILABLE, None, currency, source, reason)


@dataclass(frozen=True, slots=True)
class TradingCostBreakdown:
    commission: CostComponent
    spread: CostComponent
    slippage: CostComponent
    exchange_fees: CostComponent
    transaction_tax: CostComponent
    fx_cost: CostComponent
    financing_cost: CostComponent
    other_variable_cost: CostComponent
    total_variable_cost: CostComponent

    @property
    def components(self) -> tuple[CostComponent, ...]:
        return (
            self.commission,
            self.spread,
            self.slippage,
            self.exchange_fees,
            self.transaction_tax,
            self.fx_cost,
            self.financing_cost,
            self.other_variable_cost,
        )

    @property
    def coverage(self) -> CostCoverage:
        return (
            CostCoverage.INCOMPLETE
            if any(item.status is CostStatus.UNAVAILABLE for item in self.components)
            else CostCoverage.COMPLETE
        )

    @property
    def amount_if_complete(self) -> Decimal | None:
        return (
            self.total_variable_cost.amount
            if self.coverage is CostCoverage.COMPLETE
            else None
        )


@dataclass(frozen=True, slots=True)
class OperatingCostBreakdown:
    period_start: datetime
    period_end: datetime
    market_data_subscription: CostComponent
    server_vps: CostComponent
    software_subscription: CostComponent
    other_fixed_cost: CostComponent
    total_operating_cost: CostComponent

    def __post_init__(self) -> None:
        _utc(self.period_start, "period_start")
        _utc(self.period_end, "period_end")
        if self.period_start >= self.period_end:
            raise ValueError("operating-cost period must be positive")

    @property
    def components(self) -> tuple[CostComponent, ...]:
        return (
            self.market_data_subscription,
            self.server_vps,
            self.software_subscription,
            self.other_fixed_cost,
        )


@dataclass(frozen=True, slots=True)
class CommissionTier:
    up_to_monthly_quantity: Decimal | None
    per_unit: Decimal

    def __post_init__(self) -> None:
        if self.up_to_monthly_quantity is not None and self.up_to_monthly_quantity <= ZERO:
            raise ValueError("commission tier boundary must be positive")
        if self.per_unit < ZERO or not self.per_unit.is_finite():
            raise ValueError("commission per_unit must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TariffProfile:
    profile_id: str
    provider: str
    plan: str
    currency: str
    markets: tuple[str, ...]
    effective_from: datetime
    effective_to: datetime | None
    source_name: str
    source_reference: str
    verified_at: datetime
    status: TariffStatus
    version: str
    fixed_per_order: Decimal
    per_unit: Decimal
    proportional_bps: Decimal
    minimum_per_order: Decimal
    maximum_per_order: Decimal | None
    maximum_notional_fraction: Decimal | None
    tiers: tuple[CommissionTier, ...]
    exchange_fee_status: CostStatus
    exchange_fee_per_unit: Decimal
    exchange_fee_bps: Decimal
    exchange_fees_included_in_commission: bool
    config_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "profile_id", "provider", "plan", "currency", "source_name",
            "source_reference", "version",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.effective_from, "effective_from")
        _utc(self.verified_at, "verified_at")
        if self.effective_to is not None:
            _utc(self.effective_to, "effective_to")
            if self.effective_to <= self.effective_from:
                raise ValueError("effective_to must follow effective_from")
        if tuple(sorted(set(self.markets))) != self.markets or not self.markets:
            raise ValueError("tariff markets must be sorted, unique, and non-empty")
        for field_name in (
            "fixed_per_order", "per_unit", "proportional_bps",
            "minimum_per_order", "exchange_fee_per_unit", "exchange_fee_bps",
        ):
            value = getattr(self, field_name)
            if value < ZERO or not value.is_finite():
                raise ValueError(f"{field_name} must be finite and non-negative")
        for field_name in ("maximum_per_order", "maximum_notional_fraction"):
            value = getattr(self, field_name)
            if value is not None and (value <= ZERO or not value.is_finite()):
                raise ValueError(f"{field_name} must be positive when supplied")
        boundaries = [
            item.up_to_monthly_quantity
            for item in self.tiers
            if item.up_to_monthly_quantity is not None
        ]
        if boundaries != sorted(boundaries) or len(boundaries) != len(set(boundaries)):
            raise ValueError("commission tiers must use increasing unique boundaries")
        if self.tiers and self.tiers[-1].up_to_monthly_quantity is not None:
            raise ValueError("last commission tier must be open-ended")
        _sha256(self.config_hash, "tariff config_hash")

    def covers(self, timestamp: datetime, market: str) -> bool:
        _utc(timestamp, "tariff timestamp")
        return (
            market in self.markets
            and timestamp >= self.effective_from
            and (self.effective_to is None or timestamp < self.effective_to)
        )


@dataclass(frozen=True, slots=True)
class TransactionTaxRule:
    rule_id: str
    name: str
    currency: str
    rate_bps: Decimal
    applicable_side: OrderSide
    effective_from: datetime
    effective_to: datetime | None
    source_name: str
    source_reference: str
    verified_at: datetime
    status: TariffStatus
    config_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "rule_id", "name", "currency", "source_name", "source_reference"
        ):
            _text(getattr(self, field_name), field_name)
        if self.rate_bps < ZERO or not self.rate_bps.is_finite():
            raise ValueError("tax rate must be finite and non-negative")
        _utc(self.effective_from, "effective_from")
        _utc(self.verified_at, "verified_at")
        if self.effective_to is not None:
            _utc(self.effective_to, "effective_to")
        _sha256(self.config_hash, "tax config_hash")

    def covers(self, timestamp: datetime) -> bool:
        return timestamp >= self.effective_from and (
            self.effective_to is None or timestamp < self.effective_to
        )


@dataclass(frozen=True, slots=True)
class InstrumentCostMetadata:
    symbol: str
    market: str
    venue: str
    currency: str
    transaction_tax_applicable: bool | None
    transaction_tax_rule_id: str | None
    metadata_status: TariffStatus
    source_reference: str

    def __post_init__(self) -> None:
        for field_name in ("symbol", "market", "venue", "currency", "source_reference"):
            _text(getattr(self, field_name), field_name)
        if self.transaction_tax_applicable is True and self.transaction_tax_rule_id is None:
            raise ValueError("tax-applicable instrument requires an explicit tax rule")
        if self.transaction_tax_applicable is not True and self.transaction_tax_rule_id is not None:
            raise ValueError("tax rule is valid only for explicitly applicable instruments")


@dataclass(frozen=True, slots=True)
class PreTradeCostRequest:
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    reference_price: Decimal
    timeframe: str
    spread_bps: Decimal
    slippage_bps: Decimal
    order_id: str
    signal_id: str | None = None
    portfolio_plan_id: str | None = None
    monthly_volume_before: Decimal = ZERO

    def __post_init__(self) -> None:
        _utc(self.timestamp, "timestamp")
        for field_name in ("symbol", "timeframe", "order_id"):
            _text(getattr(self, field_name), field_name)
        if self.quantity <= ZERO or not self.quantity.is_finite():
            raise ValueError("quantity must be positive and finite")
        if self.reference_price <= ZERO or not self.reference_price.is_finite():
            raise ValueError("reference_price must be positive and finite")
        for field_name in ("spread_bps", "slippage_bps"):
            value = getattr(self, field_name)
            if value < ZERO or not value.is_finite():
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.signal_id is not None:
            _text(self.signal_id, "signal_id")
        if self.portfolio_plan_id is not None:
            _text(self.portfolio_plan_id, "portfolio_plan_id")
        if self.monthly_volume_before < ZERO or not self.monthly_volume_before.is_finite():
            raise ValueError("monthly_volume_before must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EstimatedCashRequirement:
    notional: Decimal
    estimated_entry_costs: Decimal | None
    cost_buffer: Decimal
    total_cash_required: Decimal | None
    unit_cash_required: Decimal | None
    currency: str
    coverage: CostCoverage

    def __post_init__(self) -> None:
        if self.notional <= ZERO or self.cost_buffer < ZERO:
            raise ValueError("cash requirement notional must be positive and buffer non-negative")
        if self.coverage is CostCoverage.COMPLETE:
            if self.estimated_entry_costs is None or self.total_cash_required is None:
                raise ValueError("complete cash requirement needs all amounts")
            if self.unit_cash_required is None or self.unit_cash_required <= ZERO:
                raise ValueError("complete cash requirement needs positive per-unit amount")
        elif any(
            value is not None
            for value in (self.estimated_entry_costs, self.total_cash_required, self.unit_cash_required)
        ):
            raise ValueError("incomplete cash requirement must not invent amounts")


@dataclass(frozen=True, slots=True)
class PreTradeCostEstimate:
    estimate_id: str
    timestamp: datetime
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    reference_price: Decimal
    entry_costs: TradingCostBreakdown
    estimated_exit_costs: TradingCostBreakdown
    round_trip_costs: TradingCostBreakdown
    cash_requirement: EstimatedCashRequirement
    engine_name: str
    engine_version: str
    config_hash: str
    tariff_profile_id: str
    tariff_status: TariffStatus
    tariff_config_hash: str
    tariff_period_covered: bool
    lineage: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "estimate_id", "order_id", "symbol", "engine_name", "engine_version",
            "tariff_profile_id",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        _sha256(self.config_hash, "cost config_hash")
        _sha256(self.tariff_config_hash, "tariff_config_hash")
        if self.lineage != tuple(sorted(self.lineage)):
            raise ValueError("cost-estimate lineage must be sorted")


@dataclass(frozen=True, slots=True)
class ActualTradingCost:
    actual_cost_id: str
    estimate_id: str
    order_id: str
    fill_id: str
    timestamp: datetime
    symbol: str
    quantity: Decimal
    reference_price: Decimal
    execution_price: Decimal
    breakdown: TradingCostBreakdown
    engine_name: str
    engine_version: str
    config_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "actual_cost_id", "estimate_id", "order_id", "fill_id", "symbol",
            "engine_name", "engine_version",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        _sha256(self.config_hash, "cost config_hash")


@dataclass(frozen=True, slots=True)
class CostReconciliation:
    reconciliation_id: str
    estimate_id: str
    actual_cost_id: str
    order_id: str
    fill_id: str
    timestamp: datetime
    estimated_total: Decimal | None
    actual_total: Decimal | None
    estimate_error: Decimal | None
    component_errors: tuple[tuple[str, Decimal | None], ...]
    coverage: CostCoverage

    def __post_init__(self) -> None:
        for field_name in (
            "reconciliation_id", "estimate_id", "actual_cost_id", "order_id", "fill_id"
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        if self.component_errors != tuple(sorted(self.component_errors)):
            raise ValueError("component reconciliation must be sorted")


@dataclass(frozen=True, slots=True)
class ExpectedEdgeEstimate:
    edge_id: str
    timestamp: datetime
    strategy_name: str
    timeframe: str
    status: EdgeStatus
    expected_gross_edge_bps: Decimal | None
    horizon_bars: int
    sample_count: int
    validation_start: datetime | None
    validation_end: datetime | None
    source: str
    provenance_hash: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("edge_id", "strategy_name", "timeframe", "source"):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        _sha256(self.provenance_hash, "edge provenance_hash")
        if self.status is EdgeStatus.AVAILABLE:
            if self.expected_gross_edge_bps is None or not self.expected_gross_edge_bps.is_finite():
                raise ValueError("available edge requires a finite gross edge")
            if self.sample_count <= 0 or self.validation_start is None or self.validation_end is None:
                raise ValueError("available edge requires historical sample provenance")
        elif self.expected_gross_edge_bps is not None:
            raise ValueError("unavailable edge must not contain a fabricated value")
        for value, name in (
            (self.validation_start, "validation_start"),
            (self.validation_end, "validation_end"),
        ):
            if value is not None:
                _utc(value, name)

    @classmethod
    def unavailable(
        cls, *, timestamp: datetime, strategy_name: str, timeframe: str, reason: str
    ) -> ExpectedEdgeEstimate:
        from trading_ai.core.hashing import stable_hash

        payload = (timestamp, strategy_name, timeframe, reason)
        digest = stable_hash(payload)
        return cls(
            edge_id=f"edge-unavailable-{digest[:24]}",
            timestamp=timestamp,
            strategy_name=strategy_name,
            timeframe=timeframe,
            status=EdgeStatus.UNAVAILABLE,
            expected_gross_edge_bps=None,
            horizon_bars=0,
            sample_count=0,
            validation_start=None,
            validation_end=None,
            source="no validated historical edge estimator",
            provenance_hash=digest,
            warnings=(reason,),
        )


@dataclass(frozen=True, slots=True)
class EconomicDecision:
    decision_id: str
    timestamp: datetime
    order_id: str
    signal_id: str | None
    cost_estimate_id: str
    expected_edge_id: str
    status: EconomicDecisionStatus
    expected_gross_edge_bps: Decimal | None
    estimated_round_trip_cost_bps: Decimal | None
    expected_net_edge_bps: Decimal | None
    edge_to_cost_ratio: Decimal | None
    reason_codes: tuple[str, ...]
    human_reasons: tuple[str, ...]
    gate_name: str
    gate_version: str
    config_hash: str
    allows_new_risk: bool

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id", "order_id", "cost_estimate_id", "expected_edge_id",
            "gate_name", "gate_version",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        _sha256(self.config_hash, "economic gate config_hash")
        if self.signal_id is not None:
            _text(self.signal_id, "signal_id")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("economic reason codes must be unique")


@dataclass(frozen=True, slots=True)
class CostSummary:
    engine_name: str
    engine_version: str
    config_hash: str
    tariff_profile_id: str
    tariff_status: TariffStatus
    estimate_count: int
    actual_count: int
    economic_pass: int
    economic_block: int
    economic_incomplete: int
    total_commission: Decimal | None
    total_spread: Decimal | None
    total_slippage: Decimal | None
    total_exchange_fees: Decimal | None
    total_transaction_tax: Decimal | None
    total_fx_cost: Decimal | None
    total_financing_cost: Decimal | None
    total_other_variable_cost: Decimal | None
    total_variable_cost: Decimal | None
    cost_coverage: CostCoverage
    gross_trading_pnl: Decimal | None
    net_trading_pnl_before_operating: Decimal | None
    operating_costs: Decimal | None
    net_economic_pnl: Decimal | None
    gross_return: float | None
    net_return_before_operating: float | None
    net_economic_return: float | None
    component_statuses: tuple[tuple[str, CostStatus], ...] = ()

    def __post_init__(self) -> None:
        _text(self.engine_name, "engine_name")
        _text(self.engine_version, "engine_version")
        _text(self.tariff_profile_id, "tariff_profile_id")
        _sha256(self.config_hash, "cost config_hash")
        if self.component_statuses != tuple(sorted(self.component_statuses)):
            raise ValueError("component_statuses must be sorted")
