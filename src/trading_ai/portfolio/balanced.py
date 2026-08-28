"""Deterministic Balanced multi-strategy target allocation and netting."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

from trading_ai.backtesting.reproducibility import stable_hash
from trading_ai.core.models import OrderSide, TradingProfile
from trading_ai.portfolio.base import PortfolioEngine
from trading_ai.portfolio.config import (
    AssetCurrencyMap,
    BalancedPortfolioConfig,
    load_balanced_portfolio_config,
    portfolio_config_hash,
)
from trading_ai.portfolio.currency import CurrencyConverter, SameCurrencyConverter
from trading_ai.portfolio.diversification import PortfolioDiversification
from trading_ai.portfolio.exceptions import PortfolioPlanningError
from trading_ai.portfolio.models import (
    ONE,
    ZERO,
    PendingPortfolioOrder,
    PortfolioAction,
    PortfolioContext,
    PortfolioDecision,
    PortfolioDecisionBatch,
    PortfolioDecisionStatus,
    PortfolioOpportunity,
    PortfolioOrderProposal,
    PortfolioPlanResult,
    PortfolioTarget,
    RebalancePlan,
    SleeveContribution,
    StrategySleeveState,
    UnknownCorrelationPolicy,
)
from trading_ai.risk.config import BalancedRiskConfig, load_balanced_risk_config


class BalancedPortfolioEngine(PortfolioEngine):
    """Fixed sleeve budgets with soft diversification and hard Risk downstream."""

    def __init__(
        self,
        config: BalancedPortfolioConfig,
        currencies: AssetCurrencyMap,
        *,
        asset_groups: tuple[tuple[str, str], ...] = (),
        currency_converter: CurrencyConverter | None = None,
    ) -> None:
        if not config.enabled:
            raise PortfolioPlanningError("disabled portfolio config cannot be activated")
        self._config = config
        self._currencies = currencies
        self._asset_groups = tuple(sorted(asset_groups))
        self._currency_converter = currency_converter or SameCurrencyConverter()
        self._config_hash = portfolio_config_hash(config, currencies)
        self._diversification = PortfolioDiversification(
            float(config.soft_correlation_threshold),
            config.correlation_min_observations,
        )
        self._sleeve_state: tuple[StrategySleeveState, ...] = ()
        self._initialized = False

    @classmethod
    def from_profile(
        cls,
        profile: TradingProfile,
        *,
        risk_config: BalancedRiskConfig | None = None,
        currency_converter: CurrencyConverter | None = None,
    ) -> "BalancedPortfolioEngine":
        loaded_risk, groups = load_balanced_risk_config(profile)
        bounded_risk = risk_config or loaded_risk
        config, currencies = load_balanced_portfolio_config(profile, bounded_risk)
        return cls(
            config,
            currencies,
            asset_groups=groups.symbol_mapping,
            currency_converter=currency_converter,
        )

    @property
    def engine_name(self) -> str:
        return self._config.engine_name

    @property
    def engine_version(self) -> str:
        return self._config.engine_version

    @property
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        return self._config.to_parameters()

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def sleeve_state(self) -> tuple[StrategySleeveState, ...]:
        return self._sleeve_state

    @property
    def config(self) -> BalancedPortfolioConfig:
        return self._config

    @property
    def currencies(self) -> AssetCurrencyMap:
        return self._currencies

    @property
    def asset_groups(self) -> tuple[tuple[str, str], ...]:
        return self._asset_groups

    def reset(self, timestamp: datetime, equity: Decimal) -> None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise PortfolioPlanningError("portfolio reset timestamp must be aware")
        if equity <= ZERO:
            raise PortfolioPlanningError("portfolio reset equity must be positive")
        self._sleeve_state = ()
        self._initialized = True

    @staticmethod
    def _rank(
        opportunities: tuple[PortfolioOpportunity, ...],
    ) -> tuple[PortfolioOpportunity, ...]:
        ranked: list[PortfolioOpportunity] = []
        strategies = sorted({item.strategy_name for item in opportunities})
        for strategy_name in strategies:
            group = sorted(
                (item for item in opportunities if item.strategy_name == strategy_name),
                key=lambda item: (
                    0 if item.action is PortfolioAction.EXIT_LONG else 1,
                    -item.signal_strength,
                    item.symbol,
                    item.signal_id,
                    item.opportunity_id,
                ),
            )
            entries = [item for item in group if item.action is PortfolioAction.ENTER_LONG]
            strength_levels = sorted(
                {item.signal_strength for item in entries}, reverse=True
            )
            strength_rank = {
                strength: index for index, strength in enumerate(strength_levels)
            }
            denominator = max(1, len(strength_levels) - 1)
            for item in group:
                percentile = (
                    1.0
                    if item.action is PortfolioAction.EXIT_LONG or len(strength_levels) == 1
                    else 1.0 - strength_rank[item.signal_strength] / denominator
                )
                ranked.append(replace(item, rank_percentile=percentile))
        return tuple(
            sorted(
                ranked,
                key=lambda item: (
                    item.timestamp,
                    item.strategy_name,
                    0 if item.action is PortfolioAction.EXIT_LONG else 1,
                    -(item.rank_percentile or 0.0),
                    item.symbol,
                    item.signal_id,
                ),
            )
        )

    def plan(
        self,
        batch: PortfolioDecisionBatch,
        context: PortfolioContext,
    ) -> PortfolioPlanResult:
        if not self._initialized:
            raise PortfolioPlanningError("portfolio engine must be reset before planning")
        if batch.timestamp != context.timestamp or batch.opportunities != context.opportunities:
            raise PortfolioPlanningError("batch and context must describe the exact same cycle")
        if context.equity <= ZERO:
            raise PortfolioPlanningError("portfolio equity must be positive")

        ranked = self._rank(batch.opportunities)
        states = {
            (item.strategy_name, item.symbol): item for item in self._sleeve_state
        }
        prior_states = dict(states)
        decisions: dict[str, PortfolioDecision] = {}

        # Exits are applied first and are exempt from ranking, diversification, and turnover.
        for opportunity in (item for item in ranked if item.action is PortfolioAction.EXIT_LONG):
            key = (opportunity.strategy_name, opportunity.symbol)
            previous = states.pop(key, None)
            status = (
                PortfolioDecisionStatus.EXIT
                if previous is not None
                else PortfolioDecisionStatus.NO_CHANGE
            )
            decisions[opportunity.opportunity_id] = self._decision(
                opportunity,
                status,
                previous.target_weight_contribution if previous else ZERO,
                ZERO,
                "SLEEVE_EXIT" if previous else "SLEEVE_NOT_ACTIVE",
                "Removed only this strategy sleeve contribution."
                if previous
                else "No active sleeve contribution required removal.",
            )

        candidates = [item for item in ranked if item.action is PortfolioAction.ENTER_LONG]
        selected_symbols = {
            item.symbol for item in states.values()
        } | {
            item.symbol
            for item in context.portfolio.positions
            if item.quantity > ZERO
        }
        while candidates:
            best_rank = max(item.rank_percentile or 0.0 for item in candidates)
            comparable = [
                item for item in candidates if (item.rank_percentile or 0.0) == best_rank
            ]

            def candidate_key(item: PortfolioOpportunity) -> tuple[object, ...]:
                assessment = self._diversification.assess(
                    item.symbol,
                    tuple(sorted(selected_symbols)),
                    context.return_series,
                    context.asset_groups,
                )
                return (
                    *self._diversification.sort_key(
                        assessment,
                        deprioritize_unknown=(
                            self._config.unknown_correlation_policy
                            is UnknownCorrelationPolicy.DEPRIORITIZE
                        ),
                    ),
                    item.strategy_name,
                    item.symbol,
                    item.signal_id,
                )

            opportunity = min(comparable, key=candidate_key)
            candidates.remove(opportunity)
            key = (opportunity.strategy_name, opportunity.symbol)
            previous = states.get(key)
            if previous is not None:
                decisions[opportunity.opportunity_id] = self._decision(
                    opportunity,
                    PortfolioDecisionStatus.NO_CHANGE,
                    previous.target_weight_contribution,
                    previous.target_weight_contribution,
                    "EXISTING_SLEEVE",
                    "Existing sleeve persists until an explicit exit.",
                )
                continue
            sleeve = self._config.sleeve_for(opportunity.strategy_name)
            if sleeve is None:
                decisions[opportunity.opportunity_id] = self._decision(
                    opportunity,
                    PortfolioDecisionStatus.REJECT,
                    ZERO,
                    ZERO,
                    "UNKNOWN_STRATEGY_SLEEVE",
                    "No configured sleeve budget exists for this strategy.",
                )
                continue
            currency = dict(context.asset_currencies).get(opportunity.symbol)
            if currency is None:
                decisions[opportunity.opportunity_id] = self._decision(
                    opportunity,
                    PortfolioDecisionStatus.REJECT,
                    ZERO,
                    ZERO,
                    "UNKNOWN_CURRENCY",
                    "Unknown asset currency is fail-closed for new allocation.",
                )
                continue
            if not self._currency_converter.has_rate(
                currency, self._config.base_currency, context.timestamp
            ):
                decisions[opportunity.opportunity_id] = self._decision(
                    opportunity,
                    PortfolioDecisionStatus.REJECT,
                    ZERO,
                    ZERO,
                    "FX_RATE_UNAVAILABLE",
                    "A point-in-time FX rate is required for mixed-currency allocation.",
                )
                continue
            if (
                opportunity.symbol not in selected_symbols
                and len(selected_symbols) >= self._config.max_unique_positions
            ):
                decisions[opportunity.opportunity_id] = self._decision(
                    opportunity,
                    PortfolioDecisionStatus.DEFER,
                    ZERO,
                    ZERO,
                    "MAX_UNIQUE_POSITIONS",
                    "No deterministic position slot is currently available.",
                )
                continue
            assessment = self._diversification.assess(
                opportunity.symbol,
                tuple(sorted(selected_symbols)),
                context.return_series,
                context.asset_groups,
            )
            if (
                assessment.correlation_unknown
                and self._config.unknown_correlation_policy
                is UnknownCorrelationPolicy.REJECT
            ):
                decisions[opportunity.opportunity_id] = self._decision(
                    opportunity,
                    PortfolioDecisionStatus.REJECT,
                    ZERO,
                    ZERO,
                    "CORRELATION_UNKNOWN",
                    "Correlation is unavailable under the configured reject policy.",
                )
                continue
            states[key] = StrategySleeveState(
                strategy_name=opportunity.strategy_name,
                strategy_version=opportunity.strategy_version,
                symbol=opportunity.symbol,
                target_weight_contribution=ZERO,
                entered_at=opportunity.timestamp,
                last_updated_at=opportunity.timestamp,
                signal_id=opportunity.signal_id,
                activation_multiplier=opportunity.activation_multiplier,
            )
            selected_symbols.add(opportunity.symbol)
            reason_codes = ["SELECTED"]
            if assessment.group_unknown:
                reason_codes.append("UNKNOWN_GROUP_DEPRIORITIZED")
            if assessment.correlation_unknown:
                reason_codes.append("CORRELATION_UNKNOWN_DEPRIORITIZED")
            elif assessment.high_correlation:
                reason_codes.append("HIGH_CORRELATION_SOFT")
            decisions[opportunity.opportunity_id] = self._decision(
                opportunity,
                PortfolioDecisionStatus.SELECT,
                ZERO,
                ZERO,
                "+".join(reason_codes),
                "Selected by deterministic intra-strategy rank and soft diversification.",
            )

        states = self._allocate_sleeves(states, context.timestamp, prior_states)
        targets = self._targets(states, context)

        # Entry turnover is a construction budget. Remove deferred new symbols and recalculate.
        new_keys = set(states) - set(prior_states)
        deferred_keys: set[tuple[str, str]] = set()
        running_entry_turnover = ZERO
        for target in targets:
            if target.delta_weight is None or target.delta_weight <= ZERO:
                continue
            new_for_symbol = sorted(key for key in new_keys if key[1] == target.symbol)
            if not new_for_symbol:
                continue
            if (
                running_entry_turnover + target.delta_weight
                > self._config.max_entry_turnover_per_cycle
            ):
                deferred_keys.update(new_for_symbol)
            else:
                running_entry_turnover += target.delta_weight
        if deferred_keys:
            for key in deferred_keys:
                state = states.pop(key)
                matching = next(
                    (
                        item
                        for item in ranked
                        if item.strategy_name == key[0]
                        and item.symbol == key[1]
                        and item.action is PortfolioAction.ENTER_LONG
                    ),
                    None,
                )
                if matching is not None:
                    decisions[matching.opportunity_id] = self._decision(
                        matching,
                        PortfolioDecisionStatus.DEFER,
                        ZERO,
                        ZERO,
                        "TURNOVER_BUDGET",
                        "New risk was deferred by the entry-turnover budget.",
                    )
            states = self._allocate_sleeves(states, context.timestamp, prior_states)
            targets = self._targets(states, context)

        opportunity_by_symbol: dict[str, list[PortfolioOpportunity]] = {}
        for item in ranked:
            opportunity_by_symbol.setdefault(item.symbol, []).append(item)
        proposals: list[PortfolioOrderProposal] = []
        deferred_order_symbols: list[str] = []
        planned_turnover = ZERO
        plan_seed = {
            "timestamp": context.timestamp,
            "config_hash": self.config_hash,
            "targets": targets,
            "opportunities": ranked,
        }
        plan_id = f"portfolio-plan-{stable_hash(plan_seed)[:24]}"
        targets = tuple(
            replace(target, portfolio_plan_id=plan_id, timestamp=context.timestamp)
            for target in targets
        )
        for target in targets:
            relevant_opportunities = opportunity_by_symbol.get(target.symbol, [])
            proposal, reason = self._proposal_for_target(
                target,
                context,
                plan_id,
                relevant_opportunities,
                tuple(decisions.values()),
            )
            if proposal is not None:
                proposals.append(proposal)
                if target.delta_weight is not None:
                    planned_turnover += abs(target.delta_weight)
            elif reason is not None:
                if reason == "PENDING_ORDER_CONFLICT":
                    deferred_order_symbols.append(target.symbol)
                status = {
                    "NO_TRADE_BAND": PortfolioDecisionStatus.NO_CHANGE,
                    "PENDING_SAME_DIRECTION": PortfolioDecisionStatus.NO_CHANGE,
                    "PENDING_ORDER_CONFLICT": PortfolioDecisionStatus.DEFER,
                }.get(reason)
                if status is not None:
                    human_reason = {
                        "NO_TRADE_BAND": (
                            "Target delta is below the configured no-trade band."
                        ),
                        "PENDING_SAME_DIRECTION": (
                            "An existing same-direction pending order already represents the target."
                        ),
                        "PENDING_ORDER_CONFLICT": (
                            "An unresolved opposing pending order makes this rebalance unsafe."
                        ),
                    }[reason]
                    for opportunity in relevant_opportunities:
                        if opportunity.action is PortfolioAction.EXIT_LONG:
                            continue
                        before = prior_states.get(
                            (opportunity.strategy_name, opportunity.symbol)
                        )
                        after = states.get(
                            (opportunity.strategy_name, opportunity.symbol)
                        )
                        decisions[opportunity.opportunity_id] = self._decision(
                            opportunity,
                            status,
                            before.target_weight_contribution if before else ZERO,
                            after.target_weight_contribution if after else ZERO,
                            reason,
                            human_reason,
                        )

        target_by_symbol = {item.symbol: item for item in targets}
        final_decisions = []
        for opportunity in ranked:
            decision = decisions[opportunity.opportunity_id]
            target = target_by_symbol.get(opportunity.symbol)
            state = states.get((opportunity.strategy_name, opportunity.symbol))
            before = prior_states.get((opportunity.strategy_name, opportunity.symbol))
            final_decisions.append(
                replace(
                    decision,
                    target_weight_before=self._symbol_weight(
                        prior_states, opportunity.symbol
                    ),
                    target_weight_after=target.target_weight if target else ZERO,
                    sleeve_weight_before=(
                        before.target_weight_contribution if before else ZERO
                    ),
                    sleeve_weight_after=(
                        state.target_weight_contribution if state else ZERO
                    ),
                )
            )
        plan = RebalancePlan(
            plan_id=plan_id,
            timestamp=context.timestamp,
            targets=tuple(targets),
            orders_to_create=tuple(sorted(proposals, key=lambda item: item.symbol)),
            orders_to_defer=tuple(sorted(deferred_order_symbols)),
            portfolio_exposure_before=self._gross_exposure(context),
            target_exposure_after=sum((item.target_weight for item in targets), ZERO),
            planned_turnover=planned_turnover,
            cash_fraction_before=(context.cash / context.equity),
            target_cash_fraction=ONE - sum((item.target_weight for item in targets), ZERO),
            config_hash=self.config_hash,
        )
        self._sleeve_state = tuple(
            sorted(states.values(), key=lambda item: (item.strategy_name, item.symbol))
        )
        return PortfolioPlanResult(
            ranked_opportunities=ranked,
            decisions=tuple(
                sorted(final_decisions, key=lambda item: item.decision_id)
            ),
            plan=plan,
            sleeve_state=self._sleeve_state,
        )

    def _allocate_sleeves(
        self,
        states: dict[tuple[str, str], StrategySleeveState],
        timestamp: datetime,
        prior_states: dict[tuple[str, str], StrategySleeveState],
    ) -> dict[tuple[str, str], StrategySleeveState]:
        weighted: dict[tuple[str, str], StrategySleeveState] = {}
        for sleeve in self._config.strategy_sleeves:
            keys = sorted(key for key in states if key[0] == sleeve.strategy_name)
            if not keys:
                continue
            prior_keys = sorted(
                key for key in prior_states if key[0] == sleeve.strategy_name
            )
            if keys == prior_keys:
                for key in keys:
                    weighted[key] = replace(
                        states[key],
                        target_weight_contribution=(
                            prior_states[key].target_weight_contribution
                        ),
                        last_updated_at=timestamp,
                    )
                continue
            equal_weight = sleeve.budget_weight / Decimal(len(keys))
            for key in keys:
                state = states[key]
                weighted[key] = replace(
                    state,
                    target_weight_contribution=(
                        equal_weight * state.activation_multiplier
                    ),
                    last_updated_at=timestamp,
                )
        by_symbol: dict[str, Decimal] = {}
        for item in weighted.values():
            by_symbol[item.symbol] = by_symbol.get(item.symbol, ZERO) + item.target_weight_contribution
        for symbol, total in by_symbol.items():
            if total <= self._config.max_target_per_symbol:
                continue
            multiplier = self._config.max_target_per_symbol / total
            for key, state in tuple(weighted.items()):
                if state.symbol == symbol:
                    weighted[key] = replace(
                        state,
                        target_weight_contribution=state.target_weight_contribution * multiplier,
                    )
        return weighted

    def _targets(
        self,
        states: dict[tuple[str, str], StrategySleeveState],
        context: PortfolioContext,
    ) -> tuple[PortfolioTarget, ...]:
        prices = dict(context.market_prices)
        currencies = dict(context.asset_currencies)
        groups = dict(context.asset_groups)
        positions = {item.symbol: item for item in context.portfolio.positions}
        symbols = sorted({item.symbol for item in states.values()} | set(positions))
        targets: list[PortfolioTarget] = []
        for symbol in symbols:
            contributions = tuple(
                sorted(
                    (
                        SleeveContribution(
                            item.strategy_name,
                            item.strategy_version,
                            item.target_weight_contribution,
                            item.signal_id,
                        )
                        for item in states.values()
                        if item.symbol == symbol
                    ),
                    key=lambda item: item.strategy_name,
                )
            )
            target_weight = sum((item.weight for item in contributions), ZERO)
            position = positions.get(symbol)
            current_weight: Decimal | None = ZERO
            if position is not None and position.quantity > ZERO:
                currency = currencies.get(symbol)
                price = prices.get(symbol)
                if (
                    currency is None
                    or price is None
                    or not self._currency_converter.has_rate(
                        currency, self._config.base_currency, context.timestamp
                    )
                ):
                    current_weight = None
                else:
                    current_weight = self._currency_converter.convert(
                        position.quantity * price,
                        currency,
                        self._config.base_currency,
                        context.timestamp,
                    ) / context.equity
            delta_weight = (
                target_weight - current_weight if current_weight is not None else None
            )
            targets.append(
                PortfolioTarget(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    delta_weight=delta_weight,
                    contributors=contributions,
                    currency=currencies.get(symbol),
                    group=groups.get(symbol),
                )
            )
        return tuple(targets)

    def _proposal_for_target(
        self,
        target: PortfolioTarget,
        context: PortfolioContext,
        plan_id: str,
        opportunities: list[PortfolioOpportunity],
        decisions: tuple[PortfolioDecision, ...],
    ) -> tuple[PortfolioOrderProposal | None, str | None]:
        positions = {item.symbol: item for item in context.portfolio.positions}
        position = positions.get(target.symbol)
        held = position.quantity if position is not None else ZERO
        price = dict(context.market_prices).get(target.symbol)
        pending = tuple(item for item in context.pending_orders if item.symbol == target.symbol)
        if target.delta_weight is None:
            if target.target_weight == ZERO and held > ZERO:
                side = OrderSide.SELL
                quantity = held
            else:
                return None, "FX_RATE_UNAVAILABLE"
        else:
            delta = target.delta_weight
            if abs(delta) < self._config.min_rebalance_weight:
                return None, "NO_TRADE_BAND"
            side = OrderSide.BUY if delta > ZERO else OrderSide.SELL
            if price is None:
                return None, "PRICE_UNAVAILABLE"
            currency = target.currency
            if currency is None:
                return None, "UNKNOWN_CURRENCY"
            base_price = self._currency_converter.convert(
                price, currency, self._config.base_currency, context.timestamp
            )
            quantity = (
                abs(delta) * context.equity / base_price
            ).quantize(self._config.quantity_step, rounding=ROUND_DOWN)
            if side is OrderSide.SELL:
                quantity = min(quantity, held)
        if quantity < self._config.quantity_step:
            return None, "NO_TRADE_BAND"
        if pending:
            if any(item.side is not side for item in pending):
                return None, "PENDING_ORDER_CONFLICT"
            return None, "PENDING_SAME_DIRECTION"
        relevant = sorted(
            opportunities,
            key=lambda item: (
                0 if item.action is PortfolioAction.EXIT_LONG else 1,
                item.strategy_name,
                item.signal_id,
            ),
        )
        if not relevant:
            # Targets are still reported, but an executable intent must retain
            # exact signal/ML/policy lineage from the current cycle.
            return None, "NO_CURRENT_SIGNAL_LINEAGE"
        signal_id = relevant[0].signal_id
        opportunity_ids = tuple(sorted(item.opportunity_id for item in relevant))
        ml_decision_id = relevant[0].ml_decision_id
        activation_decision_id = relevant[0].activation_decision_id
        linked = next(
            (
                item
                for item in decisions
                if item.opportunity_id in set(opportunity_ids)
            ),
            None,
        )
        decision_id = (
            linked.decision_id
            if linked is not None
            else f"portfolio-rebalance-{stable_hash((plan_id, target.symbol))[:24]}"
        )
        timeframe = relevant[0].timeframe if relevant else "1d"
        return (
            PortfolioOrderProposal(
                symbol=target.symbol,
                side=side,
                quantity=quantity,
                timeframe=timeframe,
                portfolio_plan_id=plan_id,
                portfolio_decision_id=decision_id,
                opportunity_ids=opportunity_ids,
                signal_id=signal_id,
                ml_decision_id=ml_decision_id,
                activation_decision_id=activation_decision_id,
            ),
            None,
        )

    def _decision(
        self,
        opportunity: PortfolioOpportunity,
        status: PortfolioDecisionStatus,
        sleeve_before: Decimal,
        sleeve_after: Decimal,
        reason_code: str,
        human_reason: str,
    ) -> PortfolioDecision:
        return PortfolioDecision(
            decision_id=(
                "portfolio-decision-"
                + stable_hash(
                    (
                        opportunity.opportunity_id,
                        status.value,
                        reason_code,
                        self.config_hash,
                    )
                )[:24]
            ),
            timestamp=opportunity.timestamp,
            opportunity_id=opportunity.opportunity_id,
            status=status,
            target_weight_before=ZERO,
            target_weight_after=ZERO,
            sleeve_weight_before=sleeve_before,
            sleeve_weight_after=sleeve_after,
            reason_codes=tuple(reason_code.split("+")),
            human_reasons=(human_reason,),
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            config_hash=self.config_hash,
            signal_id=opportunity.signal_id,
        )

    @staticmethod
    def _symbol_weight(
        states: dict[tuple[str, str], StrategySleeveState], symbol: str
    ) -> Decimal:
        return sum(
            (
                item.target_weight_contribution
                for item in states.values()
                if item.symbol == symbol
            ),
            ZERO,
        )

    def _gross_exposure(self, context: PortfolioContext) -> Decimal:
        prices = dict(context.market_prices)
        currencies = dict(context.asset_currencies)
        value = ZERO
        for position in context.portfolio.positions:
            currency = currencies.get(position.symbol)
            price = prices.get(position.symbol)
            if currency is None or price is None:
                continue
            if not self._currency_converter.has_rate(
                currency, self._config.base_currency, context.timestamp
            ):
                continue
            value += abs(
                self._currency_converter.convert(
                    position.quantity * price,
                    currency,
                    self._config.base_currency,
                    context.timestamp,
                )
            )
        return value / context.equity
