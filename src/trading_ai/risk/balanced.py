"""Balanced Risk Engine 1.0: deterministic limits for offline simulation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_ai.core.models import (
    OrderRequest,
    OrderSide,
    PortfolioSnapshot,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
    TradingProfile,
    TradingProfileName,
)
from trading_ai.risk.base import RiskEngine
from trading_ai.risk.circuit_breaker import RiskCircuitBreaker
from trading_ai.risk.concentration import RiskGroupResolver
from trading_ai.risk.config import (
    BalancedRiskConfig,
    RiskAssetGroups,
    load_balanced_risk_config,
    risk_config_hash,
)
from trading_ai.risk.correlation import CorrelationGuard
from trading_ai.risk.exposure import (
    ExposureProjection,
    max_increment_by_portfolio_exposure,
    max_increment_by_position_exposure,
    position_values,
    project_exposure,
)
from trading_ai.risk.models import (
    CircuitBreakerReason,
    RiskContext,
    RiskReasonCode,
    RiskState,
    RiskStateSnapshot,
    RiskStateTransition,
    RiskSummary,
    UnknownRiskPolicy,
)
from trading_ai.risk.reporting import summarize_risk
from trading_ai.risk.sizing import capped_quantity, quantity_by_trade_risk
from trading_ai.risk.state import RiskStateTracker
from trading_ai.risk.volatility import VolatilityGuard


ZERO = Decimal("0")


_HUMAN_REASONS: dict[RiskReasonCode, str] = {
    RiskReasonCode.APPROVED: "All configured Balanced risk limits permit the request.",
    RiskReasonCode.POSITION_LIMIT: "The requested position exceeds the single-position exposure limit.",
    RiskReasonCode.PORTFOLIO_EXPOSURE_LIMIT: "The request exceeds the portfolio gross-exposure limit.",
    RiskReasonCode.MAX_POSITIONS: "The request would exceed the maximum number of open positions.",
    RiskReasonCode.INSUFFICIENT_CASH: "Available cash is insufficient for the requested quantity without leverage.",
    RiskReasonCode.SHORT_NOT_ALLOWED: "Balanced is long-only; the sale cannot exceed the held quantity.",
    RiskReasonCode.DAILY_LOSS_LIMIT: "The daily loss circuit breaker blocks new risk.",
    RiskReasonCode.SOFT_DRAWDOWN: "Soft drawdown protection reduces new risk.",
    RiskReasonCode.HARD_DRAWDOWN: "The persistent hard-drawdown circuit breaker blocks new risk.",
    RiskReasonCode.VOLATILITY_LIMIT: "Observed Feature Engine volatility reduces or blocks new risk.",
    RiskReasonCode.VOLATILITY_UNKNOWN: "The configured volatility feature is not currently available.",
    RiskReasonCode.CORRELATION_LIMIT: "Highly correlated portfolio exposure limits this request.",
    RiskReasonCode.CORRELATION_UNKNOWN: "There are insufficient exact-common return observations for correlation.",
    RiskReasonCode.CONCENTRATION_LIMIT: "The configured asset-group concentration limit applies.",
    RiskReasonCode.UNKNOWN_GROUP: "The symbol has no configured concentration group.",
    RiskReasonCode.INVALID_RISK_CONTEXT: "The point-in-time risk context is invalid or incomplete.",
    RiskReasonCode.NO_EXPLICIT_RISK_DISTANCE: "No stop or invalidation was supplied; no precise trade-loss claim is made.",
    RiskReasonCode.TRADE_RISK_LIMIT: "Explicit invalidation distance limits the quantity by trade-risk budget.",
    RiskReasonCode.CIRCUIT_BREAKER_ACTIVE: "A circuit breaker blocks new or increased risk.",
    RiskReasonCode.RISK_REDUCING_ORDER: "The order reduces an existing long position.",
    RiskReasonCode.REDUCED_RISK_STATE: "The system is in reduced-risk state.",
}


class BalancedRiskEngine(RiskEngine):
    """Final size authority for Balanced simulation; never a signal source."""

    def __init__(
        self,
        profile: TradingProfile,
        config: BalancedRiskConfig,
        groups: RiskAssetGroups,
    ) -> None:
        if profile.name is not TradingProfileName.BALANCED or not profile.enabled:
            raise ValueError("BalancedRiskEngine requires the enabled Balanced profile")
        if profile.allow_short:
            raise ValueError("BalancedRiskEngine requires a long-only profile")
        if config.name is not profile.name or not config.enabled:
            raise ValueError("risk configuration must be enabled and match Balanced")
        if config.max_positions > profile.max_positions:
            raise ValueError("risk max_positions exceeds the trading profile")
        if config.max_portfolio_exposure > Decimal(str(profile.max_exposure)):
            raise ValueError("risk exposure exceeds the trading profile")
        if config.max_trade_risk_fraction > Decimal(str(profile.risk_budget)):
            raise ValueError("risk trade budget exceeds the trading profile")
        self.profile = profile
        self.config = config
        self.groups = groups
        self._config_hash = risk_config_hash(config, groups)
        self._state_tracker = RiskStateTracker(config)
        self.circuit_breaker = RiskCircuitBreaker(self._state_tracker)
        self._group_resolver = RiskGroupResolver(groups)
        self._correlation_guard = CorrelationGuard(
            threshold=float(config.high_correlation_threshold),
            minimum_observations=config.correlation_min_observations,
        )
        self._volatility_guard = VolatilityGuard(config)

    @classmethod
    def from_profile(cls, profile: TradingProfile) -> BalancedRiskEngine:
        config, groups = load_balanced_risk_config(profile)
        return cls(profile, config, groups)

    @property
    def engine_name(self) -> str:
        return self.config.engine_name

    @property
    def engine_version(self) -> str:
        return self.config.engine_version

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        group_parameters = tuple(
            (f"asset_group.{name}", ",".join(symbols))
            for name, symbols in self.groups.groups
        )
        return tuple(sorted((*self.config.to_parameters(), *group_parameters)))

    @property
    def state_snapshot(self) -> RiskStateSnapshot:
        return self._state_tracker.snapshot

    @property
    def state_transitions(self) -> tuple[RiskStateTransition, ...]:
        return self._state_tracker.transitions

    def reset(self, timestamp: datetime, equity: Decimal) -> None:
        self._state_tracker.reset(timestamp, equity)

    def observe(self, timestamp: datetime, equity: Decimal) -> None:
        self._state_tracker.observe(timestamp, equity)

    def current_state(
        self, timestamp: datetime, equity: Decimal
    ) -> RiskStateSnapshot:
        return self._state_tracker.observe(timestamp, equity)

    def halt(
        self,
        reason: CircuitBreakerReason,
        timestamp: datetime,
        equity: Decimal,
    ) -> RiskStateSnapshot:
        return self.circuit_breaker.halt(reason, timestamp, equity)

    def reset_halt(
        self,
        timestamp: datetime,
        equity: Decimal,
        *,
        authorization_reason: str,
    ) -> RiskStateSnapshot:
        return self.circuit_breaker.reset(
            timestamp,
            equity,
            authorization_reason=authorization_reason,
        )

    def evaluate(
        self,
        order: OrderRequest,
        portfolio: PortfolioSnapshot,
        context: TradingContext,
    ) -> RiskDecision:
        """Generic execution lacks market/risk inputs and therefore fails closed."""

        del context
        return self._decision(
            order=order,
            timestamp=order.created_at or portfolio.as_of,
            status=RiskDecisionStatus.REJECT,
            approved=ZERO,
            codes=(RiskReasonCode.INVALID_RISK_CONTEXT,),
            state=RiskState.HALTED,
            equity=portfolio.total_equity,
            cash=portfolio.cash,
        )

    def evaluate_context(self, context: RiskContext) -> RiskDecision:
        """Apply point-in-time limits and return one immutable final decision."""

        try:
            state = self._state_tracker.observe(context.timestamp, context.equity)
            if context.profile != self.profile:
                raise ValueError("risk context profile differs from configured profile")
            if context.order.symbol not in self.profile.asset_universe:
                raise ValueError("risk order symbol is outside configured universe")
            if context.equity <= ZERO:
                raise ValueError("positive equity is required for new risk")
            if context.cash < ZERO or any(
                position.quantity < ZERO for position in context.portfolio.positions
            ):
                raise ValueError("Balanced context cannot contain debt or short positions")
            if context.order.side is OrderSide.SELL:
                return self._evaluate_sell(context, state)
            return self._evaluate_buy(context, state)
        except (ArithmeticError, KeyError, TypeError, ValueError):
            return self._decision(
                order=context.order,
                timestamp=context.timestamp,
                status=RiskDecisionStatus.REJECT,
                approved=ZERO,
                codes=(RiskReasonCode.INVALID_RISK_CONTEXT,),
                state=RiskState.HALTED,
                equity=context.portfolio.total_equity,
                cash=context.portfolio.cash,
            )

    def summary(
        self,
        decisions: tuple[RiskDecision, ...],
        completed_at: datetime,
    ) -> RiskSummary:
        return summarize_risk(
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            config_hash=self.config_hash,
            decisions=decisions,
            tracker=self._state_tracker,
            completed_at=completed_at,
        )

    def _evaluate_sell(
        self, context: RiskContext, state: RiskStateSnapshot
    ) -> RiskDecision:
        held = context.available_quantity_for_exit(context.order.symbol)
        if held <= ZERO:
            return self._decision_from_context(
                context,
                RiskDecisionStatus.REJECT,
                ZERO,
                (RiskReasonCode.SHORT_NOT_ALLOWED,),
                state,
            )
        approved = min(context.order.quantity, held)
        codes = [RiskReasonCode.RISK_REDUCING_ORDER]
        status = RiskDecisionStatus.APPROVE
        if approved < context.order.quantity:
            status = RiskDecisionStatus.REDUCE
            codes.append(RiskReasonCode.SHORT_NOT_ALLOWED)
        projection = project_exposure(
            portfolio=context.portfolio,
            market_prices=context.market_prices,
            symbol=context.order.symbol,
            side=OrderSide.SELL,
            quantity=approved,
            expected_price=context.expected_entry_price,
        )
        return self._decision_from_context(
            context, status, approved, tuple(codes), state, projection=projection
        )

    def _evaluate_buy(
        self, context: RiskContext, state: RiskStateSnapshot
    ) -> RiskDecision:
        order = context.order
        position = context.position_for(order.symbol)
        held = position.quantity if position is not None else ZERO
        if state.state is RiskState.HALTED:
            state_code = (
                RiskReasonCode.HARD_DRAWDOWN
                if state.halt_reason == CircuitBreakerReason.HARD_DRAWDOWN.value
                else RiskReasonCode.DAILY_LOSS_LIMIT
                if state.halt_reason == CircuitBreakerReason.DAILY_LOSS_LIMIT.value
                else RiskReasonCode.CIRCUIT_BREAKER_ACTIVE
            )
            return self._decision_from_context(
                context,
                RiskDecisionStatus.REJECT,
                ZERO,
                (state_code, RiskReasonCode.CIRCUIT_BREAKER_ACTIVE),
                state,
            )
        if held <= ZERO and len(context.open_positions) >= self.config.max_positions:
            return self._decision_from_context(
                context,
                RiskDecisionStatus.REJECT,
                ZERO,
                (RiskReasonCode.MAX_POSITIONS,),
                state,
            )

        group = self._group_resolver.resolve(order.symbol)
        if group is None and self.config.unknown_group_policy is UnknownRiskPolicy.REJECT:
            return self._decision_from_context(
                context,
                RiskDecisionStatus.REJECT,
                ZERO,
                (RiskReasonCode.UNKNOWN_GROUP,),
                state,
            )

        values = position_values(context.portfolio, context.market_prices)
        initial_projection = project_exposure(
            portfolio=context.portfolio,
            market_prices=context.market_prices,
            symbol=order.symbol,
            side=OrderSide.BUY,
            quantity=ZERO,
            expected_price=context.expected_entry_price,
        )
        caps: list[Decimal] = []
        codes: list[RiskReasonCode] = []

        if state.state is RiskState.REDUCED:
            caps.append(order.quantity * self.config.reduced_risk_multiplier)
            codes.extend(
                (RiskReasonCode.SOFT_DRAWDOWN, RiskReasonCode.REDUCED_RISK_STATE)
            )

        volatility = self._volatility_guard.assess(
            context.feature_snapshot,
            timeframe=context.feature_snapshot.timeframe
            if context.feature_snapshot is not None
            else self._infer_timeframe(context),
        )
        if volatility.reason_code is RiskReasonCode.VOLATILITY_UNKNOWN:
            codes.append(RiskReasonCode.VOLATILITY_UNKNOWN)
            if self.config.missing_volatility_policy is UnknownRiskPolicy.REJECT:
                return self._decision_from_context(
                    context,
                    RiskDecisionStatus.REJECT,
                    ZERO,
                    tuple(codes),
                    state,
                    volatility_metric=volatility.metric,
                )
        elif volatility.multiplier < Decimal("1"):
            caps.append(order.quantity * volatility.multiplier)
            codes.append(RiskReasonCode.VOLATILITY_LIMIT)

        # When an explicitly injected TransactionCostEngine supplied a complete
        # point-in-time estimate, reserve notional + entry costs + buffer per
        # unit. Legacy/no-cost contexts retain the historical price-only cap.
        if order.estimated_cash_requirement is not None:
            # Keep every point-in-time estimated order cost and buffer reserved
            # while reducing notional. This is deliberately conservative for
            # minimum-per-order commissions and can never create negative cash.
            requested_notional = order.quantity * context.expected_entry_price
            non_notional_reserve = max(
                Decimal("0"),
                order.estimated_cash_requirement - requested_notional,
            )
            cash_cap = max(
                Decimal("0"),
                (context.cash - non_notional_reserve)
                / context.expected_entry_price,
            )
        else:
            cash_per_unit = (
                order.estimated_unit_cash_requirement
                or context.expected_entry_price
            )
            cash_cap = context.cash / cash_per_unit
        caps.append(cash_cap)
        if cash_cap < order.quantity:
            codes.append(RiskReasonCode.INSUFFICIENT_CASH)

        portfolio_cap = max_increment_by_portfolio_exposure(
            equity=context.equity,
            gross_value_before=initial_projection.gross_value_before,
            limit=self.config.max_portfolio_exposure,
            price=context.expected_entry_price,
        )
        caps.append(portfolio_cap)
        if portfolio_cap < order.quantity:
            codes.append(RiskReasonCode.PORTFOLIO_EXPOSURE_LIMIT)

        position_cap = max_increment_by_position_exposure(
            equity=context.equity,
            position_value_before=initial_projection.position_value_before,
            limit=self.config.max_single_position_exposure,
            price=context.expected_entry_price,
        )
        caps.append(position_cap)
        if position_cap < order.quantity:
            codes.append(RiskReasonCode.POSITION_LIMIT)

        if group is None:
            codes.append(RiskReasonCode.UNKNOWN_GROUP)
        else:
            group_cap = self._group_resolver.max_increment(
                group=group,
                position_values=values,
                equity=context.equity,
                limit=self.config.max_group_exposure,
                expected_price=context.expected_entry_price,
            )
            caps.append(group_cap)
            if group_cap < order.quantity:
                codes.append(RiskReasonCode.CONCENTRATION_LIMIT)

        open_symbols = tuple(position.symbol for position in context.open_positions)
        assessments = self._correlation_guard.assess(
            order.symbol, open_symbols, context.return_series
        )
        unknown_correlation = any(item.coefficient is None for item in assessments)
        if unknown_correlation:
            codes.append(RiskReasonCode.CORRELATION_UNKNOWN)
            if self.config.correlation_unknown_policy is UnknownRiskPolicy.REJECT:
                return self._decision_from_context(
                    context,
                    RiskDecisionStatus.REJECT,
                    ZERO,
                    tuple(codes),
                    state,
                    volatility_metric=volatility.metric,
                )
        correlated = tuple(item for item in assessments if item.highly_correlated)
        if correlated:
            correlated_symbols = {item.symbol for item in correlated}
            correlated_value = values.get(order.symbol, ZERO) + sum(
                (
                    value
                    for symbol, value in values.items()
                    if symbol in correlated_symbols
                ),
                ZERO,
            )
            correlation_cap = max(
                ZERO,
                context.equity * self.config.max_highly_correlated_exposure
                - correlated_value,
            ) / context.expected_entry_price
            caps.append(correlation_cap)
            if correlation_cap < order.quantity:
                codes.append(RiskReasonCode.CORRELATION_LIMIT)

        risk_per_share = order.risk_distance
        if risk_per_share is None and order.invalidation_price is not None:
            risk_per_share = abs(
                context.expected_entry_price - order.invalidation_price
            )
        if risk_per_share is None:
            codes.append(RiskReasonCode.NO_EXPLICIT_RISK_DISTANCE)
        elif risk_per_share <= ZERO:
            return self._decision_from_context(
                context,
                RiskDecisionStatus.REJECT,
                ZERO,
                (RiskReasonCode.INVALID_RISK_CONTEXT,),
                state,
            )
        else:
            trade_risk_cap = quantity_by_trade_risk(
                equity=context.equity,
                risk_fraction=self.config.max_trade_risk_fraction,
                risk_per_share=risk_per_share,
            )
            caps.append(trade_risk_cap)
            if trade_risk_cap < order.quantity:
                codes.append(RiskReasonCode.TRADE_RISK_LIMIT)

        approved = capped_quantity(
            order.quantity,
            tuple(caps),
            step=self.config.quantity_step,
        )
        projection = project_exposure(
            portfolio=context.portfolio,
            market_prices=context.market_prices,
            symbol=order.symbol,
            side=OrderSide.BUY,
            quantity=approved,
            expected_price=context.expected_entry_price,
        )
        unique_codes = tuple(dict.fromkeys(codes))
        if approved <= ZERO:
            status = RiskDecisionStatus.REJECT
        elif approved < order.quantity:
            status = RiskDecisionStatus.REDUCE
        else:
            status = RiskDecisionStatus.APPROVE
            unique_codes = tuple(dict.fromkeys((RiskReasonCode.APPROVED, *unique_codes)))
        correlation_metric = max(
            (
                item.coefficient
                for item in assessments
                if item.coefficient is not None
            ),
            default=None,
        )
        return self._decision_from_context(
            context,
            status,
            approved,
            unique_codes,
            state,
            projection=projection,
            volatility_metric=volatility.metric,
            correlation_metric=correlation_metric,
        )

    @staticmethod
    def _infer_timeframe(context: RiskContext) -> str:
        return context.timeframe

    def _decision_from_context(
        self,
        context: RiskContext,
        status: RiskDecisionStatus,
        approved: Decimal,
        codes: tuple[RiskReasonCode, ...],
        state: RiskStateSnapshot,
        *,
        projection: ExposureProjection | None = None,
        volatility_metric: float | None = None,
        correlation_metric: float | None = None,
    ) -> RiskDecision:
        if projection is None:
            try:
                projection = project_exposure(
                    portfolio=context.portfolio,
                    market_prices=context.market_prices,
                    symbol=context.order.symbol,
                    side=context.order.side,
                    quantity=approved,
                    expected_price=context.expected_entry_price,
                )
            except ValueError:
                projection = None
        return self._decision(
            order=context.order,
            timestamp=context.timestamp,
            status=status,
            approved=approved,
            codes=codes,
            state=state.state,
            equity=context.equity,
            cash=context.cash,
            state_snapshot=state,
            projection=projection,
            volatility_metric=volatility_metric,
            correlation_metric=correlation_metric,
        )

    def _decision(
        self,
        *,
        order: OrderRequest,
        timestamp: datetime,
        status: RiskDecisionStatus,
        approved: Decimal,
        codes: tuple[RiskReasonCode, ...],
        state: RiskState,
        equity: Decimal,
        cash: Decimal,
        state_snapshot: RiskStateSnapshot | None = None,
        projection: ExposureProjection | None = None,
        volatility_metric: float | None = None,
        correlation_metric: float | None = None,
    ) -> RiskDecision:
        unique_codes = tuple(dict.fromkeys(codes))
        reasons = tuple(_HUMAN_REASONS.get(code, code.value) for code in unique_codes)
        reason = "; ".join(reasons) or "Balanced risk decision"
        return RiskDecision(
            decision_id=f"balanced-risk:{order.order_id}",
            order_id=order.order_id,
            status=status,
            reason=reason,
            risk_engine=self.engine_name,
            timestamp=timestamp,
            engine_version=self.engine_version,
            requested_quantity=order.quantity,
            approved_quantity=approved,
            reason_codes=tuple(code.value for code in unique_codes),
            human_readable_reasons=reasons,
            risk_state=state.value,
            config_hash=self.config_hash,
            equity=equity,
            cash=cash,
            gross_exposure_before=(
                projection.gross_before(equity) if projection is not None else None
            ),
            gross_exposure_after=(
                projection.gross_after(equity) if projection is not None else None
            ),
            position_exposure_before=(
                projection.position_before(equity) if projection is not None else None
            ),
            position_exposure_after=(
                projection.position_after(equity) if projection is not None else None
            ),
            daily_loss_pct=(
                state_snapshot.daily_loss_pct if state_snapshot is not None else None
            ),
            drawdown_pct=(
                state_snapshot.drawdown_pct if state_snapshot is not None else None
            ),
            volatility_metric=volatility_metric,
            correlation_metric=correlation_metric,
        )
