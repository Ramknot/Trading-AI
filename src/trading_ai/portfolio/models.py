"""Immutable Portfolio Engine models and lineage for deterministic allocation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from trading_ai.core.models import OrderSide, PortfolioSnapshot
from trading_ai.features.models import ReturnSeries


ZERO = Decimal("0")
ONE = Decimal("1")


def _text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")


def _weight(value: Decimal, field_name: str, *, signed: bool = False) -> None:
    lower = -ONE if signed else ZERO
    if not value.is_finite() or value < lower or value > ONE:
        interval = "[-1, 1]" if signed else "[0, 1]"
        raise ValueError(f"{field_name} must be finite and in {interval}")


class PortfolioAction(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"


class PortfolioDecisionStatus(str, Enum):
    SELECT = "SELECT"
    DEFER = "DEFER"
    REJECT = "REJECT"
    EXIT = "EXIT"
    NO_CHANGE = "NO_CHANGE"


class UnknownCorrelationPolicy(str, Enum):
    DEPRIORITIZE = "DEPRIORITIZE"
    REJECT = "REJECT"


class MixedCurrencyPolicy(str, Enum):
    REJECT_WITHOUT_FX = "REJECT_WITHOUT_FX"


@dataclass(frozen=True, slots=True)
class StrategySleeve:
    strategy_name: str
    budget_weight: Decimal

    def __post_init__(self) -> None:
        _text(self.strategy_name, "strategy_name")
        if not self.budget_weight.is_finite() or not ZERO < self.budget_weight <= ONE:
            raise ValueError("sleeve budget_weight must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class StrategySleeveState:
    strategy_name: str
    strategy_version: str
    symbol: str
    target_weight_contribution: Decimal
    entered_at: datetime
    last_updated_at: datetime
    signal_id: str
    activation_multiplier: Decimal = ONE

    def __post_init__(self) -> None:
        for field_name in ("strategy_name", "strategy_version", "symbol", "signal_id"):
            _text(getattr(self, field_name), field_name)
        _weight(self.target_weight_contribution, "target_weight_contribution")
        _weight(self.activation_multiplier, "activation_multiplier")
        _utc(self.entered_at, "entered_at")
        _utc(self.last_updated_at, "last_updated_at")
        if self.last_updated_at < self.entered_at:
            raise ValueError("last_updated_at cannot precede entered_at")


@dataclass(frozen=True, slots=True)
class PortfolioOpportunity:
    opportunity_id: str
    timestamp: datetime
    symbol: str
    strategy_name: str
    strategy_version: str
    signal_id: str
    action: PortfolioAction
    signal_strength: float
    ml_mode: str
    ml_prediction_id: str | None
    ml_decision_id: str | None
    activation_decision_id: str
    activation_multiplier: Decimal
    regime_snapshot_id: str
    current_sleeve_weight: Decimal
    reason: str
    rank_percentile: float | None = None
    timeframe: str = "1d"

    def __post_init__(self) -> None:
        for field_name in (
            "opportunity_id", "symbol", "strategy_name", "strategy_version",
            "signal_id", "ml_mode", "activation_decision_id",
            "regime_snapshot_id", "reason", "timeframe",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        if not 0.0 <= self.signal_strength <= 1.0:
            raise ValueError("signal_strength must be in [0, 1]")
        _weight(self.activation_multiplier, "activation_multiplier")
        _weight(self.current_sleeve_weight, "current_sleeve_weight")
        if self.rank_percentile is not None and not 0.0 <= self.rank_percentile <= 1.0:
            raise ValueError("rank_percentile must be in [0, 1]")
        if self.ml_mode not in {"DISABLED", "SCORE_ONLY", "FILTER"}:
            raise ValueError("ml_mode must be DISABLED, SCORE_ONLY, or FILTER")
        for field_name in ("ml_prediction_id", "ml_decision_id"):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)


@dataclass(frozen=True, slots=True)
class PortfolioDecisionBatch:
    timestamp: datetime
    opportunities: tuple[PortfolioOpportunity, ...]

    def __post_init__(self) -> None:
        _utc(self.timestamp, "timestamp")
        if any(item.timestamp != self.timestamp for item in self.opportunities):
            raise ValueError("batch opportunities must share one timestamp")
        identifiers = [item.opportunity_id for item in self.opportunities]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("batch opportunity IDs must be unique")


@dataclass(frozen=True, slots=True)
class PendingPortfolioOrder:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    created_at: datetime

    def __post_init__(self) -> None:
        _text(self.order_id, "order_id")
        _text(self.symbol, "symbol")
        _utc(self.created_at, "created_at")
        if not self.quantity.is_finite() or self.quantity <= ZERO:
            raise ValueError("pending quantity must be positive and finite")


@dataclass(frozen=True, slots=True)
class PortfolioContext:
    timestamp: datetime
    portfolio: PortfolioSnapshot
    pending_orders: tuple[PendingPortfolioOrder, ...]
    sleeve_state: tuple[StrategySleeveState, ...]
    opportunities: tuple[PortfolioOpportunity, ...]
    market_prices: tuple[tuple[str, Decimal], ...]
    return_series: tuple[ReturnSeries, ...]
    asset_groups: tuple[tuple[str, str | None], ...]
    asset_currencies: tuple[tuple[str, str | None], ...]
    portfolio_config: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _utc(self.timestamp, "timestamp")
        if self.portfolio.as_of != self.timestamp:
            raise ValueError("portfolio snapshot must use the context timestamp")
        if any(item.timestamp != self.timestamp for item in self.opportunities):
            raise ValueError("context opportunities must be point-in-time")
        if any(item.created_at > self.timestamp for item in self.pending_orders):
            raise ValueError("pending orders cannot come from the future")
        if any(
            observation.timestamp > self.timestamp
            for series in self.return_series
            for observation in series.observations
        ):
            raise ValueError("return series must not contain future observations")
        for name, values in (
            ("market_prices", self.market_prices),
            ("asset_groups", self.asset_groups),
            ("asset_currencies", self.asset_currencies),
        ):
            keys = [key for key, _ in values]
            if keys != sorted(keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} must have unique sorted symbols")
        if tuple(sorted(self.portfolio_config)) != self.portfolio_config:
            raise ValueError("portfolio_config must be deterministically sorted")
        config_names = [name for name, _ in self.portfolio_config]
        if len(config_names) != len(set(config_names)):
            raise ValueError("portfolio_config names must be unique")
        for symbol, price in self.market_prices:
            _text(symbol, "market price symbol")
            if not price.is_finite() or price <= ZERO:
                raise ValueError("market prices must be positive and finite")

    @property
    def equity(self) -> Decimal:
        return self.portfolio.total_equity

    @property
    def cash(self) -> Decimal:
        return self.portfolio.cash


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    decision_id: str
    timestamp: datetime
    opportunity_id: str
    status: PortfolioDecisionStatus
    target_weight_before: Decimal
    target_weight_after: Decimal
    sleeve_weight_before: Decimal
    sleeve_weight_after: Decimal
    reason_codes: tuple[str, ...]
    human_reasons: tuple[str, ...]
    engine_name: str
    engine_version: str
    config_hash: str
    signal_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id", "opportunity_id", "engine_name", "engine_version",
            "config_hash",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        if self.signal_id is not None:
            _text(self.signal_id, "signal_id")
        for field_name in (
            "target_weight_before", "target_weight_after",
            "sleeve_weight_before", "sleeve_weight_after",
        ):
            _weight(getattr(self, field_name), field_name)
        if not self.reason_codes or not self.human_reasons:
            raise ValueError("portfolio decisions require reasons")


@dataclass(frozen=True, slots=True)
class SleeveContribution:
    strategy_name: str
    strategy_version: str
    weight: Decimal
    signal_id: str

    def __post_init__(self) -> None:
        for field_name in ("strategy_name", "strategy_version", "signal_id"):
            _text(getattr(self, field_name), field_name)
        _weight(self.weight, "weight")


@dataclass(frozen=True, slots=True)
class PortfolioTarget:
    symbol: str
    target_weight: Decimal
    current_weight: Decimal | None
    delta_weight: Decimal | None
    contributors: tuple[SleeveContribution, ...]
    currency: str | None = None
    group: str | None = None
    portfolio_plan_id: str | None = None
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.symbol, "symbol")
        _weight(self.target_weight, "target_weight")
        if self.current_weight is not None:
            _weight(self.current_weight, "current_weight")
        if self.delta_weight is not None:
            _weight(self.delta_weight, "delta_weight", signed=True)
        if self.current_weight is None and self.delta_weight is not None:
            raise ValueError("delta_weight needs a known current_weight")
        if self.current_weight is not None and self.delta_weight != (
            self.target_weight - self.current_weight
        ):
            raise ValueError("delta_weight must equal target minus current weight")
        if self.currency is not None:
            _text(self.currency, "currency")
        if self.portfolio_plan_id is not None:
            _text(self.portfolio_plan_id, "portfolio_plan_id")
        if self.timestamp is not None:
            _utc(self.timestamp, "timestamp")
        if (self.portfolio_plan_id is None) != (self.timestamp is None):
            raise ValueError("target plan ID and timestamp must be recorded together")
        contributor_names = [item.strategy_name for item in self.contributors]
        if contributor_names != sorted(contributor_names) or len(
            contributor_names
        ) != len(set(contributor_names)):
            raise ValueError("target contributors must be unique and strategy-sorted")
        if sum((item.weight for item in self.contributors), ZERO) != self.target_weight:
            raise ValueError("target weight must equal sleeve contributions")


@dataclass(frozen=True, slots=True)
class PortfolioOrderProposal:
    symbol: str
    side: OrderSide
    quantity: Decimal
    timeframe: str
    portfolio_plan_id: str
    portfolio_decision_id: str
    opportunity_ids: tuple[str, ...]
    signal_id: str
    ml_decision_id: str | None
    activation_decision_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "symbol", "timeframe", "portfolio_plan_id", "portfolio_decision_id",
            "signal_id", "activation_decision_id",
        ):
            _text(getattr(self, field_name), field_name)
        if not self.quantity.is_finite() or self.quantity <= ZERO:
            raise ValueError("proposal quantity must be positive and finite")
        if not self.opportunity_ids:
            raise ValueError("proposal must reference portfolio opportunities")
        if self.opportunity_ids != tuple(sorted(set(self.opportunity_ids))):
            raise ValueError("proposal opportunity IDs must be unique and sorted")
        if self.ml_decision_id is not None:
            _text(self.ml_decision_id, "ml_decision_id")


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    plan_id: str
    timestamp: datetime
    targets: tuple[PortfolioTarget, ...]
    orders_to_create: tuple[PortfolioOrderProposal, ...]
    orders_to_defer: tuple[str, ...]
    portfolio_exposure_before: Decimal
    target_exposure_after: Decimal
    planned_turnover: Decimal
    cash_fraction_before: Decimal
    target_cash_fraction: Decimal
    config_hash: str

    def __post_init__(self) -> None:
        _text(self.plan_id, "plan_id")
        _text(self.config_hash, "config_hash")
        _utc(self.timestamp, "timestamp")
        for field_name in (
            "portfolio_exposure_before", "target_exposure_after",
            "planned_turnover", "cash_fraction_before", "target_cash_fraction",
        ):
            value = getattr(self, field_name)
            if not value.is_finite() or value < ZERO:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.target_exposure_after > ONE or self.target_cash_fraction > ONE:
            raise ValueError("target exposure/cash fractions must not exceed one")
        if any(
            target.portfolio_plan_id != self.plan_id
            or target.timestamp != self.timestamp
            for target in self.targets
        ):
            raise ValueError("every target must reference its exact plan and timestamp")
        if any(order.portfolio_plan_id != self.plan_id for order in self.orders_to_create):
            raise ValueError("every proposed order must reference its exact plan")
        target_symbols = [item.symbol for item in self.targets]
        if target_symbols != sorted(target_symbols) or len(target_symbols) != len(
            set(target_symbols)
        ):
            raise ValueError("plan targets must have unique sorted symbols")
        order_symbols = [item.symbol for item in self.orders_to_create]
        if order_symbols != sorted(order_symbols) or len(order_symbols) != len(
            set(order_symbols)
        ):
            raise ValueError("a plan may create at most one sorted order per symbol")
        if self.orders_to_defer != tuple(sorted(set(self.orders_to_defer))):
            raise ValueError("deferred order symbols must be unique and sorted")


@dataclass(frozen=True, slots=True)
class PortfolioPlanResult:
    ranked_opportunities: tuple[PortfolioOpportunity, ...]
    decisions: tuple[PortfolioDecision, ...]
    plan: RebalancePlan
    sleeve_state: tuple[StrategySleeveState, ...]


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    average_gross_exposure: float
    max_gross_exposure: float
    average_cash_fraction: float
    max_unique_positions: int
    planned_turnover: float
    executed_turnover: float
    opportunities_selected: int
    opportunities_deferred: int
    opportunities_rejected: int
    targets_by_strategy_sleeve: tuple[tuple[str, float], ...]
    time_with_unused_strategy_budget: int
    group_exposure_over_time: tuple[tuple[str, str, float], ...]
    high_correlation_selections: int
    unknown_correlation_cases: int
