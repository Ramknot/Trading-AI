"""Immutable models for point-in-time Balanced risk decisions and state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from trading_ai.core.models import (
    OrderRequest,
    PortfolioSnapshot,
    Position,
    TradingProfile,
)
from trading_ai.features.models import FeatureSnapshot, ReturnSeries


ZERO = Decimal("0")


def _aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    HALTED = "HALTED"


class RiskReasonCode(str, Enum):
    APPROVED = "APPROVED"
    POSITION_LIMIT = "POSITION_LIMIT"
    PORTFOLIO_EXPOSURE_LIMIT = "PORTFOLIO_EXPOSURE_LIMIT"
    MAX_POSITIONS = "MAX_POSITIONS"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    SHORT_NOT_ALLOWED = "SHORT_NOT_ALLOWED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    SOFT_DRAWDOWN = "SOFT_DRAWDOWN"
    HARD_DRAWDOWN = "HARD_DRAWDOWN"
    VOLATILITY_LIMIT = "VOLATILITY_LIMIT"
    VOLATILITY_UNKNOWN = "VOLATILITY_UNKNOWN"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    CORRELATION_UNKNOWN = "CORRELATION_UNKNOWN"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    UNKNOWN_GROUP = "UNKNOWN_GROUP"
    INVALID_RISK_CONTEXT = "INVALID_RISK_CONTEXT"
    RISK_CONFIG_INVALID = "RISK_CONFIG_INVALID"
    NO_EXPLICIT_RISK_DISTANCE = "NO_EXPLICIT_RISK_DISTANCE"
    TRADE_RISK_LIMIT = "TRADE_RISK_LIMIT"
    CIRCUIT_BREAKER_ACTIVE = "CIRCUIT_BREAKER_ACTIVE"
    RISK_REDUCING_ORDER = "RISK_REDUCING_ORDER"
    REDUCED_RISK_STATE = "REDUCED_RISK_STATE"


class CircuitBreakerReason(str, Enum):
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    HARD_DRAWDOWN = "HARD_DRAWDOWN"
    INVALID_RISK_STATE = "INVALID_RISK_STATE"
    INVALID_MARKET_DATA = "INVALID_MARKET_DATA"
    MANUAL_HALT = "MANUAL_HALT"
    BROKER_DESYNC = "BROKER_DESYNC"
    EXCESSIVE_LATENCY = "EXCESSIVE_LATENCY"


class UnknownRiskPolicy(str, Enum):
    ALLOW_WITH_WARNING = "ALLOW_WITH_WARNING"
    REJECT = "REJECT"


class VolatilityLevel(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RiskStateSnapshot:
    timestamp: datetime
    state: RiskState
    peak_equity: Decimal
    day_start_equity: Decimal
    current_equity: Decimal
    risk_day: date
    daily_loss_pct: float
    drawdown_pct: float
    halt_reason: str | None = None

    def __post_init__(self) -> None:
        _aware(self.timestamp, "timestamp")
        for field_name in ("peak_equity", "day_start_equity", "current_equity"):
            if getattr(self, field_name) < ZERO:
                raise ValueError(f"{field_name} must not be negative")
        for field_name in ("daily_loss_pct", "drawdown_pct"):
            if not 0.0 <= getattr(self, field_name) <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.halt_reason is not None:
            _text(self.halt_reason, "halt_reason")


@dataclass(frozen=True, slots=True)
class RiskStateTransition:
    transition_id: str
    timestamp: datetime
    previous_state: RiskState
    new_state: RiskState
    reason: str
    equity: Decimal
    daily_loss_pct: float
    drawdown_pct: float

    def __post_init__(self) -> None:
        _text(self.transition_id, "transition_id")
        _aware(self.timestamp, "timestamp")
        _text(self.reason, "reason")
        if self.previous_state is self.new_state:
            raise ValueError("risk transition must change state")


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Only information actually observable when one order is evaluated."""

    timestamp: datetime
    profile: TradingProfile
    portfolio: PortfolioSnapshot
    order: OrderRequest
    expected_entry_price: Decimal
    market_prices: tuple[tuple[str, Decimal], ...]
    risk_state: RiskStateSnapshot
    timeframe: str = "1d"
    feature_snapshot: FeatureSnapshot | None = None
    return_series: tuple[ReturnSeries, ...] = ()
    pending_sell_quantities: tuple[tuple[str, Decimal], ...] = ()

    def __post_init__(self) -> None:
        _aware(self.timestamp, "timestamp")
        _text(self.timeframe, "timeframe")
        if self.portfolio.as_of != self.timestamp:
            raise ValueError("portfolio snapshot must be current at risk timestamp")
        if (
            self.risk_state.timestamp != self.timestamp
            or self.risk_state.current_equity != self.portfolio.total_equity
        ):
            raise ValueError("risk state must match current timestamp and equity")
        if self.order.created_at is not None and self.order.created_at != self.timestamp:
            raise ValueError("order timestamp must match risk timestamp")
        if not self.expected_entry_price.is_finite() or self.expected_entry_price <= ZERO:
            raise ValueError("expected_entry_price must be positive and finite")
        symbols = [symbol for symbol, _ in self.market_prices]
        if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
            raise ValueError("market_prices must be sorted with unique symbols")
        if any(price <= ZERO or not price.is_finite() for _, price in self.market_prices):
            raise ValueError("market prices must be positive and finite")
        if self.feature_snapshot is not None:
            if self.feature_snapshot.symbol != self.order.symbol:
                raise ValueError("feature snapshot must match the order symbol")
            if self.feature_snapshot.timestamp != self.timestamp:
                raise ValueError("feature snapshot must be current, never future or stale")
        series_symbols = [series.symbol for series in self.return_series]
        if series_symbols != sorted(series_symbols) or len(series_symbols) != len(set(series_symbols)):
            raise ValueError("return_series must be sorted with unique symbols")
        if any(
            observation.timestamp > self.timestamp
            for series in self.return_series
            for observation in series.observations
        ):
            raise ValueError("risk returns must never contain future observations")
        pending_symbols = [symbol for symbol, _ in self.pending_sell_quantities]
        if pending_symbols != sorted(pending_symbols) or len(pending_symbols) != len(
            set(pending_symbols)
        ):
            raise ValueError("pending sells must be sorted with unique symbols")
        if any(
            not quantity.is_finite() or quantity < ZERO
            for _, quantity in self.pending_sell_quantities
        ):
            raise ValueError("pending sell quantities must be finite and non-negative")

    @property
    def equity(self) -> Decimal:
        return self.portfolio.total_equity

    @property
    def cash(self) -> Decimal:
        return self.portfolio.cash

    @property
    def open_positions(self) -> tuple[Position, ...]:
        return tuple(
            position
            for position in self.portfolio.positions
            if position.quantity != ZERO
        )

    def price_for(self, symbol: str) -> Decimal | None:
        return next(
            (price for item_symbol, price in self.market_prices if item_symbol == symbol),
            None,
        )

    def position_for(self, symbol: str) -> Position | None:
        return next(
            (position for position in self.portfolio.positions if position.symbol == symbol),
            None,
        )

    def available_quantity_for_exit(self, symbol: str) -> Decimal:
        position = self.position_for(symbol)
        held = position.quantity if position is not None else ZERO
        reserved = next(
            (
                quantity
                for item_symbol, quantity in self.pending_sell_quantities
                if item_symbol == symbol
            ),
            ZERO,
        )
        return max(ZERO, held - reserved)


@dataclass(frozen=True, slots=True)
class CorrelationAssessment:
    symbol: str
    coefficient: float | None
    observations: int
    highly_correlated: bool


@dataclass(frozen=True, slots=True)
class VolatilityAssessment:
    level: VolatilityLevel
    metric: float | None
    multiplier: Decimal
    reason_code: RiskReasonCode | None = None


@dataclass(frozen=True, slots=True)
class RiskSummary:
    risk_engine_name: str
    risk_engine_version: str
    risk_config_hash: str
    approved_orders: int
    reduced_orders: int
    rejected_orders: int
    rejection_reasons: tuple[tuple[str, int], ...]
    max_portfolio_exposure: float
    max_single_position_exposure: float
    max_observed_drawdown: float
    max_daily_loss: float
    time_in_reduced_state_seconds: float
    time_in_halted_state_seconds: float

    def __post_init__(self) -> None:
        _text(self.risk_engine_name, "risk_engine_name")
        _text(self.risk_engine_version, "risk_engine_version")
        if len(self.risk_config_hash) != 64:
            raise ValueError("risk_config_hash must be a SHA-256 digest")
        if min(self.approved_orders, self.reduced_orders, self.rejected_orders) < 0:
            raise ValueError("risk decision counts must not be negative")
