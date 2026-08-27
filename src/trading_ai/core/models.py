"""Immutable, typed business models shared across Trading AI components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from trading_ai.backtesting.models import (
        BacktestConfig,
        BacktestMetrics,
        BacktestOrder,
        BenchmarkResult,
        DatasetReference,
        EquityPoint,
        Fill,
        LedgerEntry,
        StrategySignal,
        Trade,
    )


class ExecutionEnvironment(str, Enum):
    """Strictly separated runtime environments."""

    DEV = "DEV"
    TEST = "TEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class TradingProfileName(str, Enum):
    """Supported profile identities; aggressive is locked in Lot 0."""

    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class RiskDecisionStatus(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class ExecutionStatus(str, Enum):
    BLOCKED = "BLOCKED"
    SUBMITTED = "SUBMITTED"


def _require_non_empty(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_aware(timestamp: datetime, field_name: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class TradingProfile:
    """Validated profile parameters loaded from TOML."""

    name: TradingProfileName
    enabled: bool
    timeframes: tuple[str, ...]
    asset_universe: tuple[str, ...]
    max_positions: int
    max_exposure: float
    max_turnover: float
    allow_short: bool
    risk_budget: float
    signal_threshold: float

    def __post_init__(self) -> None:
        if not self.timeframes:
            raise ValueError("timeframes must not be empty")
        if not self.asset_universe:
            raise ValueError("asset_universe must not be empty")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        for field_name in (
            "max_exposure",
            "max_turnover",
            "risk_budget",
            "signal_threshold",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TradingContext:
    """Environment and profile attached to every decision chain."""

    environment: ExecutionEnvironment
    profile: TradingProfileName


@dataclass(frozen=True, slots=True)
class Signal:
    """A normalized strategy input signal without execution authority."""

    signal_id: str
    symbol: str
    strength: float
    generated_at: datetime
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.signal_id, "signal_id")
        _require_non_empty(self.symbol, "symbol")
        _require_aware(self.generated_at, "generated_at")
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """A strategy opinion; it is never permission to place an order."""

    decision_id: str
    signal_id: str
    action: TradeAction
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        _require_non_empty(self.decision_id, "decision_id")
        _require_non_empty(self.signal_id, "signal_id")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """An order proposal that still requires RiskEngine authorization."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    strategy_decision_id: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.order_id, "order_id")
        _require_non_empty(self.symbol, "symbol")
        if self.quantity <= Decimal("0"):
            raise ValueError("quantity must be positive")
        if self.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError(
                f"{self.order_type.value} is reserved for a future implementation lot"
            )
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for LIMIT orders")
        if self.order_type is not OrderType.LIMIT and self.limit_price is not None:
            raise ValueError("limit_price is only valid for LIMIT orders")
        if self.limit_price is not None and self.limit_price <= Decimal("0"):
            raise ValueError("limit_price must be positive")
        if self.created_at is not None:
            _require_aware(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class Position:
    """A portfolio position; negative quantity represents a short."""

    symbol: str
    quantity: Decimal
    average_price: Decimal

    def __post_init__(self) -> None:
        _require_non_empty(self.symbol, "symbol")
        if self.average_price < Decimal("0"):
            raise ValueError("average_price must not be negative")


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """An immutable point-in-time portfolio view."""

    as_of: datetime
    cash: Decimal
    total_equity: Decimal
    positions: tuple[Position, ...] = ()

    def __post_init__(self) -> None:
        _require_aware(self.as_of, "as_of")
        if self.total_equity < Decimal("0"):
            raise ValueError("total_equity must not be negative")
        symbols = [position.symbol for position in self.positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("positions must have unique symbols")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The only decision that may authorize entry into execution."""

    decision_id: str
    order_id: str
    status: RiskDecisionStatus
    reason: str
    risk_engine: str

    def __post_init__(self) -> None:
        _require_non_empty(self.decision_id, "decision_id")
        _require_non_empty(self.order_id, "order_id")
        _require_non_empty(self.reason, "reason")
        _require_non_empty(self.risk_engine, "risk_engine")


@dataclass(frozen=True, slots=True)
class RiskApprovedOrder:
    """Order envelope accepted by brokers after a positive risk decision."""

    order: OrderRequest
    risk_decision: RiskDecision

    def __post_init__(self) -> None:
        if self.risk_decision.status is not RiskDecisionStatus.APPROVE:
            raise ValueError("only approved risk decisions may create this envelope")
        if self.risk_decision.order_id != self.order.order_id:
            raise ValueError("risk decision and order IDs must match")


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Broker acknowledgement returned after guarded submission."""

    order_id: str
    broker_order_id: str
    accepted_at: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.order_id, "order_id")
        _require_non_empty(self.broker_order_id, "broker_order_id")
        _require_aware(self.accepted_at, "accepted_at")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of the guarded execution entry point."""

    order_id: str
    status: ExecutionStatus
    message: str
    risk_decision: RiskDecision
    receipt: ExecutionReceipt | None = None


@dataclass(frozen=True, slots=True)
class MarketBar:
    """Normalized immutable OHLCV bar using explicit timezone-aware time."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    adjusted_close: Decimal | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        _require_non_empty(self.symbol, "symbol")
        _require_non_empty(self.timeframe, "timeframe")
        _require_non_empty(self.source, "source")
        _require_aware(self.timestamp, "timestamp")
        for field_name in ("open", "high", "low", "close"):
            if getattr(self, field_name) <= Decimal("0"):
                raise ValueError(f"{field_name} must be positive")
        if self.high < self.low:
            raise ValueError("high must not be lower than low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be at least open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be at most open and close")
        if self.volume < Decimal("0"):
            raise ValueError("volume must not be negative")
        if self.adjusted_close is not None and self.adjusted_close <= Decimal("0"):
            raise ValueError("adjusted_close must be positive when present")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable, traceable output of one deterministic historical simulation."""

    run_id: str
    status: str
    started_at: datetime
    completed_at: datetime
    created_at: datetime
    strategy_name: str
    strategy_version: str
    strategy_parameters: tuple[tuple[str, str], ...]
    dataset_references: tuple[DatasetReference, ...]
    config: BacktestConfig
    initial_cash: Decimal
    final_equity: Decimal
    metrics: BacktestMetrics
    equity_curve: tuple[EquityPoint, ...]
    orders: tuple[BacktestOrder, ...]
    fills: tuple[Fill, ...]
    trades: tuple[Trade, ...]
    signals: tuple[StrategySignal, ...]
    ledger_entries: tuple[LedgerEntry, ...]
    warnings: tuple[str, ...]
    benchmark: BenchmarkResult | None
    result_hash: str
    code_version: str | None
    source_hash_sha256: str

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.status, "status")
        _require_non_empty(self.strategy_name, "strategy_name")
        _require_non_empty(self.strategy_version, "strategy_version")
        if tuple(sorted(self.strategy_parameters)) != self.strategy_parameters:
            raise ValueError("strategy_parameters must be sorted")
        parameter_names = [name for name, _ in self.strategy_parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("strategy_parameters must have unique names")
        for name, value in self.strategy_parameters:
            _require_non_empty(name, "strategy parameter name")
            _require_non_empty(value, "strategy parameter value")
        _require_aware(self.started_at, "started_at")
        _require_aware(self.completed_at, "completed_at")
        _require_aware(self.created_at, "created_at")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.initial_cash <= Decimal("0"):
            raise ValueError("initial_cash must be positive")
        if self.final_equity < Decimal("0"):
            raise ValueError("final_equity must not be negative")
        if not self.equity_curve:
            raise ValueError("equity_curve must not be empty")
        if self.started_at != self.equity_curve[0].timestamp:
            raise ValueError("started_at must match the first equity point")
        if self.completed_at != self.equity_curve[-1].timestamp:
            raise ValueError("completed_at must match the final equity point")
        if self.final_equity != self.equity_curve[-1].equity:
            raise ValueError("final_equity must match the final equity point")
        if self.metrics.final_equity != self.final_equity:
            raise ValueError("metrics final_equity must match the result")
        if len(self.result_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.result_hash.lower()
        ):
            raise ValueError("result_hash must be a SHA-256 hexadecimal digest")
        if len(self.source_hash_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_hash_sha256.lower()
        ):
            raise ValueError("source_hash_sha256 must be a SHA-256 digest")
        signal_ids = [signal.signal_id for signal in self.signals]
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("strategy signal IDs must be unique")
        if any(
            order.signal_id is not None and order.signal_id not in set(signal_ids)
            for order in self.orders
        ):
            raise ValueError("every linked order must reference a result signal")

    @property
    def finished_at(self) -> datetime:
        """Compatibility alias for the original Lot 0 result model."""

        return self.completed_at
