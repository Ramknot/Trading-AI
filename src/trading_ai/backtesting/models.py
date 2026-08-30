"""Immutable domain models for deterministic historical simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from trading_ai.core.models import (
    MarketBar,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    TradingContext,
    TradingProfile,
)
from trading_ai.data.models import CorporateAction, DataQualityReport

if TYPE_CHECKING:
    from trading_ai.regimes.models import RegimeSnapshot


ZERO = Decimal("0")


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class BacktestStatus(str, Enum):
    COMPLETED = "COMPLETED"


class ExecutionPolicy(str, Enum):
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"


class DataQualityPolicy(str, Enum):
    STRICT = "STRICT"
    ALLOW_WARNINGS = "ALLOW_WARNINGS"


class PricePolicy(str, Enum):
    RAW_WITH_CORPORATE_ACTIONS = "RAW_WITH_CORPORATE_ACTIONS"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class StrategySignalAction(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    HOLD = "HOLD"


class LedgerEntryType(str, Enum):
    FILL = "FILL"
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"


@dataclass(frozen=True, slots=True)
class CommissionConfig:
    """Broker-neutral commission assumptions for one fill."""

    fixed: Decimal = ZERO
    percentage_bps: Decimal = ZERO
    minimum: Decimal = ZERO

    def __post_init__(self) -> None:
        for field_name in ("fixed", "percentage_bps", "minimum"):
            if getattr(self, field_name) < ZERO:
                raise ValueError(f"commission {field_name} must not be negative")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Simulation assumptions, kept separate from the trading profile."""

    starting_cash: Decimal = Decimal("100000")
    spread_bps: Decimal = ZERO
    slippage_bps: Decimal = ZERO
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    execution_policy: ExecutionPolicy = ExecutionPolicy.NEXT_BAR_OPEN
    allow_short: bool = False
    risk_free_rate: Decimal = ZERO
    data_quality_policy: DataQualityPolicy = DataQualityPolicy.STRICT
    order_expiration_bars: int | None = None
    primary_timeframe: str = "1d"
    benchmark_symbol: str | None = None
    price_policy: PricePolicy = PricePolicy.RAW_WITH_CORPORATE_ACTIONS

    def __post_init__(self) -> None:
        if self.starting_cash <= ZERO:
            raise ValueError("starting_cash must be positive")
        for field_name in ("spread_bps", "slippage_bps"):
            if getattr(self, field_name) < ZERO:
                raise ValueError(f"{field_name} must not be negative")
        if self.spread_bps + self.slippage_bps >= Decimal("10000"):
            raise ValueError("combined spread and slippage must be below 10000 bps")
        if self.risk_free_rate <= Decimal("-1"):
            raise ValueError("risk_free_rate must be greater than -1")
        if self.order_expiration_bars is not None and self.order_expiration_bars < 1:
            raise ValueError("order_expiration_bars must be positive when set")
        if self.primary_timeframe not in {"1h", "4h", "1d"}:
            raise ValueError("primary_timeframe must be 1h, 4h, or 1d")
        if self.benchmark_symbol is not None:
            _require_text(self.benchmark_symbol, "benchmark_symbol")


@dataclass(frozen=True, slots=True)
class DatasetReference:
    """Exact historical-data provenance retained by a backtest result."""

    dataset_id: str
    provider: str
    symbol: str
    timeframe: str
    checksum_sha256: str
    data_kind: str
    requested_start: datetime
    requested_end: datetime
    actual_start: datetime | None
    actual_end: datetime | None
    manifest_file_path: str | None = None
    derived_from: tuple[str, ...] = ()
    corporate_actions_dataset_id: str | None = None
    corporate_actions_checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "provider",
            "symbol",
            "timeframe",
            "checksum_sha256",
            "data_kind",
        ):
            _require_text(getattr(self, field_name), field_name)
        if len(self.checksum_sha256) != 64:
            raise ValueError("checksum_sha256 must be a SHA-256 digest")
        if any(
            character not in "0123456789abcdef"
            for character in self.checksum_sha256.lower()
        ):
            raise ValueError("checksum_sha256 must be hexadecimal")
        _require_aware(self.requested_start, "requested_start")
        _require_aware(self.requested_end, "requested_end")
        if self.requested_start >= self.requested_end:
            raise ValueError("requested_start must precede requested_end")
        if self.actual_start is not None:
            _require_aware(self.actual_start, "actual_start")
        if self.actual_end is not None:
            _require_aware(self.actual_end, "actual_end")
        if (
            self.actual_start is not None
            and self.actual_end is not None
            and self.actual_start > self.actual_end
        ):
            raise ValueError("actual_start must not follow actual_end")
        if (self.corporate_actions_dataset_id is None) != (
            self.corporate_actions_checksum_sha256 is None
        ):
            raise ValueError(
                "corporate-action dataset ID and checksum must be recorded together"
            )
        if self.corporate_actions_checksum_sha256 is not None and (
            len(self.corporate_actions_checksum_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.corporate_actions_checksum_sha256.lower()
            )
        ):
            raise ValueError("corporate-action checksum must be a SHA-256 digest")
        if (
            self.data_kind == "DERIVED_RAW_WITH_ADJUSTED_CLOSE"
            and not self.derived_from
        ):
            raise ValueError("derived dataset references must retain source lineage")


@dataclass(frozen=True, slots=True)
class BacktestDataset:
    """One validated, provider-neutral historical series plus its lineage."""

    bars: tuple[MarketBar, ...]
    corporate_actions: tuple[CorporateAction, ...]
    quality_report: DataQualityReport
    reference: DatasetReference

    def __post_init__(self) -> None:
        if not self.bars:
            raise ValueError("backtest dataset must contain at least one bar")
        keys = [(bar.timestamp, bar.symbol, bar.timeframe) for bar in self.bars]
        if keys != sorted(keys):
            raise ValueError("backtest dataset bars must be deterministically sorted")
        if len(keys) != len(set(keys)):
            raise ValueError("backtest dataset must not contain duplicate bars")
        if any(
            bar.symbol != self.reference.symbol
            or bar.timeframe != self.reference.timeframe
            for bar in self.bars
        ):
            raise ValueError("bar identity must match dataset reference")
        if self.quality_report.row_count != len(self.bars):
            raise ValueError("quality row_count must match dataset bars")
        if (
            self.quality_report.symbol != self.reference.symbol
            or self.quality_report.timeframe != self.reference.timeframe
        ):
            raise ValueError("quality identity must match dataset reference")
        if any(bar.timestamp.utcoffset() != timedelta(0) for bar in self.bars):
            raise ValueError("normalized backtest bars must use UTC timestamps")
        if self.reference.actual_start != self.bars[0].timestamp:
            raise ValueError("reference actual_start must match the first bar")
        if self.reference.actual_end != self.bars[-1].timestamp:
            raise ValueError("reference actual_end must match the last bar")
        if any(
            not self.reference.requested_start
            <= bar.timestamp
            < self.reference.requested_end
            for bar in self.bars
        ):
            raise ValueError("bars must stay inside the referenced request range")
        if (
            self.quality_report.first_timestamp != self.bars[0].timestamp
            or self.quality_report.last_timestamp != self.bars[-1].timestamp
        ):
            raise ValueError("quality timestamp bounds must match dataset bars")
        action_keys = [
            (action.timestamp, action.symbol, action.action_type.value, action.value)
            for action in self.corporate_actions
        ]
        if action_keys != sorted(action_keys):
            raise ValueError("corporate actions must be deterministically sorted")
        if any(action.symbol != self.reference.symbol for action in self.corporate_actions):
            raise ValueError("corporate-action symbol must match dataset reference")
        if any(
            action.timestamp.utcoffset() != timedelta(0)
            for action in self.corporate_actions
        ):
            raise ValueError("normalized corporate actions must use UTC timestamps")
        if any(
            not self.reference.requested_start
            <= action.timestamp
            < self.reference.requested_end
            for action in self.corporate_actions
        ):
            raise ValueError("corporate actions must stay inside the request range")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A strategy request with quantity but no authority outside simulation."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    timeframe: str | None = None
    signal_id: str | None = None
    expected_entry_price: Decimal | None = None
    invalidation_price: Decimal | None = None
    risk_distance: Decimal | None = None
    portfolio_plan_id: str | None = None
    portfolio_decision_id: str | None = None
    portfolio_opportunity_ids: tuple[str, ...] = ()
    ml_decision_id: str | None = None
    activation_decision_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.symbol, "symbol")
        if self.quantity <= ZERO:
            raise ValueError("quantity must be positive")
        if self.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError(
                f"{self.order_type.value} is reserved for a future execution model"
            )
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None or self.limit_price <= ZERO:
                raise ValueError("positive limit_price is required for LIMIT orders")
        elif self.limit_price is not None:
            raise ValueError("limit_price is only valid for LIMIT orders")
        if self.timeframe is not None:
            _require_text(self.timeframe, "timeframe")
        if self.signal_id is not None:
            _require_text(self.signal_id, "signal_id")
        for field_name in (
            "expected_entry_price",
            "invalidation_price",
            "risk_distance",
        ):
            value = getattr(self, field_name)
            if value is not None and (not value.is_finite() or value <= ZERO):
                raise ValueError(f"{field_name} must be positive and finite")
        if self.invalidation_price is not None and self.risk_distance is not None:
            raise ValueError("provide invalidation_price or risk_distance, not both")
        for field_name in (
            "portfolio_plan_id",
            "portfolio_decision_id",
            "ml_decision_id",
            "activation_decision_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if tuple(sorted(set(self.portfolio_opportunity_ids))) != self.portfolio_opportunity_ids:
            raise ValueError("portfolio_opportunity_ids must be sorted and unique")
        if (self.portfolio_plan_id is None) != (self.portfolio_decision_id is None):
            raise ValueError("portfolio plan and decision lineage must be recorded together")


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """Explainable strategy event with no execution authority."""

    signal_id: str
    strategy_name: str
    strategy_version: str
    symbol: str
    timeframe: str
    timestamp: datetime
    action: StrategySignalAction
    strength: float
    reason: str
    features_used: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "signal_id",
            "strategy_name",
            "strategy_version",
            "symbol",
            "timeframe",
            "reason",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_aware(self.timestamp, "timestamp")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("signal strength must be between 0 and 1")
        if tuple(sorted(self.features_used)) != self.features_used:
            raise ValueError("features_used must be sorted by stable feature name")
        names = [name for name, _ in self.features_used]
        if len(names) != len(set(names)):
            raise ValueError("features_used must have unique names")
        for name, value in self.features_used:
            _require_text(name, "feature name")
            _require_text(value, "feature value")


@dataclass(frozen=True, slots=True)
class BacktestOrder:
    order_id: str
    symbol: str
    timeframe: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    created_at: datetime
    signal_id: str | None = None
    ml_decision_id: str | None = None
    activation_decision_id: str | None = None
    portfolio_plan_id: str | None = None
    portfolio_decision_id: str | None = None
    portfolio_opportunity_ids: tuple[str, ...] = ()
    risk_decision_id: str | None = None
    cost_estimate_id: str | None = None
    economic_decision_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Decimal | None = None
    status_reason: str | None = None
    completed_at: datetime | None = None
    eligible_bar_count: int = 0

    def __post_init__(self) -> None:
        _require_text(self.order_id, "order_id")
        _require_text(self.symbol, "symbol")
        _require_text(self.timeframe, "timeframe")
        _require_aware(self.created_at, "created_at")
        if self.signal_id is not None:
            _require_text(self.signal_id, "signal_id")
        if self.ml_decision_id is not None:
            _require_text(self.ml_decision_id, "ml_decision_id")
        if self.activation_decision_id is not None:
            _require_text(self.activation_decision_id, "activation_decision_id")
        for field_name in ("portfolio_plan_id", "portfolio_decision_id"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if tuple(sorted(set(self.portfolio_opportunity_ids))) != self.portfolio_opportunity_ids:
            raise ValueError("portfolio_opportunity_ids must be sorted and unique")
        if (self.portfolio_plan_id is None) != (self.portfolio_decision_id is None):
            raise ValueError("portfolio plan and decision lineage must be recorded together")
        if self.risk_decision_id is not None:
            _require_text(self.risk_decision_id, "risk_decision_id")
        for field_name in ("cost_estimate_id", "economic_decision_id"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if (self.cost_estimate_id is None) != (self.economic_decision_id is None):
            raise ValueError("cost estimate and economic decision lineage must be recorded together")
        if self.completed_at is not None:
            _require_aware(self.completed_at, "completed_at")
        if self.quantity <= ZERO:
            raise ValueError("quantity must be positive")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for LIMIT orders")
        if self.order_type is not OrderType.LIMIT and self.limit_price is not None:
            raise ValueError("limit_price is only valid for LIMIT orders")
        if self.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError(
                f"{self.order_type.value} is reserved for a future execution model"
            )
        if self.limit_price is not None and self.limit_price <= ZERO:
            raise ValueError("limit_price must be positive")
        if self.eligible_bar_count < 0:
            raise ValueError("eligible_bar_count must not be negative")
        if self.status is OrderStatus.PENDING and self.completed_at is not None:
            raise ValueError("pending orders cannot have completed_at")
        if self.status is not OrderStatus.PENDING and self.completed_at is None:
            raise ValueError("terminal orders require completed_at")


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    reference_price: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal
    slippage_cost: Decimal
    spread_cost: Decimal
    exchange_fees: Decimal = ZERO
    transaction_tax: Decimal = ZERO
    fx_cost: Decimal = ZERO
    financing_cost: Decimal = ZERO
    other_variable_cost: Decimal = ZERO
    total_variable_cost: Decimal | None = None
    cost_estimate_id: str | None = None
    economic_decision_id: str | None = None
    actual_cost_id: str | None = None
    unavailable_cost_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("fill_id", "order_id", "symbol"):
            _require_text(getattr(self, field_name), field_name)
        _require_aware(self.timestamp, "timestamp")
        if self.quantity <= ZERO:
            raise ValueError("fill quantity must be positive")
        if self.reference_price <= ZERO or self.price <= ZERO:
            raise ValueError("fill prices must be positive")
        for field_name in (
            "commission", "slippage_cost", "spread_cost", "exchange_fees",
            "transaction_tax", "fx_cost", "financing_cost", "other_variable_cost",
        ):
            if getattr(self, field_name) < ZERO:
                raise ValueError(f"{field_name} must not be negative")
        if self.total_variable_cost is not None and self.total_variable_cost < ZERO:
            raise ValueError("total_variable_cost must not be negative")
        for field_name in ("cost_estimate_id", "economic_decision_id", "actual_cost_id"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if tuple(sorted(set(self.unavailable_cost_components))) != self.unavailable_cost_components:
            raise ValueError("unavailable_cost_components must be sorted and unique")

    @property
    def cash_fees_excluding_price_impact(self) -> Decimal:
        """Fees debited separately; spread/slippage are already embedded in price."""

        return (
            self.commission
            + self.exchange_fees
            + self.transaction_tax
            + self.fx_cost
            + self.financing_cost
            + self.other_variable_cost
        )


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    entry_id: str
    timestamp: datetime
    entry_type: LedgerEntryType
    symbol: str
    cash_change: Decimal
    quantity_change: Decimal
    amount: Decimal
    reference_id: str
    message: str

    def __post_init__(self) -> None:
        for field_name in ("entry_id", "symbol", "reference_id", "message"):
            _require_text(getattr(self, field_name), field_name)
        _require_aware(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class PositionView:
    symbol: str
    quantity: Decimal
    average_entry_price: Decimal
    market_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    cash: Decimal
    positions_value: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "timestamp")
        if self.cash < ZERO:
            raise ValueError("equity curve cash must not be negative")
        if self.positions_value < ZERO or self.equity < ZERO:
            raise ValueError("equity values must not be negative")


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    gross_pnl: Decimal
    fees: Decimal
    spread_cost: Decimal
    slippage_cost: Decimal
    net_pnl: Decimal
    return_pct: Decimal
    holding_period_seconds: float
    exchange_fees: Decimal = ZERO
    transaction_tax: Decimal = ZERO
    fx_cost: Decimal = ZERO
    financing_cost: Decimal = ZERO
    other_variable_cost: Decimal = ZERO

    def __post_init__(self) -> None:
        _require_text(self.trade_id, "trade_id")
        _require_text(self.symbol, "symbol")
        _require_aware(self.entry_time, "entry_time")
        _require_aware(self.exit_time, "exit_time")
        if self.exit_time < self.entry_time:
            raise ValueError("trade exit must not precede entry")
        if self.quantity <= ZERO:
            raise ValueError("trade quantity must be positive")


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    initial_capital: Decimal
    final_equity: Decimal
    total_return: float
    annualized_return: float | None
    volatility: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float
    calmar_ratio: float | None
    drawdown_start: datetime | None
    drawdown_bottom: datetime | None
    drawdown_recovery: datetime | None
    drawdown_duration_seconds: float
    profit_factor: float | None
    win_rate: float | None
    average_win: Decimal | None
    average_loss: Decimal | None
    expectancy: Decimal | None
    number_of_trades: int
    turnover: float
    exposure: float
    best_trade: Decimal | None
    worst_trade: Decimal | None
    average_holding_period_seconds: float | None
    gross_profit: Decimal
    gross_loss: Decimal
    total_commission: Decimal
    total_spread_cost: Decimal
    total_slippage_cost: Decimal
    dividend_income: Decimal
    total_exchange_fees: Decimal = ZERO
    total_transaction_tax: Decimal = ZERO
    total_fx_cost: Decimal = ZERO
    total_financing_cost: Decimal = ZERO
    total_other_variable_cost: Decimal = ZERO
    total_variable_cost: Decimal = ZERO


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    symbol: str
    initial_equity: Decimal
    final_equity: Decimal
    total_return: float
    max_drawdown_pct: float
    excess_return: float
    equity_curve: tuple[EquityPoint, ...]

    def __post_init__(self) -> None:
        _require_text(self.symbol, "symbol")


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Read-only strategy view containing no bars later than current_time."""

    current_time: datetime
    current_bar: MarketBar
    history: tuple[MarketBar, ...]
    portfolio: PortfolioSnapshot
    trading_context: TradingContext
    profile: TradingProfile
    current_regime: RegimeSnapshot | None = None
    regime_history: tuple[RegimeSnapshot, ...] = ()

    def __post_init__(self) -> None:
        _require_aware(self.current_time, "current_time")
        if self.current_bar.timestamp != self.current_time:
            raise ValueError("current bar timestamp must equal current_time")
        if not self.history or self.history[-1] != self.current_bar:
            raise ValueError("history must end with current_bar")
        if any(bar.timestamp > self.current_time for bar in self.history):
            raise ValueError("strategy history must never contain future bars")
        if self.current_regime is not None and (
            self.current_regime.symbol != self.current_bar.symbol
            or self.current_regime.timeframe != self.current_bar.timeframe
            or self.current_regime.timestamp != self.current_time
        ):
            raise ValueError("current_regime must describe the current bar")
        if any(item.timestamp > self.current_time for item in self.regime_history):
            raise ValueError("strategy regime history must never contain future snapshots")
        if self.current_regime is not None and (
            not self.regime_history or self.regime_history[-1] != self.current_regime
        ):
            raise ValueError("regime_history must end with current_regime")

    def history_for(
        self, symbol: str, timeframe: str | None = None
    ) -> tuple[MarketBar, ...]:
        return tuple(
            bar
            for bar in self.history
            if bar.symbol == symbol
            and (timeframe is None or bar.timeframe == timeframe)
        )

    def regime_history_for(
        self, symbol: str, timeframe: str | None = None
    ) -> tuple[RegimeSnapshot, ...]:
        return tuple(
            snapshot
            for snapshot in self.regime_history
            if snapshot.symbol == symbol
            and (timeframe is None or snapshot.timeframe == timeframe)
        )
