from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from backtest_support import START, bar, dataset
from risk_support import PermissiveBacktestEngine as BacktestEngine
from trading_ai.backtesting.exceptions import (
    BacktestConfigurationError,
    BacktestDataError,
)
from trading_ai.backtesting.execution import ExecutionModel
from trading_ai.backtesting.models import (
    BacktestConfig,
    DataQualityPolicy,
    LedgerEntryType,
    OrderIntent,
    OrderStatus,
)
from trading_ai.backtesting.strategy import BacktestStrategy, BuyAndHoldDemoStrategy
from trading_ai.core.models import (
    ExecutionEnvironment,
    OrderSide,
    OrderType,
    TradingContext,
    TradingProfileName,
)
from trading_ai.core.exceptions import LiveTradingLockedError, ProfileDisabledError
from trading_ai.data.models import Dividend, QualityStatus, StockSplit


class NoopStrategy(BacktestStrategy):
    @property
    def name(self) -> str:
        return "noop-test"

    def on_bar(self, context):
        return ()


class RoundTripStrategy(BacktestStrategy):
    def __init__(self) -> None:
        self.events = 0

    @property
    def name(self) -> str:
        return "round-trip-test"

    def reset(self) -> None:
        self.events = 0

    def on_bar(self, context):
        self.events += 1
        if self.events == 1:
            return (OrderIntent("AAPL", OrderSide.BUY, Decimal("10")),)
        if self.events == 3:
            return (OrderIntent("AAPL", OrderSide.SELL, Decimal("10")),)
        return ()


class OneIntentStrategy(BacktestStrategy):
    def __init__(self, intent: OrderIntent) -> None:
        self.intent = intent
        self.sent = False

    @property
    def name(self) -> str:
        return "one-intent-test"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return (
            ("order_type", self.intent.order_type.value),
            ("quantity", str(self.intent.quantity)),
            ("side", self.intent.side.value),
            ("symbol", self.intent.symbol),
        )

    def reset(self) -> None:
        self.sent = False

    def on_bar(self, context):
        if self.sent:
            return ()
        self.sent = True
        return (self.intent,)


class AuditHistoryStrategy(BacktestStrategy):
    def __init__(self) -> None:
        self.seen: list[tuple[object, str, int]] = []

    @property
    def name(self) -> str:
        return "history-audit-test"

    def reset(self) -> None:
        self.seen = []

    def on_bar(self, context):
        assert all(item.timestamp <= context.current_time for item in context.history)
        self.seen.append(
            (context.current_time, context.current_bar.symbol, len(context.history))
        )
        with pytest.raises(AttributeError):
            getattr(context, "next_close")
        return ()


class NeverFillExecutionModel(ExecutionModel):
    def try_fill(self, order, bar, fill_id):
        return None


def _daily_bars(symbol: str = "AAPL"):
    return tuple(
        bar(
            index,
            symbol=symbol,
            opening=str(100 + 10 * index),
            high=str(103 + 10 * index),
            low=str(98 + 10 * index),
            close=str(101 + 10 * index),
        )
        for index in range(5)
    )


def test_event_loop_market_orders_fill_at_next_bar_open(paper_context) -> None:
    result = BacktestEngine(code_version="test-code").run(
        RoundTripStrategy(),
        (dataset(_daily_bars()),),
        paper_context,
        BacktestConfig(starting_cash=Decimal("10000")),
    )

    assert [fill.timestamp for fill in result.fills] == [
        START + timedelta(days=1),
        START + timedelta(days=3),
    ]
    assert [fill.price for fill in result.fills] == [
        Decimal("110.00000000"),
        Decimal("130.00000000"),
    ]
    assert result.trades[0].gross_pnl == Decimal("200.00000000")
    assert result.final_equity == Decimal("10200.00000000")


def test_normal_strategy_api_cannot_observe_future_data(paper_context) -> None:
    strategy = AuditHistoryStrategy()

    BacktestEngine(code_version="test-code").run(
        strategy,
        (dataset(_daily_bars()),),
        paper_context,
        BacktestConfig(),
    )

    assert [event[2] for event in strategy.seen] == [1, 2, 3, 4, 5]


def test_multi_asset_timeline_is_timestamp_then_symbol(paper_context) -> None:
    strategy = AuditHistoryStrategy()
    aapl = dataset(_daily_bars("AAPL")[:2])
    msft = dataset(_daily_bars("MSFT")[:2])

    BacktestEngine(code_version="test-code").run(
        strategy,
        (msft, aapl),
        paper_context,
        BacktestConfig(),
    )

    assert [(time, symbol) for time, symbol, _ in strategy.seen] == [
        (START, "AAPL"),
        (START, "MSFT"),
        (START + timedelta(days=1), "AAPL"),
        (START + timedelta(days=1), "MSFT"),
    ]


def test_each_balanced_timeframe_can_be_the_primary_event_stream(
    paper_context,
) -> None:
    hourly = tuple(bar(index, timeframe="1h") for index in range(3))
    strategy = AuditHistoryStrategy()

    result = BacktestEngine(code_version="test-code").run(
        strategy,
        (dataset(hourly),),
        paper_context,
        BacktestConfig(primary_timeframe="1h"),
    )

    assert len(result.equity_curve) == 3
    assert all(event[1] == "AAPL" for event in strategy.seen)


def test_secondary_timeframes_are_rejected_until_a_future_extension(
    paper_context,
) -> None:
    daily = dataset(_daily_bars()[:2])
    hourly = dataset(tuple(bar(index, timeframe="1h") for index in range(2)))

    with pytest.raises(BacktestConfigurationError, match="one primary timeframe"):
        BacktestEngine().run(
            NoopStrategy(),
            (daily, hourly),
            paper_context,
            BacktestConfig(primary_timeframe="1d"),
        )


def test_execution_model_is_injectable_and_independent(paper_context) -> None:
    result = BacktestEngine(
        execution_model_factory=lambda config: NeverFillExecutionModel(),
        code_version="test-code",
    ).run(
        OneIntentStrategy(OrderIntent("AAPL", OrderSide.BUY, Decimal("1"))),
        (dataset(_daily_bars()[:2]),),
        paper_context,
        BacktestConfig(),
    )

    assert result.fills == ()
    assert result.orders[0].status is OrderStatus.PENDING


def test_unreached_limit_stays_pending_without_expiry(paper_context) -> None:
    strategy = OneIntentStrategy(
        OrderIntent(
            "AAPL",
            OrderSide.BUY,
            Decimal("1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50"),
        )
    )

    result = BacktestEngine(code_version="test-code").run(
        strategy,
        (dataset(_daily_bars()),),
        paper_context,
        BacktestConfig(),
    )

    assert result.orders[0].status is OrderStatus.PENDING
    assert result.orders[0].eligible_bar_count == 4
    assert result.fills == ()
    assert "remained PENDING" in result.warnings[-1]


def test_unreached_limit_expires_after_configured_eligible_bars(
    paper_context,
) -> None:
    strategy = OneIntentStrategy(
        OrderIntent(
            "AAPL",
            OrderSide.BUY,
            Decimal("1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50"),
        )
    )

    result = BacktestEngine(code_version="test-code").run(
        strategy,
        (dataset(_daily_bars()),),
        paper_context,
        BacktestConfig(order_expiration_bars=2),
    )

    assert result.orders[0].status is OrderStatus.EXPIRED
    assert result.orders[0].completed_at == START + timedelta(days=2)
    assert result.orders[0].eligible_bar_count == 2


def test_insufficient_cash_rejects_buy_without_negative_cash(paper_context) -> None:
    strategy = OneIntentStrategy(
        OrderIntent("AAPL", OrderSide.BUY, Decimal("1000"))
    )

    result = BacktestEngine(code_version="test-code").run(
        strategy,
        (dataset(_daily_bars()),),
        paper_context,
        BacktestConfig(starting_cash=Decimal("1000")),
    )

    assert result.orders[0].status is OrderStatus.REJECTED
    assert "insufficient cash" in (result.orders[0].status_reason or "")
    assert result.fills == ()
    assert all(point.cash >= 0 for point in result.equity_curve)


def test_balanced_sell_cannot_create_negative_position(paper_context) -> None:
    strategy = OneIntentStrategy(
        OrderIntent("AAPL", OrderSide.SELL, Decimal("1"))
    )

    result = BacktestEngine(code_version="test-code").run(
        strategy,
        (dataset(_daily_bars()),),
        paper_context,
        BacktestConfig(),
    )

    assert result.orders[0].status is OrderStatus.REJECTED
    assert "short selling is disabled" in (result.orders[0].status_reason or "")
    assert result.fills == ()


def test_allow_short_cannot_exceed_balanced_profile(paper_context) -> None:
    with pytest.raises(BacktestConfigurationError, match="Balanced is long-only"):
        BacktestEngine().run(
            NoopStrategy(),
            (dataset(_daily_bars()),),
            paper_context,
            BacktestConfig(allow_short=True),
        )


def test_data_quality_fail_is_always_rejected(paper_context) -> None:
    failing = dataset(_daily_bars(), status=QualityStatus.FAIL)

    with pytest.raises(BacktestDataError, match="DataQuality FAIL"):
        BacktestEngine().run(
            NoopStrategy(), failing and (failing,), paper_context, BacktestConfig()
        )


def test_data_quality_warning_obeys_policy(paper_context) -> None:
    warning = dataset(
        _daily_bars(),
        status=QualityStatus.WARNING,
        missing=1,
        unexpected_gaps=1,
        warnings=("synthetic gap",),
    )
    with pytest.raises(BacktestDataError, match="DataQuality WARNING"):
        BacktestEngine().run(
            NoopStrategy(), (warning,), paper_context, BacktestConfig()
        )

    result = BacktestEngine(code_version="test-code").run(
        NoopStrategy(),
        (warning,),
        paper_context,
        BacktestConfig(data_quality_policy=DataQualityPolicy.ALLOW_WARNINGS),
    )

    assert "synthetic gap" in result.warnings
    assert any("missing_expected_bars=1" in item for item in result.warnings)


def test_adjusted_only_dataset_kind_is_rejected(paper_context) -> None:
    original = dataset(_daily_bars())
    adjusted = replace(
        original,
        reference=replace(original.reference, data_kind="ADJUSTED"),
    )

    with pytest.raises(BacktestDataError, match="raw OHLC"):
        BacktestEngine().run(
            NoopStrategy(), (adjusted,), paper_context, BacktestConfig()
        )


def test_dividend_and_split_use_raw_prices_without_double_counting(
    paper_context,
) -> None:
    bars = (
        bar(0, opening="100", high="101", low="99", close="100", adjusted_close="1"),
        bar(1, opening="100", high="101", low="99", close="100", adjusted_close="1"),
        bar(2, opening="100", high="101", low="99", close="100", adjusted_close="1"),
        bar(3, opening="50", high="51", low="49", close="50", adjusted_close="1"),
    )
    actions = (
        Dividend(
            "AAPL", START + timedelta(days=2), Decimal("0.50"), "synthetic"
        ),
        StockSplit(
            "AAPL", START + timedelta(days=3), Decimal("2"), "synthetic"
        ),
    )
    strategy = OneIntentStrategy(
        OrderIntent("AAPL", OrderSide.BUY, Decimal("10"))
    )

    result = BacktestEngine(code_version="test-code").run(
        strategy,
        (dataset(bars, actions=actions),),
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("2000"), benchmark_symbol="AAPL"
        ),
    )

    assert result.final_equity == Decimal("2005.00000000")
    assert result.metrics.dividend_income == Decimal("5.000000000")
    assert [entry.entry_type for entry in result.ledger_entries] == [
        LedgerEntryType.FILL,
        LedgerEntryType.DIVIDEND,
        LedgerEntryType.SPLIT,
    ]
    assert result.equity_curve[-1].positions_value == Decimal("1000.00000000")
    assert result.benchmark is not None
    assert result.benchmark.final_equity == Decimal("2010.00")


def test_result_references_exact_dataset_and_is_reproducible(paper_context) -> None:
    historical = dataset(_daily_bars())
    config = BacktestConfig(
        starting_cash=Decimal("10000"), benchmark_symbol="AAPL"
    )
    engine = BacktestEngine(code_version="fixed-commit")

    first = engine.run(RoundTripStrategy(), (historical,), paper_context, config)
    second = engine.run(RoundTripStrategy(), (historical,), paper_context, config)

    assert first.run_id == second.run_id
    assert first.result_hash == second.result_hash
    assert first.orders == second.orders
    assert first.fills == second.fills
    assert first.trades == second.trades
    assert first.equity_curve == second.equity_curve
    assert first.metrics == second.metrics
    assert first.dataset_references == (historical.reference,)
    assert first.code_version == "fixed-commit"
    assert len(first.source_hash_sha256) == 64


def test_strategy_parameters_are_part_of_the_stable_run_identity(
    paper_context,
) -> None:
    historical = dataset(_daily_bars())
    engine = BacktestEngine(code_version="fixed-commit")

    one_share = engine.run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("1")),
        (historical,),
        paper_context,
        BacktestConfig(),
    )
    two_shares = engine.run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("2")),
        (historical,),
        paper_context,
        BacktestConfig(),
    )

    assert one_share.strategy_parameters == (
        ("quantity", "1"),
        ("symbol", "AAPL"),
    )
    assert one_share.run_id != two_shares.run_id
    assert one_share.result_hash != two_shares.result_hash


def test_non_overlapping_partitions_of_one_series_are_supported(
    paper_context,
) -> None:
    first = dataset(_daily_bars()[:2])
    second = dataset(_daily_bars()[2:])

    result = BacktestEngine(code_version="test-code").run(
        NoopStrategy(),
        (second, first),
        paper_context,
        BacktestConfig(),
    )

    assert len(result.equity_curve) == 5
    assert len(result.dataset_references) == 2


def test_overlapping_partitions_fail_on_duplicate_bar(paper_context) -> None:
    first = dataset(_daily_bars()[:3])
    second = dataset(_daily_bars()[2:])

    with pytest.raises(BacktestDataError, match="duplicate"):
        BacktestEngine().run(
            NoopStrategy(), (first, second), paper_context, BacktestConfig()
        )


def test_benchmark_uses_configured_symbol_and_aligned_data(paper_context) -> None:
    result = BacktestEngine(code_version="test-code").run(
        RoundTripStrategy(),
        (dataset(_daily_bars()),),
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("10000"), benchmark_symbol="AAPL"
        ),
    )

    assert result.benchmark is not None
    assert result.benchmark.symbol == "AAPL"
    assert result.benchmark.equity_curve[0].timestamp == result.started_at
    assert result.benchmark.equity_curve[-1].timestamp == result.completed_at
    assert result.benchmark.excess_return == pytest.approx(
        result.metrics.total_return - result.benchmark.total_return
    )


def test_benchmark_requires_an_explicit_supplied_dataset(paper_context) -> None:
    with pytest.raises(BacktestDataError, match="benchmark dataset is missing"):
        BacktestEngine().run(
            NoopStrategy(),
            (dataset(_daily_bars()),),
            paper_context,
            BacktestConfig(benchmark_symbol="SPY"),
        )


def test_transaction_costs_reduce_net_result(paper_context) -> None:
    historical = dataset(_daily_bars())
    free = BacktestEngine(code_version="test-code").run(
        RoundTripStrategy(),
        (historical,),
        paper_context,
        BacktestConfig(starting_cash=Decimal("10000")),
    )
    costly = BacktestEngine(code_version="test-code").run(
        RoundTripStrategy(),
        (historical,),
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("10000"),
            spread_bps=Decimal("5"),
            slippage_bps=Decimal("10"),
        ),
    )

    assert costly.final_equity < free.final_equity
    assert costly.metrics.total_spread_cost > 0
    assert costly.metrics.total_slippage_cost > 0


@pytest.mark.parametrize(
    ("context", "error_type"),
    [
        (
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.AGGRESSIVE),
            ProfileDisabledError,
        ),
        (
            TradingContext(ExecutionEnvironment.LIVE, TradingProfileName.BALANCED),
            LiveTradingLockedError,
        ),
    ],
)
def test_backtest_engine_preserves_runtime_safety_locks(context, error_type) -> None:
    with pytest.raises(error_type):
        BacktestEngine().run(
            NoopStrategy(), (dataset(_daily_bars()),), context, BacktestConfig()
        )
