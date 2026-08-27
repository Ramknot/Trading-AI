from dataclasses import FrozenInstanceError
from datetime import timedelta
from decimal import Decimal

import pytest

from backtest_support import START, bar, dataset
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.models import (
    BacktestConfig,
    StrategySignalAction,
)
from trading_ai.core.config import load_runtime_settings
from trading_ai.core.models import PortfolioSnapshot
from trading_ai.features import FEATURE_SCHEMA_VERSION
from trading_ai.strategies import (
    BASELINE_STRATEGIES,
    BaselineSizer,
    BreakoutConfig,
    BreakoutStrategy,
    MomentumConfig,
    MomentumStrategy,
    TrendConfig,
    TrendFollowingStrategy,
    compare_reports,
    strategy_report,
)


def _series(
    closes: list[float], symbol: str = "AAPL", timeframe: str = "1d"
):
    return tuple(
        bar(
            index,
            symbol=symbol,
            timeframe=timeframe,
            opening=str(close),
            high=str(close + 0.5),
            low=str(close - 0.5),
            close=str(close),
            timestamp=START + timedelta(days=index),
        )
        for index, close in enumerate(closes)
    )


def _run(strategy, datasets, paper_context, *, benchmark="AAPL"):
    return BacktestEngine(code_version="lot3-test").run(
        strategy,
        datasets,
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("10000"),
            primary_timeframe=datasets[0].reference.timeframe,
            benchmark_symbol=benchmark,
        ),
    )


def test_strategy_configs_are_immutable_validated_and_windows_are_bars() -> None:
    config = TrendConfig(fast_window=2, slow_window=3)

    with pytest.raises(FrozenInstanceError):
        config.fast_window = 10  # type: ignore[misc]
    with pytest.raises(ValueError, match="lower"):
        TrendConfig(fast_window=20, slow_window=20)
    with pytest.raises(ValueError, match="top_k"):
        MomentumStrategy(("AAPL",), "1d", MomentumConfig(top_k=2))
    with pytest.raises(ValueError, match="allocation_fraction"):
        BreakoutConfig(allocation_fraction=Decimal("0"))


def test_baseline_sizer_is_cash_bounded_positive_and_long_only_helper() -> None:
    snapshot = PortfolioSnapshot(
        as_of=START,
        cash=Decimal("1000"),
        total_equity=Decimal("1000"),
    )
    sizer = BaselineSizer(Decimal("0.5"))

    assert sizer.entry_quantity(snapshot, Decimal("100")) == Decimal("5.000000")
    assert sizer.entry_quantity(snapshot, Decimal("100"), slots=2) == Decimal("2.500000")
    empty = PortfolioSnapshot(as_of=START, cash=Decimal("0"), total_equity=Decimal("1000"))
    assert sizer.entry_quantity(empty, Decimal("100")) is None
    with pytest.raises(ValueError, match="positive"):
        sizer.entry_quantity(snapshot, Decimal("0"))


def test_trend_enters_after_warmup_and_exits_on_reversal(paper_context) -> None:
    bars = _series([10, 11, 12, 13, 14, 10, 9, 8])
    result = _run(
        TrendFollowingStrategy(
            ("AAPL",),
            "1d",
            TrendConfig(
                fast_window=2,
                slow_window=3,
                slope_lookback=1,
                allocation_fraction=Decimal("0.5"),
            ),
        ),
        (dataset(bars),),
        paper_context,
    )

    assert [signal.action for signal in result.signals] == [
        StrategySignalAction.ENTER_LONG,
        StrategySignalAction.EXIT_LONG,
    ]
    assert result.orders[0].created_at == bars[2].timestamp
    assert result.fills[0].timestamp == bars[3].timestamp
    assert result.orders[0].signal_id == result.signals[0].signal_id
    assert result.metrics.number_of_trades == 1
    assert "EMA2 > EMA3" in result.signals[0].reason
    assert dict(result.strategy_parameters)["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    with pytest.raises(FrozenInstanceError):
        result.signals[0].reason = "opaque"  # type: ignore[misc]


def test_trend_has_no_signal_in_flat_market_or_before_warmup(paper_context) -> None:
    flat = dataset(_series([10, 10, 10, 10, 10]))
    short = dataset(_series([10, 11]))
    config = TrendConfig(fast_window=2, slow_window=3, slope_lookback=1)

    assert not _run(TrendFollowingStrategy(("AAPL",), "1d", config), (flat,), paper_context).orders
    assert not _run(TrendFollowingStrategy(("AAPL",), "1d", config), (short,), paper_context).orders


def test_breakout_uses_previous_range_then_exits(paper_context) -> None:
    bars = _series([10, 10, 10, 12, 13, 8, 7, 7])
    result = _run(
        BreakoutStrategy(
            ("AAPL",),
            "1d",
            BreakoutConfig(
                entry_window=3,
                exit_window=2,
                allocation_fraction=Decimal("0.5"),
            ),
        ),
        (dataset(bars),),
        paper_context,
    )

    assert [signal.action for signal in result.signals] == [
        StrategySignalAction.ENTER_LONG,
        StrategySignalAction.EXIT_LONG,
    ]
    assert result.orders[0].created_at == bars[3].timestamp
    assert dict(result.signals[0].features_used)["previous_high_3"] == "10.5"
    assert result.metrics.number_of_trades == 1


def test_breakout_does_not_trigger_when_close_only_matches_previous_high(
    paper_context,
) -> None:
    bars = _series([10, 10, 10, 10.5, 10.5])
    result = _run(
        BreakoutStrategy(
            ("AAPL",), "1d", BreakoutConfig(entry_window=3, exit_window=2)
        ),
        (dataset(bars),),
        paper_context,
    )

    assert result.orders == ()
    assert result.signals == ()


def test_momentum_ranks_assets_on_coherent_snapshots_and_rebalances(
    paper_context,
) -> None:
    inputs = (
        dataset(_series([100, 110, 120, 105, 100, 95], "AAPL")),
        dataset(_series([100, 99, 98, 97, 96, 95], "META")),
        dataset(_series([100, 105, 110, 125, 135, 140], "MSFT")),
    )
    strategy = MomentumStrategy(
        ("AAPL", "META", "MSFT"),
        "1d",
        MomentumConfig(
            lookback=2,
            top_k=1,
            rebalance_every=1,
            allocation_fraction=Decimal("0.5"),
        ),
    )

    result = _run(strategy, inputs, paper_context)

    assert result.signals[0].symbol == "AAPL"
    assert result.signals[0].action is StrategySignalAction.ENTER_LONG
    assert any(
        signal.symbol == "AAPL" and signal.action is StrategySignalAction.EXIT_LONG
        for signal in result.signals
    )
    assert any(
        signal.symbol == "MSFT" and signal.action is StrategySignalAction.ENTER_LONG
        for signal in result.signals
    )
    assert result.metrics.number_of_trades >= 1
    assert dict(result.strategy_parameters)["ranking_policy"] == (
        "exact_timestamp_no_forward_fill"
    )


def test_momentum_never_forward_fills_missing_cross_asset_bar(paper_context) -> None:
    aapl = dataset(_series([100, 110, 120, 130], "AAPL"))
    msft_bars = _series([100, 105, 110], "MSFT")
    msft = dataset(msft_bars)
    strategy = MomentumStrategy(
        ("AAPL", "MSFT"),
        "1d",
        MomentumConfig(lookback=2, top_k=1, rebalance_every=1),
    )

    result = _run(strategy, (aapl, msft), paper_context)

    assert all(signal.timestamp <= msft_bars[-1].timestamp for signal in result.signals)
    assert not any(signal.timestamp == aapl.bars[-1].timestamp for signal in result.signals)


@pytest.mark.parametrize("timeframe", ["1h", "4h", "1d"])
def test_baseline_runs_on_each_balanced_timeframe(timeframe, paper_context) -> None:
    result = _run(
        TrendFollowingStrategy(
            ("AAPL",),
            timeframe,
            TrendConfig(fast_window=2, slow_window=3, slope_lookback=1),
        ),
        (dataset(_series([10, 11, 12, 13, 14], timeframe=timeframe)),),
        paper_context,
    )

    assert result.status == "COMPLETED"
    assert result.orders


def test_registry_lists_and_builds_all_baselines_without_symbol_constants() -> None:
    assert BASELINE_STRATEGIES.names == ("breakout", "momentum", "trend")
    assert all(descriptor.default_parameters for descriptor in BASELINE_STRATEGIES.descriptors)
    assert BASELINE_STRATEGIES.create(
        "trend",
        symbols=("SAP",),
        timeframe="4h",
        config=TrendConfig(fast_window=2, slow_window=3),
    ).symbols == ("SAP",)
    with pytest.raises(ValueError, match="unknown"):
        BASELINE_STRATEGIES.create("winner", symbols=("AAPL",), timeframe="1d")


def test_same_inputs_produce_same_features_signals_orders_and_results(
    paper_context,
) -> None:
    inputs = (dataset(_series([10, 11, 12, 13, 14, 10, 9, 8])),)
    config = TrendConfig(fast_window=2, slow_window=3, slope_lookback=1)

    first = _run(TrendFollowingStrategy(("AAPL",), "1d", config), inputs, paper_context)
    second = _run(TrendFollowingStrategy(("AAPL",), "1d", config), inputs, paper_context)

    assert first.run_id == second.run_id
    assert first.signals == second.signals
    assert first.orders == second.orders
    assert first.fills == second.fills
    assert first.trades == second.trades
    assert first.equity_curve == second.equity_curve
    assert first.metrics == second.metrics
    assert first.result_hash == second.result_hash


def test_strategy_report_compares_to_benchmark_without_selecting_winner(
    paper_context,
) -> None:
    bars = (dataset(_series([10, 11, 12, 13, 14, 10, 9, 8])),)
    result = _run(
        TrendFollowingStrategy(
            ("AAPL",), "1d", TrendConfig(fast_window=2, slow_window=3)
        ),
        bars,
        paper_context,
    )

    report = strategy_report(result)
    comparison = compare_reports((result,))

    assert comparison == (report,)
    assert report.benchmark_return == result.benchmark.total_return
    assert report.excess_return == pytest.approx(
        report.total_return - report.benchmark_return
    )
    assert not hasattr(report, "best_strategy")
