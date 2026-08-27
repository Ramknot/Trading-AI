from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import json
import pyarrow.parquet as parquet

from backtest_support import START, bar, dataset
from regime_support import ScriptedTestRegimeDetector
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.models import (
    BacktestConfig,
    OrderIntent,
    StrategyContext,
    StrategySignal,
    StrategySignalAction,
)
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.backtesting.strategy import BacktestStrategy
from trading_ai.core.config import load_runtime_settings
from trading_ai.core.models import OrderSide, RiskDecisionStatus
from trading_ai.regimes import (
    ActivationStatus,
    BalancedStrategyActivationPolicy,
    StructureRegime,
    VolatilityRegime,
)
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.config import load_balanced_risk_config
from trading_ai.risk.models import CircuitBreakerReason
from trading_ai.strategies import (
    BreakoutConfig,
    BreakoutStrategy,
    MeanReversionConfig,
    MeanReversionStrategy,
    MomentumConfig,
    MomentumStrategy,
    TrendConfig,
    TrendFollowingStrategy,
)


PROFILE = load_runtime_settings().profile


def _series(closes: list[float], symbol: str = "AAPL"):
    return tuple(
        bar(
            index,
            symbol=symbol,
            opening=str(value),
            high=str(value + 0.25),
            low=str(value - 0.25),
            close=str(value),
            timestamp=START + timedelta(days=index),
        )
        for index, value in enumerate(closes)
    )


def _detector(
    structure: StructureRegime = StructureRegime.RANGE,
    volatility: VolatilityRegime = VolatilityRegime.NORMAL,
):
    return ScriptedTestRegimeDetector(lambda _: (structure, volatility))


def _run(
    strategy,
    datasets,
    paper_context,
    *,
    detector=None,
    risk_engine=None,
    benchmark="AAPL",
):
    return BacktestEngine(
        risk_engine=risk_engine or BalancedRiskEngine.from_profile(PROFILE),
        regime_detector=detector or _detector(),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(PROFILE),
        code_version="lot5-test",
    ).run(
        strategy,
        datasets,
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("10000"),
            primary_timeframe="1d",
            benchmark_symbol=benchmark,
        ),
    )


class OneBreakoutEntryStrategy(BacktestStrategy):
    name = "breakout"
    version = "1.0"

    def __init__(self, quantity: Decimal = Decimal("100")) -> None:
        self.quantity = quantity
        self._submitted = False
        self._signals: list[StrategySignal] = []

    @property
    def signals(self):
        return tuple(self._signals)

    def reset(self) -> None:
        self._submitted = False
        self._signals.clear()

    def on_bar(self, context: StrategyContext):
        if self._submitted:
            return ()
        self._submitted = True
        signal = StrategySignal(
            signal_id="signal-breakout-000001",
            strategy_name=self.name,
            strategy_version=self.version,
            symbol=context.current_bar.symbol,
            timeframe=context.current_bar.timeframe,
            timestamp=context.current_time,
            action=StrategySignalAction.ENTER_LONG,
            strength=1.0,
            reason="explicit integration proposal",
            features_used=(),
        )
        self._signals.append(signal)
        return (
            OrderIntent(
                symbol=context.current_bar.symbol,
                side=OrderSide.BUY,
                quantity=self.quantity,
                timeframe=context.current_bar.timeframe,
                signal_id=signal.signal_id,
            ),
        )


def test_policy_then_risk_never_increase_quantity_and_lineage_is_complete(
    paper_context,
) -> None:
    result = _run(
        OneBreakoutEntryStrategy(),
        (dataset(_series([100, 100, 100])),),
        paper_context,
    )

    activation = result.activation_decisions[0]
    risk = result.risk_decisions[0]
    order = result.orders[0]
    fill = result.fills[0]
    assert activation.status is ActivationStatus.REDUCE
    assert activation.proposed_quantity == Decimal("100")
    assert activation.adjusted_quantity == Decimal("50.0")
    assert risk.requested_quantity == Decimal("50.0")
    assert risk.approved_quantity == Decimal("15.000000")
    assert fill.quantity == Decimal("15.000000")
    assert fill.quantity <= activation.adjusted_quantity <= activation.proposed_quantity
    assert order.signal_id == activation.signal_id
    assert order.activation_decision_id == activation.decision_id
    assert order.risk_decision_id == risk.decision_id


def test_mean_reversion_range_entry_return_to_mean_exit_and_no_averaging_down(
    paper_context,
) -> None:
    strategy = MeanReversionStrategy(
        ("AAPL",),
        "1d",
        MeanReversionConfig(
            lookback=5,
            entry_zscore=Decimal("-1.5"),
            exit_zscore=Decimal("-0.25"),
            allocation_fraction=Decimal("0.20"),
        ),
    )
    result = _run(
        strategy,
        (dataset(_series([100, 101, 100, 101, 100, 95, 94, 93, 100, 101])),),
        paper_context,
    )

    entries = [
        signal
        for signal in result.signals
        if signal.action is StrategySignalAction.ENTER_LONG
    ]
    exits = [
        signal
        for signal in result.signals
        if signal.action is StrategySignalAction.EXIT_LONG
    ]
    assert len(entries) == 1
    assert len(exits) == 1
    assert len(result.fills) == 2
    assert len(result.trades) == 1
    assert all(
        decision.status is ActivationStatus.ALLOW
        for decision in result.activation_decisions
    )
    assert all(order.activation_decision_id for order in result.orders)


def test_mean_reversion_candidate_is_blocked_by_policy_outside_range(
    paper_context,
) -> None:
    strategy = MeanReversionStrategy(
        ("AAPL",),
        "1d",
        MeanReversionConfig(lookback=5),
    )
    result = _run(
        strategy,
        (dataset(_series([100, 101, 100, 101, 100, 95, 94, 93])),),
        paper_context,
        detector=_detector(StructureRegime.TREND_DOWN),
    )
    assert result.signals
    assert result.activation_decisions
    assert all(
        decision.status is ActivationStatus.BLOCK
        for decision in result.activation_decisions
    )
    assert not result.orders
    assert not result.fills


def test_mean_reversion_exits_when_structure_leaves_range(paper_context) -> None:
    change_time = START + timedelta(days=6)
    detector = ScriptedTestRegimeDetector(
        lambda features: (
            StructureRegime.TREND_DOWN
            if features.timestamp >= change_time
            else StructureRegime.RANGE,
            VolatilityRegime.NORMAL,
        )
    )
    strategy = MeanReversionStrategy(
        ("AAPL",),
        "1d",
        MeanReversionConfig(lookback=5),
    )
    result = _run(
        strategy,
        (dataset(_series([100, 101, 100, 101, 100, 95, 94, 93])),),
        paper_context,
        detector=detector,
    )

    exit_signal = next(
        signal
        for signal in result.signals
        if signal.action is StrategySignalAction.EXIT_LONG
    )
    exit_activation = next(
        decision
        for decision in result.activation_decisions
        if decision.signal_id == exit_signal.signal_id
    )
    assert "structure left RANGE" in exit_signal.reason
    assert exit_activation.status is ActivationStatus.ALLOW
    assert result.fills[-1].side is OrderSide.SELL


def test_breakout_in_range_is_reduced_before_risk(paper_context) -> None:
    strategy = BreakoutStrategy(
        ("AAPL",),
        "1d",
        BreakoutConfig(
            entry_window=2,
            exit_window=1,
            allocation_fraction=Decimal("0.25"),
        ),
    )
    result = _run(
        strategy,
        (dataset(_series([100, 100, 102, 103, 104])),),
        paper_context,
    )
    decision = next(
        item
        for item in result.activation_decisions
        if item.strategy_name == "breakout"
    )
    assert decision.status is ActivationStatus.REDUCE
    assert decision.allocation_multiplier == Decimal("0.5")
    assert result.risk_decisions[0].requested_quantity == decision.adjusted_quantity


def test_policy_blocked_entry_can_be_reconsidered_after_regime_changes(
    paper_context,
) -> None:
    eligible_time = START + timedelta(days=2)
    detector = ScriptedTestRegimeDetector(
        lambda features: (
            StructureRegime.TREND_UP
            if features.timestamp >= eligible_time
            else StructureRegime.RANGE,
            VolatilityRegime.NORMAL,
        )
    )
    strategy = TrendFollowingStrategy(
        ("AAPL",),
        "1d",
        TrendConfig(fast_window=1, slow_window=2, slope_lookback=1),
    )
    result = _run(
        strategy,
        (dataset(_series([100, 102, 104, 106, 108])),),
        paper_context,
        detector=detector,
    )

    assert result.activation_decisions[0].status is ActivationStatus.BLOCK
    assert any(
        decision.status is ActivationStatus.ALLOW
        for decision in result.activation_decisions[1:]
    )
    assert result.orders
    assert result.fills


def test_momentum_best_asset_is_blocked_when_its_regime_is_trend_down(
    paper_context,
) -> None:
    datasets = (
        dataset(_series([100, 110, 130, 150], "AAPL")),
        dataset(_series([100, 105, 110, 120], "MSFT")),
        dataset(_series([100, 102, 105, 110], "NVDA")),
    )
    detector = ScriptedTestRegimeDetector(
        lambda features: (
            StructureRegime.TREND_DOWN
            if features.symbol == "AAPL"
            else StructureRegime.TREND_UP,
            VolatilityRegime.NORMAL,
        )
    )
    strategy = MomentumStrategy(
        ("AAPL", "MSFT", "NVDA"),
        "1d",
        MomentumConfig(
            lookback=2,
            top_k=3,
            rebalance_every=1,
            allocation_fraction=Decimal("0.60"),
        ),
    )
    result = _run(
        strategy,
        datasets,
        paper_context,
        detector=detector,
        benchmark="AAPL",
    )

    aapl = next(
        decision
        for decision in result.activation_decisions
        if decision.symbol == "AAPL"
    )
    assert aapl.status is ActivationStatus.BLOCK
    assert all(order.symbol != "AAPL" for order in result.orders)
    assert {order.symbol for order in result.orders} == {"MSFT", "NVDA"}


class HaltedBalancedRiskEngine(BalancedRiskEngine):
    def reset(self, timestamp, equity) -> None:
        super().reset(timestamp, equity)
        self.halt(CircuitBreakerReason.MANUAL_HALT, timestamp, equity)


def test_risk_engine_remains_sovereign_after_policy_allow(paper_context) -> None:
    risk_config, groups = load_balanced_risk_config(PROFILE)
    halted = HaltedBalancedRiskEngine(PROFILE, risk_config, groups)
    result = _run(
        OneBreakoutEntryStrategy(Decimal("10")),
        (dataset(_series([100, 100, 100])),),
        paper_context,
        detector=_detector(StructureRegime.TREND_UP),
        risk_engine=halted,
    )

    assert result.activation_decisions[0].status is ActivationStatus.ALLOW
    assert result.risk_decisions[0].status is RiskDecisionStatus.REJECT
    assert not result.fills


def test_same_inputs_produce_same_regimes_decisions_and_result_hash(
    paper_context,
) -> None:
    datasets = (dataset(_series([100, 100, 102, 103, 104])),)
    strategy = BreakoutStrategy(
        ("AAPL",), "1d", BreakoutConfig(entry_window=2, exit_window=1)
    )
    engine = BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(PROFILE),
        regime_detector=_detector(),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(PROFILE),
        code_version="lot5-determinism",
    )
    config = BacktestConfig(
        starting_cash=Decimal("10000"),
        primary_timeframe="1d",
        benchmark_symbol="AAPL",
    )

    first = engine.run(strategy, datasets, paper_context, config)
    second = engine.run(strategy, datasets, paper_context, config)

    assert first.regime_snapshots == second.regime_snapshots
    assert first.regime_transitions == second.regime_transitions
    assert first.activation_decisions == second.activation_decisions
    assert first.risk_decisions == second.risk_decisions
    assert first.fills == second.fills
    assert first.result_hash == second.result_hash


def test_regime_exports_are_parquet_hashed_and_reported(tmp_path, paper_context) -> None:
    result = _run(
        OneBreakoutEntryStrategy(),
        (dataset(_series([100, 100, 100])),),
        paper_context,
    )
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    checksums = json.loads(
        (directory / "checksums.json").read_text(encoding="utf-8")
    )

    assert parquet.read_table(directory / "regime_snapshots.parquet").num_rows == 3
    assert parquet.read_table(directory / "regime_transitions.parquet").num_rows == 1
    assert parquet.read_table(directory / "activation_decisions.parquet").num_rows == 1
    assert summary["schema_version"] == "1.3"
    assert summary["regime"]["detector_name"] == "scripted-test-regime"
    assert summary["regime"]["policy_name"] == "balanced-strategy-policy"
    assert summary["regime"]["report"]["activation_reduce"] == 1
    for name in (
        "regime_snapshots.parquet",
        "regime_transitions.parquet",
        "activation_decisions.parquet",
    ):
        assert len(checksums["files"][name]) == 64
    assert store.verify_integrity(result.run_id) is True
