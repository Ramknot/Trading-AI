from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pyarrow.parquet as parquet
import pytest

from backtest_support import START, bar, dataset
from ml_support import ConstantAdapter, model_artifact
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
from trading_ai.backtesting.reproducibility import stable_result_hash
from trading_ai.backtesting.strategy import BacktestStrategy
from trading_ai.core.config import load_runtime_settings
from trading_ai.core.models import OrderSide, RiskDecisionStatus
from trading_ai.ml.datasets import SignalTrainingDatasetBuilder
from trading_ai.ml.exceptions import MLConfigurationError
from trading_ai.ml.features import MLFeatureBuilder
from trading_ai.ml.inference import InferenceEngine, SignalMLScorer
from trading_ai.ml.models import MLFilterStatus, MLMode, ModelStatus
from trading_ai.ml.models import TimeRange
from trading_ai.ml.reporting import compare_quant_to_ml
from trading_ai.regimes import (
    ActivationStatus,
    BalancedStrategyActivationPolicy,
    StructureRegime,
    VolatilityRegime,
)
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.deny_all import DenyAllRiskEngine


PROFILE = load_runtime_settings().profile


def _series(count: int = 150):
    values = []
    for index in range(count):
        close = Decimal("100") + Decimal(index) / Decimal("10") + Decimal(index % 3) / Decimal("100")
        values.append(
            bar(
                index,
                opening=close - Decimal("0.05"),
                high=close + Decimal("0.50"),
                low=close - Decimal("0.50"),
                close=close,
                timestamp=START + timedelta(days=index),
            )
        )
    return tuple(values)


def _detector(
    structure: StructureRegime = StructureRegime.TREND_UP,
    volatility: VolatilityRegime = VolatilityRegime.NORMAL,
):
    return ScriptedTestRegimeDetector(lambda _: (structure, volatility))


class TimedSignalStrategy(BacktestStrategy):
    version = "1.0"

    def __init__(
        self,
        *,
        name: str = "breakout",
        quantity: Decimal = Decimal("10"),
        entry_bar: int = 131,
        exit_bar: int | None = None,
    ) -> None:
        self._name = name
        self.quantity = quantity
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar
        self._signals = []
        self._entry_submitted = False
        self._exit_submitted = False

    @property
    def name(self):
        return self._name

    @property
    def signals(self):
        return tuple(self._signals)

    def reset(self):
        self._signals.clear()
        self._entry_submitted = False
        self._exit_submitted = False

    def _intent(self, context, action, quantity):
        signal = StrategySignal(
            signal_id=f"signal-{self.name}-{len(self._signals) + 1:06d}",
            strategy_name=self.name,
            strategy_version=self.version,
            symbol=context.current_bar.symbol,
            timeframe=context.current_bar.timeframe,
            timestamp=context.current_time,
            action=action,
            strength=1.0,
            reason="explicit ML integration candidate",
            features_used=(),
        )
        self._signals.append(signal)
        return OrderIntent(
            symbol=signal.symbol,
            side=(OrderSide.BUY if action is StrategySignalAction.ENTER_LONG else OrderSide.SELL),
            quantity=quantity,
            timeframe=signal.timeframe,
            signal_id=signal.signal_id,
        )

    def on_bar(self, context: StrategyContext):
        count = len(context.history_for(context.current_bar.symbol, "1d"))
        position = next(
            (item for item in context.portfolio.positions if item.symbol == "AAPL"),
            None,
        )
        if not self._entry_submitted and count == self.entry_bar:
            self._entry_submitted = True
            return (
                self._intent(
                    context, StrategySignalAction.ENTER_LONG, self.quantity
                ),
            )
        if (
            self.exit_bar is not None
            and count == self.exit_bar
            and position is not None
            and not self._exit_submitted
        ):
            self._exit_submitted = True
            return (
                self._intent(
                    context, StrategySignalAction.EXIT_LONG, position.quantity
                ),
            )
        return ()


def _scorer(
    probability: float,
    *,
    mode: MLMode,
    strategy_name: str = "breakout",
    status: ModelStatus = ModelStatus.APPROVED,
    feature_schema_version: str = "1.1",
):
    adapter = ConstantAdapter(
        probability,
        MLFeatureBuilder.feature_names(strategy_name),
    )
    artifact = model_artifact(
        adapter,
        status=status,
        strategy_name=strategy_name,
        feature_schema_version=feature_schema_version,
    )
    return SignalMLScorer(
        mode=mode,
        inference_engine=InferenceEngine(artifact, adapter),
        threshold=0.55,
    )


def _run(
    paper_context,
    *,
    strategy=None,
    scorer=None,
    detector=None,
    risk_engine=None,
):
    return BacktestEngine(
        risk_engine=risk_engine or BalancedRiskEngine.from_profile(PROFILE),
        regime_detector=detector or _detector(),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(PROFILE),
        ml_scorer=scorer,
        code_version="lot6-test",
    ).run(
        strategy or TimedSignalStrategy(),
        (dataset(_series()),),
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("10000"),
            primary_timeframe="1d",
            benchmark_symbol="AAPL",
        ),
    )


def test_disabled_mode_is_trading_non_regression(paper_context) -> None:
    plain = _run(paper_context)
    disabled = _run(
        paper_context,
        scorer=SignalMLScorer(mode=MLMode.DISABLED),
    )
    for field_name in (
        "signals", "activation_decisions", "risk_decisions", "orders", "fills",
        "trades", "equity_curve", "metrics", "result_hash",
    ):
        assert getattr(disabled, field_name) == getattr(plain, field_name)
    assert disabled.ml_mode == "DISABLED"
    assert disabled.ml_predictions == ()


def test_score_only_low_probability_never_changes_trading(paper_context) -> None:
    quant = _run(paper_context)
    scored = _run(
        paper_context,
        scorer=_scorer(
            0.01, mode=MLMode.SCORE_ONLY, status=ModelStatus.CANDIDATE
        ),
    )
    assert tuple(replace(item, ml_decision_id=None) for item in scored.orders) == quant.orders
    assert scored.fills == quant.fills
    assert scored.metrics == quant.metrics
    assert scored.ml_predictions[0].probability_positive == 0.01
    assert scored.ml_decisions[0].status is MLFilterStatus.PASS


def test_filter_pass_preserves_full_ml_policy_risk_fill_lineage(paper_context) -> None:
    result = _run(paper_context, scorer=_scorer(0.99, mode=MLMode.FILTER))
    prediction = result.ml_predictions[0]
    ml_decision = result.ml_decisions[0]
    activation = result.activation_decisions[0]
    risk = result.risk_decisions[0]
    order = result.orders[0]

    assert ml_decision.status is MLFilterStatus.PASS
    assert ml_decision.prediction_id == prediction.prediction_id
    assert order.signal_id == ml_decision.signal_id == activation.signal_id
    assert order.ml_decision_id == ml_decision.decision_id
    assert order.activation_decision_id == activation.decision_id
    assert order.risk_decision_id == risk.decision_id
    assert len(result.fills) == 1
    assert result.ml_mode == "FILTER"
    assert result.ml_model_status == "APPROVED"


def test_filter_block_stops_before_sizer_policy_risk_and_order(paper_context) -> None:
    result = _run(paper_context, scorer=_scorer(0.10, mode=MLMode.FILTER))
    assert result.ml_decisions[0].status is MLFilterStatus.BLOCK
    assert result.activation_decisions == ()
    assert result.risk_decisions == ()
    assert result.orders == ()
    assert result.fills == ()


def test_missing_ml_feature_blocks_filter_but_score_only_would_continue(paper_context) -> None:
    early = TimedSignalStrategy(entry_bar=3)
    filtered = _run(
        paper_context,
        strategy=early,
        scorer=_scorer(0.99, mode=MLMode.FILTER),
    )
    assert filtered.ml_decisions[0].status is MLFilterStatus.UNAVAILABLE
    assert filtered.orders == ()

    scored = _run(
        paper_context,
        strategy=TimedSignalStrategy(entry_bar=3),
        scorer=_scorer(0.99, mode=MLMode.SCORE_ONLY),
    )
    assert scored.ml_decisions[0].status is MLFilterStatus.UNAVAILABLE
    assert len(scored.orders) == 1


def test_filter_requires_approved_model_and_schema_mismatch_fails_closed(paper_context) -> None:
    with pytest.raises(MLConfigurationError, match="APPROVED"):
        _scorer(0.99, mode=MLMode.FILTER, status=ModelStatus.VALIDATED)

    mismatch = _run(
        paper_context,
        scorer=_scorer(
            0.99,
            mode=MLMode.FILTER,
            feature_schema_version="0.9",
        ),
    )
    assert mismatch.ml_decisions[0].status is MLFilterStatus.UNAVAILABLE
    assert mismatch.orders == ()


def test_exit_is_not_applicable_to_ml_and_reaches_policy_risk(paper_context) -> None:
    result = _run(
        paper_context,
        strategy=TimedSignalStrategy(entry_bar=131, exit_bar=136),
        scorer=_scorer(0.99, mode=MLMode.FILTER),
    )
    exit_signal = next(
        item for item in result.signals if item.action is StrategySignalAction.EXIT_LONG
    )
    exit_decision = next(
        item for item in result.ml_decisions if item.signal_id == exit_signal.signal_id
    )
    exit_order = next(item for item in result.orders if item.side is OrderSide.SELL)
    assert exit_decision.status is MLFilterStatus.NOT_APPLICABLE
    assert exit_decision.prediction_id is None
    assert exit_order.ml_decision_id == exit_decision.decision_id
    assert len(result.fills) == 2


def test_policy_and_risk_remain_sovereign_after_high_ml_score(paper_context) -> None:
    policy_blocked = _run(
        paper_context,
        strategy=TimedSignalStrategy(name="trend"),
        scorer=_scorer(0.99, mode=MLMode.FILTER, strategy_name="trend"),
        detector=_detector(StructureRegime.RANGE),
    )
    assert policy_blocked.ml_decisions[0].status is MLFilterStatus.PASS
    assert policy_blocked.activation_decisions[0].status is ActivationStatus.BLOCK
    assert policy_blocked.orders == ()
    assert policy_blocked.risk_decisions == ()

    risk_blocked = _run(
        paper_context,
        scorer=_scorer(0.99, mode=MLMode.FILTER),
        risk_engine=DenyAllRiskEngine(),
    )
    assert risk_blocked.risk_decisions[0].status is RiskDecisionStatus.REJECT
    assert risk_blocked.fills == ()


def test_double_reduction_is_monotone_and_ml_never_sizes(paper_context) -> None:
    result = _run(
        paper_context,
        strategy=TimedSignalStrategy(quantity=Decimal("100")),
        scorer=_scorer(0.99, mode=MLMode.FILTER),
        detector=_detector(StructureRegime.RANGE),
    )
    activation = result.activation_decisions[0]
    risk = result.risk_decisions[0]
    fill = result.fills[0]
    assert activation.status is ActivationStatus.REDUCE
    assert activation.proposed_quantity == Decimal("100")
    assert activation.adjusted_quantity == Decimal("50.0")
    assert fill.quantity == risk.approved_quantity
    assert fill.quantity <= activation.adjusted_quantity <= activation.proposed_quantity


def test_ml_result_is_deterministic_and_training_dataset_builder_keeps_lineage(
    paper_context,
) -> None:
    first = _run(paper_context, scorer=_scorer(0.99, mode=MLMode.FILTER))
    second = _run(paper_context, scorer=_scorer(0.99, mode=MLMode.FILTER))
    assert first.ml_predictions == second.ml_predictions
    assert first.ml_decisions == second.ml_decisions
    assert first.orders == second.orders
    assert first.fills == second.fills
    assert first.metrics == second.metrics
    assert first.result_hash == second.result_hash

    quant = _run(paper_context)
    built = SignalTrainingDatasetBuilder().build(
        quant, (dataset(_series()),)
    )
    assert built.report.candidate_signals == 1
    assert built.report.labeled_examples == 1
    assert built.dataset.examples[0].model_input.timestamp == quant.signals[0].timestamp
    assert built.dataset.examples[0].label_end_timestamp > quant.signals[0].timestamp


def test_result_hash_excludes_technical_latency_and_comparison_is_oos(paper_context) -> None:
    quant = _run(paper_context)
    scored = _run(
        paper_context,
        scorer=_scorer(0.99, mode=MLMode.SCORE_ONLY),
    )
    prediction_with_latency = replace(
        scored.ml_predictions[0], technical_latency_ms=123.45
    )
    assert stable_result_hash(scored) == stable_result_hash(
        replace(
            scored,
            ml_predictions=(prediction_with_latency,),
            result_hash="0" * 64,
        )
    )
    report = compare_quant_to_ml(
        quant,
        scored,
        model_id=scored.ml_model_id,
        training_period=TimeRange(
            START - timedelta(days=500), START - timedelta(days=300)
        ),
        validation_period=TimeRange(
            START - timedelta(days=300), START - timedelta(days=1)
        ),
        test_period=TimeRange(
            START, START + timedelta(days=200)
        ),
    )
    assert report.sample_scope == "OUT_OF_SAMPLE"
    assert report.quant.total_return == report.quant_plus_ml.total_return
    assert not hasattr(report, "best_model")


def test_ml_exports_are_hashed_and_backward_summary_is_readable(tmp_path, paper_context) -> None:
    result = _run(paper_context, scorer=_scorer(0.99, mode=MLMode.FILTER))
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)
    summary = store.inspect(result.run_id)
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == "1.4"
    assert summary["ml"]["mode"] == "FILTER"
    assert parquet.read_table(directory / "ml_predictions.parquet").num_rows == 1
    assert parquet.read_table(directory / "ml_decisions.parquet").num_rows == 1
    assert "ml_predictions.parquet" in checksums["files"]
    assert "ml_decisions.parquet" in checksums["files"]
    assert store.verify_integrity(result.run_id) is True
