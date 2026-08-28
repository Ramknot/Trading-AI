from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from backtest_support import START, bar
from ml_support import ML_START, tabular_input, temporal_split, training_dataset
from trading_ai.backtesting.models import StrategySignal, StrategySignalAction
from trading_ai.features import FeatureEngine
from trading_ai.ml.datasets import SignalTrainingExample
from trading_ai.ml.exceptions import MLDataError
from trading_ai.ml.features import (
    COMMON_ML_FEATURE_NAMES,
    ML_FEATURE_SCHEMA_VERSION,
    MLFeatureBuilder,
)
from trading_ai.ml.inputs import TabularModelInput
from trading_ai.ml.labels import LabelBuilder, LabelConfig
from trading_ai.ml.splits import PurgedWalkForwardSplitter, TemporalSplitConfig
from trading_ai.ml.models import TimeRange
from trading_ai.regimes.models import RegimeSnapshot, StructureRegime, VolatilityRegime


def _bars(count: int, *, future_extreme: bool = False):
    values = []
    for index in range(count):
        close = 100 + index * 0.2 + ((index % 5) - 2) * 0.3
        if future_extreme and index >= 120:
            close = 1000 + index * 10
        values.append(
            bar(
                index,
                opening=str(close - 0.1),
                high=str(close + 0.5),
                low=str(close - 0.5),
                close=str(close),
                timestamp=START + timedelta(days=index),
            )
        )
    return tuple(values)


def _signal(timestamp, *, strategy="trend"):
    return StrategySignal(
        signal_id="signal-test-1",
        strategy_name=strategy,
        strategy_version="1.0",
        symbol="AAPL",
        timeframe="1d",
        timestamp=timestamp,
        action=StrategySignalAction.ENTER_LONG,
        strength=1.0,
        reason="test candidate",
        features_used=(
            (("relative_strength_percentile", "0.75"),)
            if strategy == "momentum"
            else ()
        ),
    )


def _regime(timestamp):
    return RegimeSnapshot(
        snapshot_id="regime-test-1",
        symbol="AAPL",
        timestamp=timestamp,
        timeframe="1d",
        structure_regime=StructureRegime.TREND_UP,
        volatility_regime=VolatilityRegime.NORMAL,
        detector_name="test",
        detector_version="1",
        config_hash="a" * 64,
        bars_in_current_structure_regime=3,
        evidence=(),
        reason_codes=("TEST",),
    )


def test_label_builder_uses_next_open_horizon_close_and_drops_short_future() -> None:
    bars = (
        bar(0, opening="100", close="100"),
        bar(1, opening="101", close="102"),
        bar(2, opening="103", close="104"),
        bar(3, opening="105", close="106"),
        bar(4, opening="107", high="111", close="110"),
    )
    label = LabelBuilder(LabelConfig(horizon_bars=2)).build(bars, 2)

    assert label.entry_price == Decimal("105")
    assert label.exit_price == Decimal("110")
    assert label.label_end_timestamp == bars[4].timestamp
    assert label.target == 1
    assert LabelBuilder(LabelConfig(horizon_bars=3)).build(bars, 2) is None


def test_label_builder_negative_threshold_and_no_target_in_features() -> None:
    bars = (
        bar(0, opening="100", close="100"),
        bar(1, opening="100", close="100"),
        bar(2, opening="100", close="100"),
        bar(3, opening="105", close="104"),
        bar(4, opening="103", low="99", close="100"),
    )
    label = LabelBuilder(LabelConfig(horizon_bars=2)).build(bars, 2)
    assert label.target == 0
    assert not any(
        forbidden in name
        for name in COMMON_ML_FEATURE_NAMES
        for forbidden in ("future", "next", "pnl", "winner", "label")
    )


def test_ml_feature_vector_is_point_in_time_stable_and_symbol_is_metadata() -> None:
    observed = _bars(120)
    extended = _bars(125, future_extreme=True)
    as_of = observed[-1].timestamp
    engine = FeatureEngine()
    builder = MLFeatureBuilder()
    before = engine.compute(observed, builder.feature_request, as_of=as_of)
    after = engine.compute(extended, builder.feature_request, as_of=as_of)

    first = builder.build(signal=_signal(as_of), features=before, regime=_regime(as_of))
    second = builder.build(signal=_signal(as_of), features=after, regime=_regime(as_of))

    assert ML_FEATURE_SCHEMA_VERSION == "1.0"
    assert first.values == second.values
    assert first.input_hash == second.input_hash
    assert "AAPL" not in first.feature_names
    assert first.feature_names == COMMON_ML_FEATURE_NAMES


def test_momentum_ml_features_include_point_in_time_relative_strength() -> None:
    bars = _bars(120)
    snapshot = FeatureEngine().compute(bars, MLFeatureBuilder.feature_request)
    model_input = MLFeatureBuilder().build(
        signal=_signal(bars[-1].timestamp, strategy="momentum"),
        features=snapshot,
        regime=_regime(bars[-1].timestamp),
    )
    assert model_input.feature_names[-1] == "relative_strength_percentile"
    assert dict(model_input.values)["relative_strength_percentile"] == 0.75


def test_tabular_input_is_immutable_utc_and_missing_features_are_not_zero_filled() -> None:
    model_input = tabular_input(1)
    with pytest.raises(FrozenInstanceError):
        model_input.symbol = "MSFT"
    with pytest.raises(ValueError, match="timezone-aware"):
        TabularModelInput(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1),
            timeframe="1d",
            strategy_name="trend",
            strategy_version="1.0",
            feature_schema_version="1.1",
            ml_feature_schema_version="1.0",
            values=(("x", 1.0),),
        )
    bars = _bars(10)
    snapshot = FeatureEngine().compute(bars, MLFeatureBuilder.feature_request)
    with pytest.raises(MLDataError, match="unavailable"):
        MLFeatureBuilder().build(
            signal=_signal(bars[-1].timestamp),
            features=snapshot,
            regime=_regime(bars[-1].timestamp),
        )


def test_temporal_split_is_ordered_purged_embargoed_and_shared_across_assets() -> None:
    dataset = training_dataset(symbols=("AAPL", "MSFT"))
    partition = PurgedWalkForwardSplitter(temporal_split()).partition(dataset.examples)

    assert all(
        item.model_input.timestamp < temporal_split().validation.start
        for item in partition.training
    )
    assert min(item.model_input.timestamp for item in partition.validation) == (
        temporal_split().validation.start + timedelta(days=1)
    )
    assert min(item.model_input.timestamp for item in partition.final_test) == (
        temporal_split().final_test.start + timedelta(days=1)
    )
    by_timestamp = {}
    for item in partition.training:
        by_timestamp.setdefault(item.model_input.timestamp, set()).add(
            item.model_input.symbol
        )
    assert all(symbols == {"AAPL", "MSFT"} for symbols in by_timestamp.values())
    keys = [
        (item.model_input.timestamp, item.model_input.symbol)
        for item in (*partition.training, *partition.validation, *partition.final_test)
    ]
    assert keys == sorted(keys[: len(partition.training)]) + sorted(
        keys[len(partition.training) : len(partition.training) + len(partition.validation)]
    ) + sorted(keys[-len(partition.final_test) :])


def test_purge_removes_label_that_crosses_validation_boundary() -> None:
    dataset = training_dataset(count=12)
    examples = list(dataset.examples)
    crossing = replace(
        examples[4], label_end_timestamp=ML_START + timedelta(days=6)
    )
    examples[4] = crossing
    config = TemporalSplitConfig(
        training=TimeRange(ML_START, ML_START + timedelta(days=5)),
        validation=TimeRange(
            ML_START + timedelta(days=6), ML_START + timedelta(days=9)
        ),
        final_test=TimeRange(
            ML_START + timedelta(days=9), ML_START + timedelta(days=12)
        ),
        embargo_bars=0,
        walk_forward_folds=1,
    )
    partition = PurgedWalkForwardSplitter(config).partition(tuple(examples))
    assert crossing not in partition.training
    assert partition.purged_count >= 1


def test_walk_forward_folds_are_expanding_and_never_shuffle() -> None:
    splitter = PurgedWalkForwardSplitter(temporal_split())
    partition = splitter.partition(training_dataset().examples)
    folds = splitter.walk_forward_folds(partition)
    assert len(folds) == 3
    assert [len(fold.training) for fold in folds] == sorted(
        len(fold.training) for fold in folds
    )
    assert not hasattr(splitter, "shuffle")
    assert all(
        item.label_end_timestamp < fold.validation_start
        for fold in folds
        for item in fold.training
    )
