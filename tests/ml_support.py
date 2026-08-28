"""Small deterministic ML fixtures with no market-network dependency."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from trading_ai.ml.base import ModelAdapter
from trading_ai.ml.datasets import SignalTrainingDataset, SignalTrainingExample
from trading_ai.ml.inputs import ModelInput, TabularModelInput
from trading_ai.ml.models import (
    InputKind,
    MLTask,
    ModelArtifact,
    ModelFamily,
    ModelStatus,
    TimeRange,
)
from trading_ai.ml.splits import TemporalSplitConfig


ML_START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def tabular_input(
    index: int,
    *,
    symbol: str = "AAPL",
    future_outlier: bool = False,
) -> TabularModelInput:
    value = float(index)
    if future_outlier and index >= 65:
        value += 1_000_000.0
    return TabularModelInput(
        symbol=symbol,
        timestamp=ML_START + timedelta(days=index),
        timeframe="1d",
        strategy_name="trend",
        strategy_version="1.0",
        feature_schema_version="1.1",
        ml_feature_schema_version="1.0",
        values=(
            ("x_direction", float((index % 4) < 2)),
            ("x_cycle", float(index % 7) / 7.0),
            ("x_level", value / 100.0),
        ),
    )


def training_dataset(
    *,
    future_outlier: bool = False,
    symbols: tuple[str, ...] = ("AAPL",),
    count: int = 120,
) -> SignalTrainingDataset:
    examples = []
    for index in range(count):
        for symbol in symbols:
            target = int(index % 4 < 2)
            model_input = tabular_input(
                index, symbol=symbol, future_outlier=future_outlier
            )
            examples.append(
                SignalTrainingExample(
                    model_input=model_input,
                    target=target,
                    label_end_timestamp=model_input.timestamp + timedelta(days=1),
                    forward_return=0.01 if target else -0.01,
                )
            )
    return SignalTrainingDataset(
        strategy_name="trend",
        strategy_version="1.0",
        timeframe="1d",
        feature_names=("x_direction", "x_cycle", "x_level"),
        label_config=(
            ("entry_reference", "NEXT_BAR_OPEN"),
            ("exit_reference", "HORIZON_CLOSE"),
            ("horizon_bars", "1"),
            ("minimum_forward_return_bps", "0"),
        ),
        dataset_ids=tuple(f"dataset-{symbol.lower()}" for symbol in sorted(symbols)),
        dataset_checksums=tuple(
            (f"dataset-{symbol.lower()}", hashlib.sha256(symbol.encode()).hexdigest())
            for symbol in sorted(symbols)
        ),
        examples=tuple(sorted(examples, key=lambda item: (
            item.model_input.timestamp, item.model_input.symbol
        ))),
    )


def temporal_split() -> TemporalSplitConfig:
    return TemporalSplitConfig(
        training=TimeRange(ML_START, ML_START + timedelta(days=60)),
        validation=TimeRange(
            ML_START + timedelta(days=65), ML_START + timedelta(days=90)
        ),
        final_test=TimeRange(
            ML_START + timedelta(days=95), ML_START + timedelta(days=120)
        ),
        embargo_bars=1,
        walk_forward_folds=3,
    )


class ConstantAdapter(ModelAdapter):
    """Test-only frozen adapter for ML/backtest gating scenarios."""

    def __init__(
        self,
        probability: float,
        feature_names: tuple[str, ...],
        *,
        input_kind: InputKind = InputKind.TABULAR,
    ) -> None:
        self.probability = probability
        self._feature_names = feature_names
        self._input_kind = input_kind

    @property
    def model_family(self):
        return ModelFamily.LOGISTIC

    @property
    def model_version(self):
        return "1.0"

    @property
    def task(self):
        return MLTask.SIGNAL_QUALITY_BINARY

    @property
    def input_kind(self):
        return self._input_kind

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def framework(self):
        return "test-adapter"

    @property
    def framework_version(self):
        return "1"

    def score_one(self, model_input: ModelInput) -> float:
        del model_input
        return self.probability

    def score_batch(self, model_inputs: Sequence[ModelInput]) -> tuple[float, ...]:
        return tuple(self.probability for _ in model_inputs)

    def serialize(self) -> bytes:
        return f"test:{self.probability}:{','.join(self.feature_names)}".encode()


def model_artifact(
    adapter: ModelAdapter,
    *,
    model_id: str = "ml-test-model",
    status: ModelStatus = ModelStatus.APPROVED,
    strategy_name: str = "breakout",
    strategy_version: str = "1.0",
    timeframe: str = "1d",
    feature_schema_version: str = "1.1",
) -> ModelArtifact:
    payload = adapter.serialize()
    return ModelArtifact(
        model_id=model_id,
        model_family=ModelFamily.LOGISTIC,
        model_version="1.0",
        task=MLTask.SIGNAL_QUALITY_BINARY,
        input_kind=adapter.input_kind,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        timeframe=timeframe,
        status=status,
        feature_schema_version=feature_schema_version,
        ml_feature_schema_version="1.0",
        feature_names=adapter.feature_names,
        label_config=(
            ("entry_reference", "NEXT_BAR_OPEN"),
            ("exit_reference", "HORIZON_CLOSE"),
            ("horizon_bars", "5"),
            ("minimum_forward_return_bps", "0"),
        ),
        split_config=(
            ("embargo_bars", "1"),
            ("walk_forward_folds", "3"),
        ),
        model_config=(("family", "logistic"), ("random_state", "42")),
        training_period=TimeRange(
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2019, 1, 1, tzinfo=timezone.utc),
        ),
        validation_period=TimeRange(
            datetime(2019, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
        ),
        test_period=TimeRange(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2021, 1, 1, tzinfo=timezone.utc),
        ),
        dataset_ids=("dataset-test",),
        dataset_checksums=(("dataset-test", "d" * 64),),
        git_sha=None,
        source_hash="a" * 64,
        framework=adapter.framework,
        framework_version=adapter.framework_version,
        artifact_checksum=hashlib.sha256(payload).hexdigest(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
