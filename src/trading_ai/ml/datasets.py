"""Signal-level supervised datasets built from point-in-time strategy candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_ai.backtesting.models import BacktestDataset, StrategySignalAction
from trading_ai.core.models import BacktestResult
from trading_ai.features import FeatureEngine
from trading_ai.ml.exceptions import MLDataError
from trading_ai.ml.features import MLFeatureBuilder
from trading_ai.ml.inputs import TabularModelInput
from trading_ai.ml.labels import LabelBuilder, LabelConfig


@dataclass(frozen=True, slots=True)
class SignalTrainingExample:
    model_input: TabularModelInput
    target: int
    label_end_timestamp: datetime
    forward_return: float

    def __post_init__(self) -> None:
        if self.target not in {0, 1}:
            raise ValueError("training target must be binary")
        if self.label_end_timestamp <= self.model_input.timestamp:
            raise ValueError("label_end_timestamp must be after model input")


@dataclass(frozen=True, slots=True)
class SignalTrainingDataset:
    strategy_name: str
    strategy_version: str
    timeframe: str
    feature_names: tuple[str, ...]
    label_config: tuple[tuple[str, str], ...]
    dataset_ids: tuple[str, ...]
    dataset_checksums: tuple[tuple[str, str], ...]
    examples: tuple[SignalTrainingExample, ...]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("signal training dataset must not be empty")
        ordered = tuple(
            sorted(
                self.examples,
                key=lambda item: (item.model_input.timestamp, item.model_input.symbol),
            )
        )
        if ordered != self.examples:
            raise ValueError("training examples must be deterministic and chronological")
        if any(
            item.model_input.strategy_name != self.strategy_name
            or item.model_input.strategy_version != self.strategy_version
            or item.model_input.timeframe != self.timeframe
            or item.model_input.feature_names != self.feature_names
            for item in self.examples
        ):
            raise ValueError("training example scope/schema must match the dataset")
        if tuple(sorted(set(self.dataset_ids))) != self.dataset_ids:
            raise ValueError("dataset_ids must be sorted and unique")
        if tuple(sorted(self.dataset_checksums)) != self.dataset_checksums:
            raise ValueError("dataset_checksums must be sorted")


@dataclass(frozen=True, slots=True)
class DatasetBuildReport:
    candidate_signals: int
    labeled_examples: int
    dropped_unlabeled: int
    dropped_missing_features: int


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    dataset: SignalTrainingDataset
    report: DatasetBuildReport


class SignalTrainingDatasetBuilder:
    """Rebuild X at signal time and isolate all future use inside LabelBuilder."""

    def __init__(
        self,
        *,
        feature_builder: MLFeatureBuilder | None = None,
        label_config: LabelConfig | None = None,
    ) -> None:
        self.feature_builder = feature_builder or MLFeatureBuilder()
        self.label_builder = LabelBuilder(label_config)

    def build(
        self,
        result: BacktestResult,
        datasets: tuple[BacktestDataset, ...],
    ) -> DatasetBuildResult:
        candidates = tuple(
            signal
            for signal in result.signals
            if signal.action is StrategySignalAction.ENTER_LONG
        )
        if not candidates:
            raise MLDataError("strategy replay produced no ENTER_LONG candidates")
        bars_by_symbol = {
            dataset.reference.symbol: dataset.bars for dataset in datasets
        }
        regimes = {
            (snapshot.symbol, snapshot.timeframe, snapshot.timestamp): snapshot
            for snapshot in result.regime_snapshots
        }
        feature_engine = FeatureEngine()
        examples: list[SignalTrainingExample] = []
        dropped_unlabeled = 0
        dropped_missing = 0
        for signal in candidates:
            bars = bars_by_symbol.get(signal.symbol)
            if bars is None:
                raise MLDataError(f"no source bars for signal symbol {signal.symbol}")
            index_by_timestamp = {bar.timestamp: index for index, bar in enumerate(bars)}
            index = index_by_timestamp.get(signal.timestamp)
            regime = regimes.get((signal.symbol, signal.timeframe, signal.timestamp))
            if index is None or regime is None:
                raise MLDataError("signal lacks exact bar or regime lineage")
            label = self.label_builder.build(bars, index)
            if label is None:
                dropped_unlabeled += 1
                continue
            features = feature_engine.compute(
                bars[: index + 1],
                self.feature_builder.feature_request,
                as_of=signal.timestamp,
            )
            try:
                model_input = self.feature_builder.build(
                    signal=signal,
                    features=features,
                    regime=regime,
                )
            except MLDataError:
                dropped_missing += 1
                continue
            examples.append(
                SignalTrainingExample(
                    model_input=model_input,
                    target=label.target,
                    label_end_timestamp=label.label_end_timestamp,
                    forward_return=label.forward_return,
                )
            )
        if not examples:
            raise MLDataError("no complete labeled signal examples were produced")
        references = tuple(sorted(result.dataset_references, key=lambda item: item.dataset_id))
        dataset = SignalTrainingDataset(
            strategy_name=result.strategy_name,
            strategy_version=result.strategy_version,
            timeframe=examples[0].model_input.timeframe,
            feature_names=examples[0].model_input.feature_names,
            label_config=self.label_builder.config.to_parameters(),
            dataset_ids=tuple(reference.dataset_id for reference in references),
            dataset_checksums=tuple(
                (reference.dataset_id, reference.checksum_sha256)
                for reference in references
            ),
            examples=tuple(
                sorted(
                    examples,
                    key=lambda item: (
                        item.model_input.timestamp,
                        item.model_input.symbol,
                    ),
                )
            ),
        )
        return DatasetBuildResult(
            dataset=dataset,
            report=DatasetBuildReport(
                candidate_signals=len(candidates),
                labeled_examples=len(examples),
                dropped_unlabeled=dropped_unlabeled,
                dropped_missing_features=dropped_missing,
            ),
        )
