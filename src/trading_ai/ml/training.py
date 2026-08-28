"""Explicit, purged temporal model-training pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from trading_ai.backtesting.reproducibility import detect_git_commit, source_tree_hash, stable_hash
from trading_ai.core.config import PROJECT_ROOT
from trading_ai.ml.adapters.sklearn import SklearnClassifierTrainer
from trading_ai.ml.base import ModelAdapter, ModelTrainer
from trading_ai.ml.datasets import SignalTrainingDataset
from trading_ai.ml.evaluation import EvaluationReport, evaluate_binary_classification
from trading_ai.ml.exceptions import MLTrainingError
from trading_ai.ml.models import ModelArtifact, ModelConfig, ModelStatus
from trading_ai.ml.splits import PurgedWalkForwardSplitter, TemporalSplitConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    minimum_training_samples: int = 100
    minimum_samples_per_class: int = 20
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_training_samples", "minimum_samples_per_class", "calibration_bins"
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    artifact: ModelArtifact
    adapter: ModelAdapter
    evaluation: EvaluationReport


class TrainingPipeline:
    """Fit fixed baselines on TRAIN and evaluate later periods without leakage."""

    def __init__(
        self,
        *,
        trainer: ModelTrainer | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.trainer = trainer or SklearnClassifierTrainer()
        self.config = config or TrainingConfig()

    def _validate_training(self, examples: tuple[object, ...]) -> None:
        if len(examples) < self.config.minimum_training_samples:
            raise MLTrainingError(
                f"training requires at least {self.config.minimum_training_samples} samples"
            )
        targets = tuple(item.target for item in examples)
        for label in (0, 1):
            if targets.count(label) < self.config.minimum_samples_per_class:
                raise MLTrainingError(
                    "training requires both classes with at least "
                    f"{self.config.minimum_samples_per_class} samples each"
                )

    @staticmethod
    def _fit(
        trainer: ModelTrainer, examples: tuple[object, ...], config: ModelConfig
    ) -> ModelAdapter:
        return trainer.fit(
            tuple(item.model_input for item in examples),
            tuple(item.target for item in examples),
            config,
        )

    def run(
        self,
        dataset: SignalTrainingDataset,
        *,
        split_config: TemporalSplitConfig,
        model_config: ModelConfig | None = None,
    ) -> TrainingOutcome:
        definition = model_config or ModelConfig()
        splitter = PurgedWalkForwardSplitter(split_config)
        partition = splitter.partition(dataset.examples)
        self._validate_training(partition.training)
        folds = splitter.walk_forward_folds(partition)
        validation_targets: list[int] = []
        validation_probabilities: list[float] = []
        for fold in folds:
            if set(item.target for item in fold.training) != {0, 1}:
                raise MLTrainingError("each walk-forward TRAIN fold needs both classes")
            fold_adapter = self._fit(self.trainer, fold.training, definition)
            probabilities = fold_adapter.score_batch(
                tuple(item.model_input for item in fold.validation)
            )
            validation_targets.extend(item.target for item in fold.validation)
            validation_probabilities.extend(probabilities)
        final_adapter = self._fit(self.trainer, partition.training, definition)
        test_probabilities = final_adapter.score_batch(
            tuple(item.model_input for item in partition.final_test)
        )
        report = EvaluationReport(
            validation=evaluate_binary_classification(
                tuple(validation_targets), tuple(validation_probabilities),
                calibration_bins=self.config.calibration_bins,
            ),
            final_test=evaluate_binary_classification(
                tuple(item.target for item in partition.final_test), test_probabilities,
                calibration_bins=self.config.calibration_bins,
            ),
            walk_forward_folds=len(folds),
            interpretation_kind=(
                "logistic_coefficients"
                if definition.family.value == "logistic"
                else "feature_importance"
            ),
            feature_interpretation=final_adapter.interpretation(),
            leakage_safeguards=tuple(sorted((
                "embargo at validation and test boundaries",
                "expanding chronological walk-forward validation",
                "final test excluded from fitting and preprocessing",
                "label-overlap purge across temporal boundaries",
                "shared calendar boundaries across symbols",
                "training transformations fitted on fold TRAIN only",
            ))),
        )
        payload = final_adapter.serialize()
        checksum = hashlib.sha256(payload).hexdigest()
        source_hash = source_tree_hash(PROJECT_ROOT)
        identity = {
            "family": definition.family,
            "version": definition.model_version,
            "strategy": (dataset.strategy_name, dataset.strategy_version),
            "timeframe": dataset.timeframe,
            "features": dataset.feature_names,
            "label": dataset.label_config,
            "split": split_config.to_parameters(),
            "config": definition.to_parameters(),
            "datasets": dataset.dataset_checksums,
            "source_hash": source_hash,
            "artifact_checksum": checksum,
        }
        artifact = ModelArtifact(
            model_id=f"ml-{stable_hash(identity)[:24]}",
            model_family=definition.family,
            model_version=definition.model_version,
            task=final_adapter.task,
            input_kind=final_adapter.input_kind,
            strategy_name=dataset.strategy_name,
            strategy_version=dataset.strategy_version,
            timeframe=dataset.timeframe,
            status=ModelStatus.CANDIDATE,
            feature_schema_version=dataset.examples[0].model_input.feature_schema_version,
            ml_feature_schema_version=dataset.examples[0].model_input.ml_feature_schema_version,
            feature_names=dataset.feature_names,
            label_config=dataset.label_config,
            split_config=split_config.to_parameters(),
            model_config=definition.to_parameters(),
            training_period=split_config.training,
            validation_period=split_config.validation,
            test_period=split_config.final_test,
            dataset_ids=dataset.dataset_ids,
            dataset_checksums=dataset.dataset_checksums,
            git_sha=detect_git_commit(PROJECT_ROOT),
            source_hash=source_hash,
            framework=final_adapter.framework,
            framework_version=final_adapter.framework_version,
            artifact_checksum=checksum,
            created_at=datetime.now(timezone.utc),
            runtime="CPU",
        )
        if hashlib.sha256(final_adapter.serialize()).hexdigest() != checksum:
            raise MLTrainingError("model serialization is not deterministic in this run")
        return TrainingOutcome(artifact, final_adapter, report)
