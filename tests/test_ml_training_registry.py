from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from ml_support import tabular_input, temporal_split, training_dataset
from trading_ai.ml.adapters.sklearn import SklearnClassifierAdapter
from trading_ai.ml.exceptions import MLDataError, MLPromotionError, MLRegistryError, MLTrainingError
from trading_ai.ml.inference import InferenceEngine
from trading_ai.ml.evaluation import evaluate_binary_classification
from trading_ai.ml.models import ModelConfig, ModelFamily, ModelStatus
from trading_ai.ml.registry import LocalModelRegistry
from trading_ai.ml.training import TrainingConfig, TrainingOutcome, TrainingPipeline


def _pipeline() -> TrainingPipeline:
    return TrainingPipeline(
        config=TrainingConfig(
            minimum_training_samples=40,
            minimum_samples_per_class=10,
            calibration_bins=5,
        )
    )


def _outcome(family: ModelFamily = ModelFamily.LOGISTIC, *, seed: int = 42):
    return _pipeline().run(
        training_dataset(),
        split_config=temporal_split(),
        model_config=ModelConfig(family=family, random_state=seed),
    )


@pytest.mark.parametrize("family", tuple(ModelFamily))
def test_three_tabular_models_train_deterministically_with_valid_metrics(family) -> None:
    first = _outcome(family)
    second = _outcome(family)
    inputs = tuple(item.model_input for item in training_dataset().examples[95:105])

    assert first.artifact.status is ModelStatus.CANDIDATE
    assert first.artifact.model_family is family
    assert first.artifact.input_kind.value == "TABULAR"
    assert first.artifact.framework == "scikit-learn"
    assert first.artifact.runtime == "CPU"
    assert first.adapter.score_batch(inputs) == pytest.approx(
        second.adapter.score_batch(inputs), abs=1e-12
    )
    assert first.artifact.artifact_checksum == second.artifact.artifact_checksum
    assert first.artifact.model_id == second.artifact.model_id
    for metrics in (first.evaluation.validation, first.evaluation.final_test):
        assert metrics.sample_count > 0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert sum(count for _, count in metrics.class_distribution) == metrics.sample_count
    assert len(first.evaluation.feature_interpretation) == len(
        first.artifact.feature_names
    )


def test_logistic_scaler_is_fit_on_train_only_not_future_outlier() -> None:
    normal = _pipeline().run(
        training_dataset(future_outlier=False),
        split_config=temporal_split(),
        model_config=ModelConfig(family=ModelFamily.LOGISTIC),
    )
    extreme_future = _pipeline().run(
        training_dataset(future_outlier=True),
        split_config=temporal_split(),
        model_config=ModelConfig(family=ModelFamily.LOGISTIC),
    )
    assert isinstance(normal.adapter, SklearnClassifierAdapter)
    assert normal.adapter.preprocessing_mean == pytest.approx(
        extreme_future.adapter.preprocessing_mean, abs=1e-15
    )


def test_training_refuses_too_few_samples_and_single_class() -> None:
    with pytest.raises(MLTrainingError, match="at least"):
        TrainingPipeline(
            config=TrainingConfig(
                minimum_training_samples=1_000, minimum_samples_per_class=1
            )
        ).run(training_dataset(), split_config=temporal_split())

    dataset = training_dataset()
    one_class = replace(
        dataset,
        examples=tuple(replace(item, target=1) for item in dataset.examples),
    )
    with pytest.raises(MLTrainingError, match="both classes"):
        _pipeline().run(one_class, split_config=temporal_split())


def test_adapter_round_trip_and_score_one_equals_score_batch() -> None:
    outcome = _outcome()
    payload = outcome.adapter.serialize()
    restored = SklearnClassifierAdapter.deserialize(payload)
    inputs = tuple(item.model_input for item in training_dataset().examples[95:100])

    batch = restored.score_batch(inputs)
    assert batch == pytest.approx(outcome.adapter.score_batch(inputs), abs=1e-15)
    assert tuple(restored.score_one(item) for item in inputs) == pytest.approx(
        batch, abs=1e-15
    )
    engine = InferenceEngine(outcome.artifact, restored)
    one = engine.score_one(inputs[0])
    batched = engine.score_batch(inputs)[0]
    assert one.probability_positive == pytest.approx(
        batched.probability_positive, abs=1e-15
    )
    assert 0.0 <= one.probability_positive <= 1.0


def test_registry_save_list_inspect_reload_and_compatibility(tmp_path) -> None:
    outcome = _outcome()
    registry = LocalModelRegistry(tmp_path / "ml")
    registry.save(outcome)

    assert registry.list() == (outcome.artifact,)
    inspected = registry.inspect(outcome.artifact.model_id)
    assert inspected["integrity"] == "verified"
    artifact, adapter, evaluation = registry.load(
        outcome.artifact.model_id,
        strategy_name="trend",
        strategy_version="1.0",
        timeframe="1d",
        feature_schema_version="1.1",
        ml_feature_schema_version="1.0",
    )
    assert artifact == outcome.artifact
    assert evaluation == outcome.evaluation
    assert adapter.score_one(tabular_input(100)) == pytest.approx(
        outcome.adapter.score_one(tabular_input(100)), abs=1e-15
    )
    with pytest.raises(MLRegistryError, match="compatibility"):
        registry.load(outcome.artifact.model_id, timeframe="1h")


def test_registry_rejects_traversal_missing_and_corrupt_payload(tmp_path) -> None:
    registry = LocalModelRegistry(tmp_path / "ml")
    with pytest.raises(MLRegistryError, match="identifier"):
        registry.load("../escape")
    with pytest.raises(MLRegistryError, match="unknown model"):
        registry.load("missing-model")

    outcome = _outcome()
    registry.save(outcome)
    model_path = tmp_path / "ml" / "models" / outcome.artifact.model_id / "model.bin"
    model_path.write_bytes(b"corrupt")
    with pytest.raises(MLRegistryError, match="checksum"):
        registry.load(outcome.artifact.model_id)


def test_promotion_is_explicit_ordered_audited_and_never_automatic(tmp_path) -> None:
    registry = LocalModelRegistry(tmp_path / "ml")
    outcome = _outcome()
    registry.save(outcome)
    assert registry.inspect(outcome.artifact.model_id)["artifact"]["status"] == "CANDIDATE"

    with pytest.raises(MLPromotionError, match="invalid"):
        registry.promote(
            outcome.artifact.model_id, ModelStatus.APPROVED, reason="no direct jump"
        )
    validated = registry.promote(
        outcome.artifact.model_id,
        ModelStatus.VALIDATED,
        reason="temporal validation reviewed",
    )
    approved = registry.promote(
        outcome.artifact.model_id,
        ModelStatus.APPROVED,
        reason="explicit research approval",
    )
    assert validated.status is ModelStatus.VALIDATED
    assert approved.status is ModelStatus.APPROVED
    assert [event["event_type"] for event in registry.audit_events] == [
        "TRAINED", "VALIDATED", "APPROVED"
    ]


def test_registry_explicit_rollback_restores_prior_approved_alias(tmp_path) -> None:
    registry = LocalModelRegistry(tmp_path / "ml")
    first = _outcome(seed=41)
    second = _outcome(seed=43)
    for outcome in (first, second):
        registry.save(outcome)
        registry.promote(
            outcome.artifact.model_id, ModelStatus.VALIDATED, reason="reviewed"
        )
        registry.promote(
            outcome.artifact.model_id, ModelStatus.APPROVED, reason="approved"
        )
    assert registry.load_approved("trend", "1d")[0].model_id == second.artifact.model_id

    rolled_back = registry.rollback("trend", "1d", reason="explicit rollback test")
    assert rolled_back.model_id == first.artifact.model_id
    assert registry.load_approved("trend", "1d")[0].model_id == first.artifact.model_id
    assert registry.audit_events[-1]["event_type"] == "ROLLBACK"


def test_model_feature_order_mismatch_fails_before_inference() -> None:
    outcome = _outcome()
    engine = InferenceEngine(outcome.artifact, outcome.adapter)
    wrong = replace(
        tabular_input(100),
        values=tuple(reversed(tabular_input(100).values)),
    )
    with pytest.raises(Exception, match="feature"):
        engine.score_one(wrong)


def test_classification_metrics_and_calibration_are_explicit_and_immutable() -> None:
    metrics = evaluate_binary_classification(
        (0, 0, 1, 1), (0.1, 0.6, 0.7, 0.9), calibration_bins=2
    )
    assert metrics.sample_count == 4
    assert metrics.confusion_matrix == ((1, 1), (0, 2))
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert sum(item.sample_count for item in metrics.calibration) == 4
    with pytest.raises(FrozenInstanceError):
        metrics.accuracy = 1.0


def test_evaluation_handles_single_class_auc_as_unavailable() -> None:
    metrics = evaluate_binary_classification((1, 1), (0.7, 0.8), calibration_bins=2)
    assert metrics.roc_auc is None
    assert metrics.pr_auc is None
    assert metrics.balanced_accuracy == pytest.approx(metrics.accuracy)
