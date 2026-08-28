"""Integrity-checking local model registry with explicit lifecycle audit."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_ai.backtesting.reproducibility import stable_hash, to_primitive
from trading_ai.ml.adapters.sklearn import SklearnClassifierAdapter
from trading_ai.ml.base import ModelAdapter, ModelRegistry
from trading_ai.ml.evaluation import CalibrationBin, ClassificationMetrics, EvaluationReport
from trading_ai.ml.exceptions import MLPromotionError, MLRegistryError
from trading_ai.ml.models import (
    InputKind, MLTask, ModelArtifact, ModelFamily, ModelStatus,
    RegistryEvent, RegistryEventType, TimeRange,
)
from trading_ai.ml.storage import read_json, safe_child, sha256_bytes, write_json
from trading_ai.ml.training import TrainingOutcome


def _time_range(payload: dict[str, str]) -> TimeRange:
    return TimeRange(
        datetime.fromisoformat(payload["start"]),
        datetime.fromisoformat(payload["end"]),
    )


def _artifact(payload: dict[str, Any]) -> ModelArtifact:
    return ModelArtifact(
        model_id=payload["model_id"],
        model_family=ModelFamily(payload["model_family"]),
        model_version=payload["model_version"],
        task=MLTask(payload["task"]),
        input_kind=InputKind(payload["input_kind"]),
        strategy_name=payload["strategy_name"],
        strategy_version=payload["strategy_version"],
        timeframe=payload["timeframe"],
        status=ModelStatus(payload["status"]),
        feature_schema_version=payload["feature_schema_version"],
        ml_feature_schema_version=payload["ml_feature_schema_version"],
        feature_names=tuple(payload["feature_names"]),
        label_config=tuple(tuple(item) for item in payload["label_config"]),
        split_config=tuple(tuple(item) for item in payload["split_config"]),
        model_config=tuple(tuple(item) for item in payload["model_config"]),
        training_period=_time_range(payload["training_period"]),
        validation_period=_time_range(payload["validation_period"]),
        test_period=_time_range(payload["test_period"]),
        dataset_ids=tuple(payload["dataset_ids"]),
        dataset_checksums=tuple(tuple(item) for item in payload["dataset_checksums"]),
        git_sha=payload.get("git_sha"),
        source_hash=payload["source_hash"],
        framework=payload["framework"],
        framework_version=payload["framework_version"],
        artifact_checksum=payload["artifact_checksum"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        runtime=payload.get("runtime", "CPU"),
    )


def _metrics(payload: dict[str, Any]) -> ClassificationMetrics:
    return ClassificationMetrics(
        sample_count=payload["sample_count"],
        class_distribution=tuple(tuple(item) for item in payload["class_distribution"]),
        accuracy=payload["accuracy"],
        balanced_accuracy=payload["balanced_accuracy"],
        precision=payload["precision"],
        recall=payload["recall"],
        f1=payload["f1"],
        roc_auc=payload["roc_auc"],
        pr_auc=payload["pr_auc"],
        log_loss=payload["log_loss"],
        brier_score=payload["brier_score"],
        confusion_matrix=tuple(tuple(item) for item in payload["confusion_matrix"]),
        calibration=tuple(CalibrationBin(**item) for item in payload["calibration"]),
    )


def _evaluation(payload: dict[str, Any]) -> EvaluationReport:
    return EvaluationReport(
        validation=_metrics(payload["validation"]),
        final_test=_metrics(payload["final_test"]),
        walk_forward_folds=payload["walk_forward_folds"],
        interpretation_kind=payload["interpretation_kind"],
        feature_interpretation=tuple(tuple(item) for item in payload["feature_interpretation"]),
        leakage_safeguards=tuple(payload["leakage_safeguards"]),
    )


class LocalModelRegistry(ModelRegistry):
    """Load only checksum-verified artifacts addressed by safe model IDs."""

    def __init__(self, root: Path | str = Path("data_local") / "ml") -> None:
        self.root = Path(root)

    def _model_directory(self, model_id: str) -> Path:
        return safe_child(self.root, "models", model_id)

    def _alias_path(self, strategy_name: str, timeframe: str) -> Path:
        key = stable_hash((strategy_name, timeframe))[:24]
        return safe_child(self.root, "aliases", key).with_suffix(".json")

    def _write_event(
        self,
        event_type: RegistryEventType,
        model_id: str,
        reason: str,
        *,
        previous_model_id: str | None = None,
    ) -> RegistryEvent:
        timestamp = datetime.now(timezone.utc)
        event = RegistryEvent(
            event_id=f"ml-event-{stable_hash((event_type, model_id, timestamp, reason, previous_model_id))[:24]}",
            event_type=event_type,
            model_id=model_id,
            timestamp=timestamp,
            reason=reason,
            previous_model_id=previous_model_id,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(to_primitive(event), sort_keys=True) + "\n")
        return event

    def save(self, outcome: TrainingOutcome) -> ModelArtifact:
        artifact = outcome.artifact
        directory = self._model_directory(artifact.model_id)
        if directory.exists():
            raise MLRegistryError(f"model already exists: {artifact.model_id}")
        payload = outcome.adapter.serialize()
        if sha256_bytes(payload) != artifact.artifact_checksum:
            raise MLRegistryError("serialized model checksum differs from artifact")
        directory.mkdir(parents=True, exist_ok=False)
        temporary = directory / "model.bin.tmp"
        temporary.write_bytes(payload)
        temporary.replace(directory / "model.bin")
        write_json(directory / "manifest.json", artifact)
        write_json(directory / "evaluation.json", outcome.evaluation)
        self._write_event(
            RegistryEventType.TRAINED,
            artifact.model_id,
            "Explicit training pipeline created CANDIDATE model.",
        )
        return artifact

    def _load_metadata(self, model_id: str) -> tuple[ModelArtifact, EvaluationReport]:
        directory = self._model_directory(model_id)
        if not directory.is_dir():
            raise MLRegistryError(f"unknown model ID: {model_id}")
        artifact = _artifact(read_json(directory / "manifest.json"))
        if artifact.model_id != model_id:
            raise MLRegistryError("manifest model ID does not match registry path")
        return artifact, _evaluation(read_json(directory / "evaluation.json"))

    def load(
        self, model_id: str, **compatibility: Any
    ) -> tuple[ModelArtifact, ModelAdapter, EvaluationReport]:
        artifact, evaluation = self._load_metadata(model_id)
        try:
            payload = (self._model_directory(model_id) / "model.bin").read_bytes()
        except OSError as exc:
            raise MLRegistryError("model payload is unavailable") from exc
        if sha256_bytes(payload) != artifact.artifact_checksum:
            raise MLRegistryError("model payload checksum verification failed")
        for field_name in (
            "feature_schema_version", "ml_feature_schema_version", "model_family",
            "strategy_name", "strategy_version", "timeframe",
        ):
            expected = compatibility.get(field_name)
            if expected is not None and getattr(artifact, field_name) != expected:
                raise MLRegistryError(f"model {field_name} compatibility check failed")
        if artifact.framework != "scikit-learn" or artifact.input_kind is not InputKind.TABULAR:
            raise MLRegistryError("no installed adapter can load this model artifact")
        adapter = SklearnClassifierAdapter.deserialize(payload)
        if (
            adapter.model_family != artifact.model_family
            or adapter.model_version != artifact.model_version
            or adapter.feature_names != artifact.feature_names
            or adapter.framework_version != artifact.framework_version
        ):
            raise MLRegistryError("model payload metadata does not match manifest")
        return artifact, adapter, evaluation

    def list(self) -> tuple[ModelArtifact, ...]:
        models_root = self.root / "models"
        if not models_root.is_dir():
            return ()
        artifacts = [
            self._load_metadata(directory.name)[0]
            for directory in sorted(models_root.iterdir(), key=lambda item: item.name)
            if directory.is_dir()
        ]
        return tuple(sorted(artifacts, key=lambda item: (item.created_at, item.model_id)))

    def inspect(self, model_id: str) -> dict[str, Any]:
        artifact, _, evaluation = self.load(model_id)
        return {
            "artifact": to_primitive(artifact),
            "evaluation": to_primitive(evaluation),
            "integrity": "verified",
        }

    def _replace_status(self, artifact: ModelArtifact, status: ModelStatus) -> ModelArtifact:
        updated = replace(artifact, status=status)
        write_json(self._model_directory(artifact.model_id) / "manifest.json", updated)
        return updated

    def promote(
        self, model_id: str, target: ModelStatus, *, reason: str
    ) -> ModelArtifact:
        if not reason.strip():
            raise MLPromotionError("promotion reason is required")
        artifact, _, _ = self.load(model_id)
        allowed = {
            ModelStatus.CANDIDATE: ModelStatus.VALIDATED,
            ModelStatus.VALIDATED: ModelStatus.APPROVED,
            ModelStatus.APPROVED: ModelStatus.RETIRED,
        }
        if allowed.get(artifact.status) is not target:
            raise MLPromotionError(
                f"invalid model transition {artifact.status.value} -> {target.value}"
            )
        updated = self._replace_status(artifact, target)
        event_type = RegistryEventType(target.value)
        if target is ModelStatus.APPROVED:
            alias_path = self._alias_path(artifact.strategy_name, artifact.timeframe)
            alias = read_json(alias_path) if alias_path.is_file() else {
                "strategy_name": artifact.strategy_name,
                "timeframe": artifact.timeframe,
                "approved_history": [],
            }
            previous = alias.get("current_model_id")
            history = list(alias.get("approved_history", []))
            if artifact.model_id not in history:
                history.append(artifact.model_id)
            alias.update({"current_model_id": artifact.model_id, "approved_history": history})
            write_json(alias_path, alias)
            self._write_event(event_type, model_id, reason, previous_model_id=previous)
        else:
            self._write_event(event_type, model_id, reason)
        return updated

    def load_approved(
        self, strategy_name: str, timeframe: str
    ) -> tuple[ModelArtifact, ModelAdapter, EvaluationReport]:
        alias_path = self._alias_path(strategy_name, timeframe)
        if not alias_path.is_file():
            raise MLRegistryError("no explicit APPROVED alias exists for this scope")
        model_id = read_json(alias_path).get("current_model_id")
        if not isinstance(model_id, str):
            raise MLRegistryError("approved alias is invalid")
        loaded = self.load(model_id, strategy_name=strategy_name, timeframe=timeframe)
        if loaded[0].status is not ModelStatus.APPROVED:
            raise MLRegistryError("approved alias does not reference an APPROVED model")
        return loaded

    def rollback(self, strategy_name: str, timeframe: str, *, reason: str) -> ModelArtifact:
        if not reason.strip():
            raise MLPromotionError("rollback reason is required")
        alias_path = self._alias_path(strategy_name, timeframe)
        if not alias_path.is_file():
            raise MLPromotionError("no approved alias exists to roll back")
        alias = read_json(alias_path)
        history = list(alias.get("approved_history", []))
        current = alias.get("current_model_id")
        if len(history) < 2 or current != history[-1]:
            raise MLPromotionError("no previous approved model is available")
        previous = history[-2]
        artifact, _, _ = self.load(previous)
        if artifact.status is not ModelStatus.APPROVED:
            raise MLPromotionError("previous model is no longer APPROVED")
        alias["current_model_id"] = previous
        alias["approved_history"] = history[:-1]
        write_json(alias_path, alias)
        self._write_event(
            RegistryEventType.ROLLBACK, previous, reason, previous_model_id=current
        )
        return artifact

    @property
    def audit_events(self) -> tuple[dict[str, Any], ...]:
        path = self.root / "audit.jsonl"
        if not path.is_file():
            return ()
        try:
            return tuple(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except Exception as exc:
            raise MLRegistryError("registry audit log is invalid") from exc
