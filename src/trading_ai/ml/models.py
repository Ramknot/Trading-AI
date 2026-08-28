"""Immutable framework-neutral ML metadata and lifecycle models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

from trading_ai.ml.exceptions import MLConfigurationError


def _text(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _aware_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")


class InputKind(str, Enum):
    TABULAR = "TABULAR"
    SEQUENCE = "SEQUENCE"
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    MULTIMODAL = "MULTIMODAL"


class MLTask(str, Enum):
    SIGNAL_QUALITY_BINARY = "SIGNAL_QUALITY_BINARY"
    MULTICLASS = "MULTICLASS"
    REGRESSION = "REGRESSION"
    SEQUENCE_FORECAST = "SEQUENCE_FORECAST"
    MULTIMODAL = "MULTIMODAL"


class ModelFamily(str, Enum):
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random-forest"
    GRADIENT_BOOSTING = "gradient-boosting"


class ModelStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"


class MLMode(str, Enum):
    DISABLED = "DISABLED"
    SCORE_ONLY = "SCORE_ONLY"
    FILTER = "FILTER"


class MLFilterStatus(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class InferenceMode(str, Enum):
    ONE = "ONE"
    BATCH = "BATCH"


class RegistryEventType(str, Enum):
    TRAINED = "TRAINED"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True, slots=True)
class TimeRange:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        _aware_utc(self.start, "start")
        _aware_utc(self.end, "end")
        if self.end <= self.start:
            raise ValueError("time range end must be after start")

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp < self.end


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Fixed, non-optimized baseline parameters for one tabular model."""

    family: ModelFamily = ModelFamily.LOGISTIC
    model_version: str = "1.0"
    random_state: int = 42
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    random_forest_estimators: int = 100
    random_forest_max_depth: int = 5
    random_forest_min_samples_leaf: int = 2
    gradient_boosting_estimators: int = 100
    gradient_boosting_learning_rate: float = 0.05
    gradient_boosting_max_depth: int = 3

    def __post_init__(self) -> None:
        _text(self.model_version, "model_version")
        if type(self.random_state) is not int:
            raise MLConfigurationError("random_state must be an integer")
        if not isfinite(self.logistic_c) or self.logistic_c <= 0:
            raise MLConfigurationError("logistic_c must be positive and finite")
        for name in (
            "logistic_max_iter",
            "random_forest_estimators",
            "random_forest_max_depth",
            "random_forest_min_samples_leaf",
            "gradient_boosting_estimators",
            "gradient_boosting_max_depth",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise MLConfigurationError(f"{name} must be a positive integer")
        if (
            not isfinite(self.gradient_boosting_learning_rate)
            or self.gradient_boosting_learning_rate <= 0
        ):
            raise MLConfigurationError(
                "gradient_boosting_learning_rate must be positive and finite"
            )

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        common = [
            ("family", self.family.value),
            ("model_version", self.model_version),
            ("random_state", str(self.random_state)),
        ]
        if self.family is ModelFamily.LOGISTIC:
            common.extend(
                (
                    ("c", str(self.logistic_c)),
                    ("max_iter", str(self.logistic_max_iter)),
                    ("preprocessing", "StandardScaler fitted on TRAIN only"),
                )
            )
        elif self.family is ModelFamily.RANDOM_FOREST:
            common.extend(
                (
                    ("estimators", str(self.random_forest_estimators)),
                    ("max_depth", str(self.random_forest_max_depth)),
                    ("min_samples_leaf", str(self.random_forest_min_samples_leaf)),
                )
            )
        elif self.family is ModelFamily.GRADIENT_BOOSTING:
            common.extend(
                (
                    ("estimators", str(self.gradient_boosting_estimators)),
                    ("learning_rate", str(self.gradient_boosting_learning_rate)),
                    ("max_depth", str(self.gradient_boosting_max_depth)),
                )
            )
        return tuple(sorted(common))


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Frozen provenance for one serialized model payload."""

    model_id: str
    model_family: ModelFamily
    model_version: str
    task: MLTask
    input_kind: InputKind
    strategy_name: str
    strategy_version: str
    timeframe: str
    status: ModelStatus
    feature_schema_version: str
    ml_feature_schema_version: str
    feature_names: tuple[str, ...]
    label_config: tuple[tuple[str, str], ...]
    split_config: tuple[tuple[str, str], ...]
    model_config: tuple[tuple[str, str], ...]
    training_period: TimeRange
    validation_period: TimeRange
    test_period: TimeRange
    dataset_ids: tuple[str, ...]
    dataset_checksums: tuple[tuple[str, str], ...]
    git_sha: str | None
    source_hash: str
    framework: str
    framework_version: str
    artifact_checksum: str
    created_at: datetime
    runtime: str = "CPU"

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_version",
            "strategy_name",
            "strategy_version",
            "timeframe",
            "feature_schema_version",
            "ml_feature_schema_version",
            "framework",
            "framework_version",
            "runtime",
        ):
            _text(getattr(self, name), name)
        _aware_utc(self.created_at, "created_at")
        _sha256(self.source_hash, "source_hash")
        _sha256(self.artifact_checksum, "artifact_checksum")
        if self.git_sha is not None:
            if len(self.git_sha) != 40 or any(
                character not in "0123456789abcdef"
                for character in self.git_sha.lower()
            ):
                raise ValueError("git_sha must be a 40-character hexadecimal commit")
        if not self.feature_names or len(self.feature_names) != len(
            set(self.feature_names)
        ):
            raise ValueError("feature_names must be non-empty and unique")
        for name in self.feature_names:
            _text(name, "feature name")
        for field_name in ("label_config", "split_config", "model_config"):
            values = getattr(self, field_name)
            if tuple(sorted(values)) != values:
                raise ValueError(f"{field_name} must be deterministically sorted")
        if tuple(sorted(set(self.dataset_ids))) != self.dataset_ids:
            raise ValueError("dataset_ids must be sorted and unique")
        if tuple(sorted(self.dataset_checksums)) != self.dataset_checksums:
            raise ValueError("dataset_checksums must be deterministically sorted")
        if {name for name, _ in self.dataset_checksums} != set(self.dataset_ids):
            raise ValueError("every dataset ID must have exactly one checksum")
        for _, checksum in self.dataset_checksums:
            _sha256(checksum, "dataset checksum")
        if self.training_period.end > self.validation_period.start:
            raise ValueError("training and validation periods must not overlap")
        if self.validation_period.end > self.test_period.start:
            raise ValueError("validation and final-test periods must not overlap")


@dataclass(frozen=True, slots=True)
class RegistryEvent:
    event_id: str
    event_type: RegistryEventType
    model_id: str
    timestamp: datetime
    reason: str
    previous_model_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "model_id", "reason"):
            _text(getattr(self, name), name)
        _aware_utc(self.timestamp, "timestamp")
        if self.previous_model_id is not None:
            _text(self.previous_model_id, "previous_model_id")
