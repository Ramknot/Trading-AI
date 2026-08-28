"""Immutable inference and signal-filter lineage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from trading_ai.backtesting.reproducibility import stable_hash
from trading_ai.ml.models import InferenceMode, MLFilterStatus, MLMode, ModelFamily


def _utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("normalized ML timestamps must use UTC")


def _digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{field_name} must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class MLPrediction:
    prediction_id: str
    timestamp: datetime
    symbol: str
    strategy_name: str
    strategy_version: str
    model_id: str
    model_family: ModelFamily
    model_version: str
    probability_positive: float
    feature_schema_version: str
    ml_feature_schema_version: str
    input_hash: str
    model_artifact_hash: str
    inference_mode: InferenceMode
    technical_latency_ms: float | None = None

    def __post_init__(self) -> None:
        _utc(self.timestamp)
        for field_name in (
            "prediction_id", "symbol", "strategy_name", "strategy_version",
            "model_id", "model_version", "feature_schema_version",
            "ml_feature_schema_version",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not isfinite(self.probability_positive) or not 0.0 <= self.probability_positive <= 1.0:
            raise ValueError("probability_positive must be in [0, 1]")
        _digest(self.input_hash, "input_hash")
        _digest(self.model_artifact_hash, "model_artifact_hash")
        if self.technical_latency_ms is not None and (
            not isfinite(self.technical_latency_ms) or self.technical_latency_ms < 0
        ):
            raise ValueError("technical_latency_ms must be finite and non-negative")

    @classmethod
    def create(cls, **values) -> MLPrediction:
        identity = dict(values)
        identity.pop("technical_latency_ms", None)
        return cls(
            prediction_id=f"ml-pred-{stable_hash(identity)[:24]}", **values
        )


@dataclass(frozen=True, slots=True)
class MLFilterDecision:
    decision_id: str
    timestamp: datetime
    symbol: str
    signal_id: str
    mode: MLMode
    status: MLFilterStatus
    threshold: float | None
    probability: float | None
    reason_code: str
    human_reason: str
    model_id: str | None
    prediction_id: str | None

    def __post_init__(self) -> None:
        _utc(self.timestamp)
        for field_name in (
            "decision_id", "symbol", "signal_id", "reason_code", "human_reason"
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("threshold", "probability"):
            value = getattr(self, field_name)
            if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.status in {MLFilterStatus.PASS, MLFilterStatus.BLOCK} and (
            self.probability is None or self.prediction_id is None
        ):
            raise ValueError("PASS/BLOCK decisions require prediction lineage")
        if self.mode is MLMode.FILTER and self.threshold is None:
            raise ValueError("FILTER decisions require a threshold")

    @classmethod
    def create(cls, **values) -> MLFilterDecision:
        return cls(
            decision_id=f"ml-decision-{stable_hash(values)[:24]}", **values
        )
