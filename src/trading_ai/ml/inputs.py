"""Framework-neutral model input contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from math import isfinite
from typing import Protocol, runtime_checkable

from trading_ai.ml.models import InferenceMode, InputKind


@runtime_checkable
class ModelInput(Protocol):
    """Minimal port accepted by current and future model adapters."""

    @property
    def input_kind(self) -> InputKind: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def input_hash(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TabularModelInput:
    symbol: str
    timestamp: datetime
    timeframe: str
    strategy_name: str
    strategy_version: str
    feature_schema_version: str
    ml_feature_schema_version: str
    values: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for name in (
            "symbol",
            "timeframe",
            "strategy_name",
            "strategy_version",
            "feature_schema_version",
            "ml_feature_schema_version",
        ):
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("normalized model inputs must use UTC")
        if not self.values:
            raise ValueError("tabular input must contain features")
        names = [name for name, _ in self.values]
        if len(names) != len(set(names)):
            raise ValueError("tabular feature names must be unique")
        if any(not name.strip() or not isfinite(value) for name, value in self.values):
            raise ValueError("tabular features must have names and finite values")

    @property
    def input_kind(self) -> InputKind:
        return InputKind.TABULAR

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.values)

    @property
    def numeric_values(self) -> tuple[float, ...]:
        return tuple(value for _, value in self.values)

    @property
    def input_hash(self) -> str:
        payload = json.dumps(
            {
                "symbol": self.symbol,
                "timestamp": self.timestamp.isoformat(),
                "timeframe": self.timeframe,
                "strategy_name": self.strategy_name,
                "strategy_version": self.strategy_version,
                "feature_schema_version": self.feature_schema_version,
                "ml_feature_schema_version": self.ml_feature_schema_version,
                "values": self.values,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Lightweight future real-time port request; no transport is implemented."""

    model_input: ModelInput
    mode: InferenceMode = InferenceMode.ONE
