"""Immutable models for stable, look-ahead-safe quantitative features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite


FEATURE_SCHEMA_VERSION = "1.0"
FEATURE_ENGINE_VERSION = "1.0"


def _normalized_windows(values: tuple[int, ...], field_name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError(f"{field_name} must contain positive integers")
    return tuple(sorted(set(values)))


def _require_aware(timestamp: datetime, field_name: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    """Serializable feature definition; all windows are expressed in bars."""

    sma_windows: tuple[int, ...] = (20, 50)
    ema_windows: tuple[int, ...] = (20, 50)
    return_windows: tuple[int, ...] = (1, 20, 60)
    structure_windows: tuple[int, ...] = (20,)
    slope_lookback: int = 5
    atr_window: int = 14
    volatility_window: int = 20
    volume_window: int = 20

    def __post_init__(self) -> None:
        for field_name in (
            "sma_windows",
            "ema_windows",
            "return_windows",
            "structure_windows",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalized_windows(getattr(self, field_name), field_name),
            )
        for field_name in (
            "slope_lookback",
            "atr_window",
            "volatility_window",
            "volume_window",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        """Return stable metadata suitable for backtest lineage."""

        return tuple(
            sorted(
                (
                    ("atr_window", str(self.atr_window)),
                    ("ema_windows", ",".join(map(str, self.ema_windows))),
                    ("return_windows", ",".join(map(str, self.return_windows))),
                    ("slope_lookback", str(self.slope_lookback)),
                    ("sma_windows", ",".join(map(str, self.sma_windows))),
                    (
                        "structure_windows",
                        ",".join(map(str, self.structure_windows)),
                    ),
                    ("volatility_window", str(self.volatility_window)),
                    ("volume_window", str(self.volume_window)),
                )
            )
        )

@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One named numerical feature; ``None`` means unavailable warm-up data."""

    name: str
    value: float | None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("feature name must not be empty")
        if self.value is not None and not isfinite(self.value):
            raise ValueError("feature values must be finite or None")


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Read-only feature values known for exactly one market bar."""

    symbol: str
    timestamp: datetime
    timeframe: str
    values: tuple[FeatureValue, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION
    engine_version: str = FEATURE_ENGINE_VERSION

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must not be empty")
        _require_aware(self.timestamp, "timestamp")
        names = [item.name for item in self.values]
        if names != sorted(names):
            raise ValueError("feature values must be sorted by stable name")
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        if not self.schema_version or not self.engine_version:
            raise ValueError("feature versions must not be empty")

    def get(self, name: str) -> float | None:
        """Return a feature value, raising for an unknown stable name."""

        for item in self.values:
            if item.name == name:
                return item.value
        raise KeyError(name)

    def is_available(self, *names: str) -> bool:
        return all(self.get(name) is not None for name in names)

    def to_dict(self) -> dict[str, float | None]:
        return {item.name: item.value for item in self.values}


@dataclass(frozen=True, slots=True)
class ReturnObservation:
    """One realized close-to-close return known at its UTC timestamp."""

    timestamp: datetime
    value: float

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "timestamp")
        if not isfinite(self.value):
            raise ValueError("return value must be finite")


@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """Immutable point-in-time return history used by risk controls."""

    symbol: str
    timeframe: str
    observations: tuple[ReturnObservation, ...]

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must not be empty")
        timestamps = [item.timestamp for item in self.observations]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("return observations must be unique and sorted")


@dataclass(frozen=True, slots=True)
class RelativeStrengthValue:
    """One asset's exact-timestamp cross-sectional momentum observation."""

    symbol: str
    rolling_return: float
    rank: float
    percentile: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        for field_name in ("rolling_return", "rank", "percentile"):
            if not isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if not 0.0 <= self.percentile <= 1.0:
            raise ValueError("percentile must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RelativeStrengthSnapshot:
    """Cross-sectional ranking using only bars observed at the same timestamp."""

    timestamp: datetime
    timeframe: str
    lookback: int
    values: tuple[RelativeStrengthValue, ...]
    missing_symbols: tuple[str, ...] = ()
    schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "timestamp")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must not be empty")
        if self.lookback < 1:
            raise ValueError("lookback must be positive")
        symbols = [item.symbol for item in self.values]
        if len(symbols) != len(set(symbols)):
            raise ValueError("relative-strength symbols must be unique")
        if tuple(sorted(set(self.missing_symbols))) != self.missing_symbols:
            raise ValueError("missing_symbols must be sorted and unique")
        if set(symbols).intersection(self.missing_symbols):
            raise ValueError("an asset cannot be both ranked and missing")

    @property
    def ranked_symbols(self) -> tuple[str, ...]:
        """Best-to-worst symbols with a deterministic alphabetical tie-break."""

        return tuple(
            item.symbol
            for item in sorted(
                self.values,
                key=lambda item: (-item.rolling_return, item.symbol),
            )
        )
