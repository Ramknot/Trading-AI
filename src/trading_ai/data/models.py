"""Provider-neutral immutable models for historical market data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from trading_ai.core.models import MarketBar


class CacheMode(str, Enum):
    CACHE_ONLY = "CACHE_ONLY"
    CACHE_FIRST = "CACHE_FIRST"
    REFRESH = "REFRESH"


class DataKind(str, Enum):
    RAW_WITH_ADJUSTED_CLOSE = "RAW_WITH_ADJUSTED_CLOSE"
    DERIVED_RAW_WITH_ADJUSTED_CLOSE = "DERIVED_RAW_WITH_ADJUSTED_CLOSE"
    CORPORATE_ACTIONS = "CORPORATE_ACTIONS"


class CorporateActionType(str, Enum):
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"


class QualityStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


def _require_non_empty(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class MarketDataRequest:
    """One explicit provider request; end is exclusive."""

    symbol: str
    timeframe: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.symbol, "symbol")
        _require_non_empty(self.timeframe, "timeframe")
        _require_aware(self.start, "start")
        _require_aware(self.end, "end")
        if self.start >= self.end:
            raise ValueError("start must precede end")


NumericValue = Decimal | int | float | None


@dataclass(frozen=True, slots=True)
class ProviderBar:
    """Untrusted provider row awaiting normalization and validation."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: NumericValue
    high: NumericValue
    low: NumericValue
    close: NumericValue
    volume: NumericValue
    adjusted_close: NumericValue = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class InstrumentMetadata:
    """Exchange metadata needed for timezone and market-session awareness."""

    symbol: str
    exchange: str
    exchange_timezone: str
    calendar: str
    source: str
    currency: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "symbol",
            "exchange",
            "exchange_timezone",
            "calendar",
            "source",
        ):
            _require_non_empty(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class ProviderBars:
    """Provider-neutral response containing raw rows plus instrument metadata."""

    bars: tuple[ProviderBar, ...]
    metadata: InstrumentMetadata
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Dividend:
    symbol: str
    timestamp: datetime
    value: Decimal
    source: str
    action_type: CorporateActionType = CorporateActionType.DIVIDEND

    def __post_init__(self) -> None:
        _require_non_empty(self.symbol, "symbol")
        _require_non_empty(self.source, "source")
        _require_aware(self.timestamp, "timestamp")
        if self.value <= Decimal("0"):
            raise ValueError("dividend value must be positive")


@dataclass(frozen=True, slots=True)
class StockSplit:
    symbol: str
    timestamp: datetime
    value: Decimal
    source: str
    action_type: CorporateActionType = CorporateActionType.SPLIT

    def __post_init__(self) -> None:
        _require_non_empty(self.symbol, "symbol")
        _require_non_empty(self.source, "source")
        _require_aware(self.timestamp, "timestamp")
        if self.value <= Decimal("0"):
            raise ValueError("split ratio must be positive")


CorporateAction = Dividend | StockSplit


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    symbol: str
    timeframe: str
    row_count: int
    duplicate_count: int
    invalid_bar_count: int
    missing_expected_bar_count: int
    unexpected_gap_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    timezone_valid: bool
    sorted: bool
    quality_status: QualityStatus
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.symbol, "symbol")
        _require_non_empty(self.timeframe, "timeframe")
        for field_name in (
            "row_count",
            "duplicate_count",
            "invalid_bar_count",
            "missing_expected_bar_count",
            "unexpected_gap_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.first_timestamp is not None:
            _require_aware(self.first_timestamp, "first_timestamp")
        if self.last_timestamp is not None:
            _require_aware(self.last_timestamp, "last_timestamp")
        if (
            self.first_timestamp is not None
            and self.last_timestamp is not None
            and self.first_timestamp > self.last_timestamp
        ):
            raise ValueError("first_timestamp must not follow last_timestamp")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_status"] = self.quality_status.value
        for key in ("first_timestamp", "last_timestamp"):
            value = payload[key]
            payload[key] = value.isoformat() if value is not None else None
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    provider: str
    provider_version: str | None
    symbol: str
    timeframe: str
    requested_start: datetime
    requested_end: datetime
    actual_start: datetime | None
    actual_end: datetime | None
    downloaded_at: datetime
    row_count: int
    timezone: str
    source_timezone: str
    exchange: str
    calendar: str
    data_kind: DataKind
    schema_version: str
    file_path: str
    checksum_sha256: str
    derived_from: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "provider",
            "symbol",
            "timeframe",
            "timezone",
            "source_timezone",
            "exchange",
            "calendar",
            "schema_version",
            "file_path",
            "checksum_sha256",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        for field_name in ("requested_start", "requested_end", "downloaded_at"):
            _require_aware(getattr(self, field_name), field_name)
        if self.actual_start is not None:
            _require_aware(self.actual_start, "actual_start")
        if self.actual_end is not None:
            _require_aware(self.actual_end, "actual_end")
        if self.requested_start >= self.requested_end:
            raise ValueError("requested_start must precede requested_end")
        if self.row_count < 0:
            raise ValueError("row_count must not be negative")
        if (
            self.actual_start is not None
            and self.actual_end is not None
            and self.actual_start > self.actual_end
        ):
            raise ValueError("actual_start must not follow actual_end")
        if len(self.checksum_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.checksum_sha256.lower()
        ):
            raise ValueError("checksum_sha256 must be a 64-character hexadecimal hash")
        if (
            self.data_kind is DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE
            and not self.derived_from
        ):
            raise ValueError("derived datasets must record source lineage")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "requested_start",
            "requested_end",
            "actual_start",
            "actual_end",
            "downloaded_at",
        ):
            value = payload[key]
            payload[key] = value.isoformat() if value is not None else None
        payload["data_kind"] = self.data_kind.value
        payload["derived_from"] = list(self.derived_from)
        payload["warnings"] = list(self.warnings)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        values = dict(payload)
        for key in (
            "requested_start",
            "requested_end",
            "actual_start",
            "actual_end",
            "downloaded_at",
        ):
            if values[key] is not None:
                values[key] = datetime.fromisoformat(values[key])
        values["data_kind"] = DataKind(values["data_kind"])
        values["derived_from"] = tuple(values.get("derived_from", ()))
        values["warnings"] = tuple(values.get("warnings", ()))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class DataFetchResult:
    bars: tuple[MarketBar, ...]
    corporate_actions: tuple[CorporateAction, ...]
    manifest: DatasetManifest
    corporate_actions_manifest: DatasetManifest | None
    quality_report: DataQualityReport
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    manifest: DatasetManifest
    quality_report: DataQualityReport
    integrity_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "quality_report": self.quality_report.to_dict(),
            "integrity_valid": self.integrity_valid,
        }
