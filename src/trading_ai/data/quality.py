"""Fail-closed normalization and explicit OHLCV quality reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable

from trading_ai.core.models import MarketBar
from trading_ai.data.calendar import MarketCalendarService
from trading_ai.data.exceptions import DataValidationError
from trading_ai.data.models import (
    DataQualityReport,
    InstrumentMetadata,
    ProviderBar,
    QualityStatus,
)


def _to_decimal(value: object, field_name: str) -> Decimal:
    if value is None:
        raise ValueError(f"{field_name} is missing")
    try:
        converted = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not numeric") from exc
    if not converted.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return converted


def normalize_provider_bars(
    rows: Iterable[ProviderBar],
    metadata: InstrumentMetadata,
    requested_start: datetime,
    requested_end: datetime,
    *,
    calendar_service: MarketCalendarService | None = None,
    provider_warnings: tuple[str, ...] = (),
    expected_timeframe: str | None = None,
) -> tuple[tuple[MarketBar, ...], DataQualityReport]:
    """Normalize to UTC, sort deterministically, and reject invalid datasets."""

    if (
        requested_start.tzinfo is None
        or requested_start.utcoffset() is None
        or requested_end.tzinfo is None
        or requested_end.utcoffset() is None
    ):
        raise DataValidationError("requested bounds must be timezone-aware")
    if requested_start >= requested_end:
        raise DataValidationError("requested start must precede end")
    calendar = calendar_service or MarketCalendarService()
    raw_rows = tuple(rows)
    invalid_count = 0
    timezone_valid = True
    valid: list[MarketBar] = []
    warnings = list(provider_warnings)
    comparable_timestamps: list[datetime] = []
    for row in raw_rows:
        try:
            if row.symbol != metadata.symbol:
                raise ValueError(
                    f"row symbol {row.symbol!r} does not match metadata symbol "
                    f"{metadata.symbol!r}"
                )
            if expected_timeframe is not None and row.timeframe != expected_timeframe:
                raise ValueError(
                    f"row timeframe {row.timeframe!r} does not match request "
                    f"{expected_timeframe!r}"
                )
            if row.timestamp.tzinfo is None or row.timestamp.utcoffset() is None:
                timezone_valid = False
                raise ValueError("timestamp must be timezone-aware")
            timestamp = row.timestamp.astimezone(timezone.utc)
            if not requested_start <= timestamp < requested_end:
                raise ValueError("timestamp falls outside the requested interval")
            comparable_timestamps.append(timestamp)
            adjusted_close = (
                None
                if row.adjusted_close is None
                else _to_decimal(row.adjusted_close, "adjusted_close")
            )
            valid.append(
                MarketBar(
                    symbol=row.symbol,
                    timeframe=row.timeframe,
                    timestamp=timestamp,
                    open=_to_decimal(row.open, "open"),
                    high=_to_decimal(row.high, "high"),
                    low=_to_decimal(row.low, "low"),
                    close=_to_decimal(row.close, "close"),
                    volume=_to_decimal(row.volume, "volume"),
                    adjusted_close=adjusted_close,
                    source=row.source,
                )
            )
        except (TypeError, ValueError):
            invalid_count += 1
    input_sorted = comparable_timestamps == sorted(comparable_timestamps)
    if not input_sorted:
        warnings.append("provider rows were reordered deterministically")
    valid.sort(key=lambda bar: (bar.symbol, bar.timeframe, bar.timestamp))
    keys = [(bar.symbol, bar.timeframe, bar.timestamp) for bar in valid]
    duplicate_count = len(keys) - len(set(keys))
    unique_bars: list[MarketBar] = []
    seen: set[tuple[str, str, datetime]] = set()
    for bar in valid:
        key = (bar.symbol, bar.timeframe, bar.timestamp)
        if key not in seen:
            unique_bars.append(bar)
            seen.add(key)
    symbol = unique_bars[0].symbol if unique_bars else metadata.symbol
    timeframe = unique_bars[0].timeframe if unique_bars else (
        raw_rows[0].timeframe if raw_rows else "unknown"
    )
    if unique_bars and any(bar.timeframe != timeframe for bar in unique_bars):
        invalid_count += sum(
            bar.timeframe != timeframe for bar in unique_bars
        )
        warnings.append("provider rows contain inconsistent timeframes")
    missing_count = 0
    gap_count = 0
    unexpected_count = 0
    if unique_bars and timeframe in {"1h", "4h", "1d"}:
        missing_count, gap_count, unexpected_count = calendar.gap_counts(
            (bar.timestamp for bar in unique_bars),
            metadata,
            requested_start,
            requested_end,
            timeframe,
        )
        if missing_count:
            warnings.append(f"{missing_count} expected market bars are missing")
        if unexpected_count:
            warnings.append(f"{unexpected_count} bars fall outside expected sessions")
    if invalid_count or duplicate_count or not timezone_valid or not unique_bars:
        status = QualityStatus.FAIL
    elif warnings or missing_count or unexpected_count:
        status = QualityStatus.WARNING
    else:
        status = QualityStatus.PASS
    report = DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        row_count=len(unique_bars),
        duplicate_count=duplicate_count,
        invalid_bar_count=invalid_count,
        missing_expected_bar_count=missing_count,
        unexpected_gap_count=gap_count,
        first_timestamp=unique_bars[0].timestamp if unique_bars else None,
        last_timestamp=unique_bars[-1].timestamp if unique_bars else None,
        timezone_valid=timezone_valid,
        sorted=True,
        quality_status=status,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    if status is QualityStatus.FAIL:
        raise DataValidationError(
            "dataset failed normalization: invalid, duplicate, naive, or empty bars",
            report,
        )
    return tuple(unique_bars), report


def assess_normalized_bars(
    bars: Iterable[MarketBar],
    metadata: InstrumentMetadata,
    requested_start: datetime,
    requested_end: datetime,
    *,
    calendar_service: MarketCalendarService | None = None,
    warnings: tuple[str, ...] = (),
) -> DataQualityReport:
    provider_rows = tuple(
        ProviderBar(
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            adjusted_close=bar.adjusted_close,
            source=bar.source,
        )
        for bar in bars
    )
    _, report = normalize_provider_bars(
        provider_rows,
        metadata,
        requested_start,
        requested_end,
        calendar_service=calendar_service,
        provider_warnings=warnings,
        expected_timeframe=(provider_rows[0].timeframe if provider_rows else None),
    )
    return report
