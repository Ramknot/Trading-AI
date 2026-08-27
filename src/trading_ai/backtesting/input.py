"""Offline adapters from validated memory or Lot 1 Parquet datasets."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from trading_ai.backtesting.exceptions import BacktestDataError
from trading_ai.backtesting.models import (
    BacktestDataset,
    DatasetReference,
)
from trading_ai.backtesting.reproducibility import stable_hash
from trading_ai.core.models import MarketBar
from trading_ai.data.models import (
    CorporateAction,
    DataKind,
    DataQualityReport,
    DatasetManifest,
    InstrumentMetadata,
)
from trading_ai.data.quality import assess_normalized_bars
from trading_ai.data.storage import ParquetDataStore


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise BacktestDataError(f"{field_name} must be timezone-aware")


def memory_dataset(
    *,
    bars: Iterable[MarketBar],
    corporate_actions: Iterable[CorporateAction],
    quality_report: DataQualityReport,
    requested_start: datetime | None = None,
    requested_end: datetime | None = None,
    provider: str = "memory",
) -> BacktestDataset:
    normalized_bars = tuple(bars)
    normalized_actions = tuple(
        sorted(
            corporate_actions,
            key=lambda action: (
                action.timestamp,
                action.symbol,
                action.action_type.value,
                action.value,
            ),
        )
    )
    if not normalized_bars:
        raise BacktestDataError("memory dataset is empty")
    first = normalized_bars[0]
    start = requested_start or first.timestamp
    end = requested_end or (normalized_bars[-1].timestamp + timedelta(microseconds=1))
    _require_aware(start, "requested_start")
    _require_aware(end, "requested_end")
    checksum = stable_hash(
        {"bars": normalized_bars, "corporate_actions": normalized_actions}
    )
    reference = DatasetReference(
        dataset_id=f"memory-{checksum[:24]}",
        provider=provider,
        symbol=first.symbol,
        timeframe=first.timeframe,
        checksum_sha256=checksum,
        data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE.value,
        requested_start=start.astimezone(timezone.utc),
        requested_end=end.astimezone(timezone.utc),
        actual_start=normalized_bars[0].timestamp,
        actual_end=normalized_bars[-1].timestamp,
    )
    return BacktestDataset(
        bars=normalized_bars,
        corporate_actions=normalized_actions,
        quality_report=quality_report,
        reference=reference,
    )


def _from_manifests(
    store: ParquetDataStore,
    manifest: DatasetManifest,
    actions_manifest: DatasetManifest | None,
) -> BacktestDataset:
    bars = store.read_bars(manifest)
    actions = (
        tuple(
            sorted(
                store.read_corporate_actions(actions_manifest),
                key=lambda action: (
                    action.timestamp,
                    action.symbol,
                    action.action_type.value,
                    action.value,
                ),
            )
        )
        if actions_manifest is not None
        else ()
    )
    metadata = InstrumentMetadata(
        symbol=manifest.symbol,
        exchange=manifest.exchange,
        exchange_timezone=manifest.source_timezone,
        calendar=manifest.calendar,
        source=manifest.provider,
    )
    quality = assess_normalized_bars(
        bars,
        metadata,
        manifest.requested_start,
        manifest.requested_end,
        warnings=manifest.warnings,
    )
    return BacktestDataset(
        bars=bars,
        corporate_actions=actions,
        quality_report=quality,
        reference=DatasetReference(
            dataset_id=manifest.dataset_id,
            provider=manifest.provider,
            symbol=manifest.symbol,
            timeframe=manifest.timeframe,
            checksum_sha256=manifest.checksum_sha256,
            data_kind=manifest.data_kind.value,
            requested_start=manifest.requested_start,
            requested_end=manifest.requested_end,
            actual_start=manifest.actual_start,
            actual_end=manifest.actual_end,
            manifest_file_path=(
                Path("manifests") / f"{manifest.dataset_id}.json"
            ).as_posix(),
            derived_from=manifest.derived_from,
            corporate_actions_dataset_id=(
                actions_manifest.dataset_id if actions_manifest is not None else None
            ),
            corporate_actions_checksum_sha256=(
                actions_manifest.checksum_sha256
                if actions_manifest is not None
                else None
            ),
        ),
    )


def load_cached_dataset(
    store: ParquetDataStore,
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> BacktestDataset:
    _require_aware(start, "start")
    _require_aware(end, "end")
    candidates = [
        manifest
        for manifest in store.manifests()
        if manifest.symbol == symbol
        and manifest.timeframe == timeframe
        and manifest.data_kind is not DataKind.CORPORATE_ACTIONS
        and manifest.requested_start == start.astimezone(timezone.utc)
        and manifest.requested_end == end.astimezone(timezone.utc)
    ]
    if not candidates:
        raise BacktestDataError(
            f"no exact cached dataset for {symbol} {timeframe}; backtests never download"
        )
    manifest = max(candidates, key=lambda item: item.downloaded_at)
    action_candidates = [
        item
        for item in store.manifests()
        if item.symbol == symbol
        and item.data_kind is DataKind.CORPORATE_ACTIONS
        and item.requested_start == manifest.requested_start
        and item.requested_end == manifest.requested_end
    ]
    actions_manifest = max(
        action_candidates, key=lambda item: item.downloaded_at, default=None
    )
    return _from_manifests(store, manifest, actions_manifest)
