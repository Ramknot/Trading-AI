"""Local Parquet datasets, JSON manifests, cache lookup, and SHA-256 integrity."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from trading_ai.core.models import MarketBar
from trading_ai.data.exceptions import DataIntegrityError, DataStorageError
from trading_ai.data.models import (
    CorporateAction,
    CorporateActionType,
    DataKind,
    DatasetManifest,
    Dividend,
    InstrumentMetadata,
    MarketDataRequest,
    StockSplit,
)


SCHEMA_VERSION = "1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_id(
    *,
    provider: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    data_kind: DataKind,
    derived_from: tuple[str, ...] = (),
) -> str:
    identity = json.dumps(
        {
            "provider": provider,
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start.astimezone(timezone.utc).isoformat(),
            "end": end.astimezone(timezone.utc).isoformat(),
            "data_kind": data_kind.value,
            "derived_from": list(derived_from),
            "schema_version": SCHEMA_VERSION,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:24]


class ParquetDataStore:
    """Repository-local store; the default root is ignored by Git."""

    def __init__(self, root: Path | str = Path("data_local")) -> None:
        self.root = Path(root)

    @property
    def manifest_directory(self) -> Path:
        return self.root / "manifests"

    def _manifest_path(self, dataset_id: str) -> Path:
        return self.manifest_directory / f"{dataset_id}.json"

    def _safe_data_path(self, relative_path: str) -> Path:
        root = self.root.resolve()
        path = (self.root / relative_path).resolve()
        if root != path and root not in path.parents:
            raise DataStorageError("manifest file_path escapes the data store")
        return path

    @staticmethod
    def _bar_schema():
        import pyarrow as pa

        decimal_type = pa.decimal128(38, 18)
        return pa.schema(
            [
                ("symbol", pa.string()),
                ("timeframe", pa.string()),
                ("timestamp", pa.timestamp("us", tz="UTC")),
                ("open", decimal_type),
                ("high", decimal_type),
                ("low", decimal_type),
                ("close", decimal_type),
                ("volume", decimal_type),
                ("adjusted_close", decimal_type),
                ("source", pa.string()),
            ]
        )

    @staticmethod
    def _action_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("symbol", pa.string()),
                ("timestamp", pa.timestamp("us", tz="UTC")),
                ("action_type", pa.string()),
                ("value", pa.decimal128(38, 18)),
                ("source", pa.string()),
            ]
        )

    @staticmethod
    def _write_parquet_atomic(table, path: Path) -> None:
        import pyarrow.parquet as parquet

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        parquet.write_table(table, temporary, compression="zstd")
        temporary.replace(path)

    def _write_manifest(self, manifest: DatasetManifest) -> None:
        path = self._manifest_path(manifest.dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _validate_bars_for_storage(
        bars: tuple[MarketBar, ...], request: MarketDataRequest
    ) -> None:
        if not bars:
            raise DataStorageError("empty market-bar datasets are not persisted")
        if any(
            bar.symbol != request.symbol or bar.timeframe != request.timeframe
            for bar in bars
        ):
            raise DataStorageError("bar identity does not match the storage request")
        if any(bar.timestamp.utcoffset() != timedelta(0) for bar in bars):
            raise DataStorageError("persisted market bars must use UTC timestamps")
        if any(not request.start <= bar.timestamp < request.end for bar in bars):
            raise DataStorageError("bar timestamp falls outside the storage request")
        expected = tuple(
            sorted(bars, key=lambda bar: (bar.symbol, bar.timeframe, bar.timestamp))
        )
        if bars != expected:
            raise DataStorageError("bars must be normalized and sorted before storage")
        keys = [(bar.symbol, bar.timeframe, bar.timestamp) for bar in bars]
        if len(keys) != len(set(keys)):
            raise DataStorageError("duplicate market bars cannot be persisted")

    def save_bars(
        self,
        *,
        bars: Iterable[MarketBar],
        request: MarketDataRequest,
        metadata: InstrumentMetadata,
        provider: str,
        provider_version: str | None,
        data_kind: DataKind,
        derived_from: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
    ) -> DatasetManifest:
        import pyarrow as pa

        normalized = tuple(bars)
        self._validate_bars_for_storage(normalized, request)
        dataset_id = _dataset_id(
            provider=provider,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start=request.start,
            end=request.end,
            data_kind=data_kind,
            derived_from=derived_from,
        )
        if data_kind is DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE:
            relative = Path("derived") / request.symbol / request.timeframe
        else:
            relative = Path("market") / provider / request.symbol / request.timeframe
        relative_path = relative / f"{dataset_id}.parquet"
        path = self.root / relative_path
        records = [
            {
                "symbol": bar.symbol,
                "timeframe": bar.timeframe,
                "timestamp": bar.timestamp.astimezone(timezone.utc),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "adjusted_close": bar.adjusted_close,
                "source": bar.source,
            }
            for bar in normalized
        ]
        try:
            table = pa.Table.from_pylist(records, schema=self._bar_schema())
            self._write_parquet_atomic(table, path)
            checksum = sha256_file(path)
            manifest = DatasetManifest(
                dataset_id=dataset_id,
                provider=provider,
                provider_version=provider_version,
                symbol=request.symbol,
                timeframe=request.timeframe,
                requested_start=request.start.astimezone(timezone.utc),
                requested_end=request.end.astimezone(timezone.utc),
                actual_start=normalized[0].timestamp if normalized else None,
                actual_end=normalized[-1].timestamp if normalized else None,
                downloaded_at=datetime.now(timezone.utc),
                row_count=len(normalized),
                timezone="UTC",
                source_timezone=metadata.exchange_timezone,
                exchange=metadata.exchange,
                calendar=metadata.calendar,
                data_kind=data_kind,
                schema_version=SCHEMA_VERSION,
                file_path=relative_path.as_posix(),
                checksum_sha256=checksum,
                derived_from=derived_from,
                warnings=warnings,
            )
            self._write_manifest(manifest)
            return manifest
        except (DataStorageError, DataIntegrityError):
            raise
        except Exception as exc:
            raise DataStorageError(
                f"unable to save Parquet bars for {request.symbol} {request.timeframe}"
            ) from exc

    def save_corporate_actions(
        self,
        *,
        actions: Iterable[CorporateAction],
        symbol: str,
        start: datetime,
        end: datetime,
        metadata: InstrumentMetadata,
        provider: str,
        provider_version: str | None,
        warnings: tuple[str, ...] = (),
    ) -> DatasetManifest:
        import pyarrow as pa

        normalized = tuple(sorted(actions, key=lambda action: action.timestamp))
        if any(action.symbol != symbol for action in normalized):
            raise DataStorageError("corporate-action symbol does not match request")
        if any(action.timestamp.utcoffset() != timedelta(0) for action in normalized):
            raise DataStorageError("persisted corporate actions must use UTC timestamps")
        if any(not start <= action.timestamp < end for action in normalized):
            raise DataStorageError(
                "corporate-action timestamp falls outside the requested interval"
            )
        action_keys = [
            (
                action.symbol,
                action.timestamp,
                action.action_type,
                action.value,
            )
            for action in normalized
        ]
        if len(action_keys) != len(set(action_keys)):
            raise DataStorageError("duplicate corporate actions cannot be persisted")
        dataset_id = _dataset_id(
            provider=provider,
            symbol=symbol,
            timeframe="actions",
            start=start,
            end=end,
            data_kind=DataKind.CORPORATE_ACTIONS,
        )
        relative_path = (
            Path("corporate_actions") / provider / symbol / f"{dataset_id}.parquet"
        )
        path = self.root / relative_path
        records = [
            {
                "symbol": action.symbol,
                "timestamp": action.timestamp.astimezone(timezone.utc),
                "action_type": action.action_type.value,
                "value": action.value,
                "source": action.source,
            }
            for action in normalized
        ]
        try:
            table = pa.Table.from_pylist(records, schema=self._action_schema())
            self._write_parquet_atomic(table, path)
            manifest = DatasetManifest(
                dataset_id=dataset_id,
                provider=provider,
                provider_version=provider_version,
                symbol=symbol,
                timeframe="actions",
                requested_start=start.astimezone(timezone.utc),
                requested_end=end.astimezone(timezone.utc),
                actual_start=normalized[0].timestamp if normalized else None,
                actual_end=normalized[-1].timestamp if normalized else None,
                downloaded_at=datetime.now(timezone.utc),
                row_count=len(normalized),
                timezone="UTC",
                source_timezone=metadata.exchange_timezone,
                exchange=metadata.exchange,
                calendar=metadata.calendar,
                data_kind=DataKind.CORPORATE_ACTIONS,
                schema_version=SCHEMA_VERSION,
                file_path=relative_path.as_posix(),
                checksum_sha256=sha256_file(path),
                warnings=warnings,
            )
            self._write_manifest(manifest)
            return manifest
        except Exception as exc:
            raise DataStorageError(
                f"unable to save corporate actions for {symbol}"
            ) from exc

    def read_manifest(self, dataset_id: str) -> DatasetManifest:
        try:
            payload = json.loads(
                self._manifest_path(dataset_id).read_text(encoding="utf-8")
            )
            return DatasetManifest.from_dict(payload)
        except FileNotFoundError as exc:
            raise DataStorageError(f"manifest not found: {dataset_id}") from exc
        except Exception as exc:
            raise DataStorageError(f"invalid manifest: {dataset_id}") from exc

    def verify_integrity(self, manifest: DatasetManifest) -> bool:
        path = self._safe_data_path(manifest.file_path)
        if not path.is_file():
            raise DataIntegrityError(f"dataset file is missing: {path}")
        actual = sha256_file(path)
        if actual != manifest.checksum_sha256:
            raise DataIntegrityError(
                f"SHA-256 mismatch for dataset {manifest.dataset_id}"
            )
        return True

    def read_bars(self, manifest: DatasetManifest) -> tuple[MarketBar, ...]:
        import pyarrow.parquet as parquet

        self.verify_integrity(manifest)
        try:
            records = parquet.read_table(
                self._safe_data_path(manifest.file_path)
            ).to_pylist()
            bars = tuple(
                MarketBar(
                    symbol=record["symbol"],
                    timeframe=record["timeframe"],
                    timestamp=record["timestamp"].astimezone(timezone.utc),
                    open=Decimal(record["open"]),
                    high=Decimal(record["high"]),
                    low=Decimal(record["low"]),
                    close=Decimal(record["close"]),
                    volume=Decimal(record["volume"]),
                    adjusted_close=(
                        Decimal(record["adjusted_close"])
                        if record["adjusted_close"] is not None
                        else None
                    ),
                    source=record["source"],
                )
                for record in records
            )
            expected = tuple(
                sorted(
                    bars,
                    key=lambda bar: (bar.symbol, bar.timeframe, bar.timestamp),
                )
            )
            keys = [(bar.symbol, bar.timeframe, bar.timestamp) for bar in bars]
            if (
                len(bars) != manifest.row_count
                or bars != expected
                or len(keys) != len(set(keys))
                or any(
                    bar.symbol != manifest.symbol
                    or bar.timeframe != manifest.timeframe
                    for bar in bars
                )
            ):
                raise DataIntegrityError(
                    f"dataset structure does not match manifest {manifest.dataset_id}"
                )
            return bars
        except (DataIntegrityError, DataStorageError):
            raise
        except Exception as exc:
            raise DataStorageError(
                f"unable to read dataset {manifest.dataset_id}"
            ) from exc

    def read_corporate_actions(
        self, manifest: DatasetManifest
    ) -> tuple[CorporateAction, ...]:
        import pyarrow.parquet as parquet

        self.verify_integrity(manifest)
        try:
            records = parquet.read_table(
                self._safe_data_path(manifest.file_path)
            ).to_pylist()
            actions: list[CorporateAction] = []
            for record in records:
                if record["action_type"] == CorporateActionType.DIVIDEND.value:
                    action_class = Dividend
                elif record["action_type"] == CorporateActionType.SPLIT.value:
                    action_class = StockSplit
                else:
                    raise DataIntegrityError(
                        f"unknown corporate action type {record['action_type']!r}"
                    )
                actions.append(
                    action_class(
                        symbol=record["symbol"],
                        timestamp=record["timestamp"].astimezone(timezone.utc),
                        value=Decimal(record["value"]),
                        source=record["source"],
                    )
                )
            expected_actions = sorted(actions, key=lambda action: action.timestamp)
            if (
                len(actions) != manifest.row_count
                or actions != expected_actions
                or any(
                    action.symbol != manifest.symbol for action in actions
                )
            ):
                raise DataIntegrityError(
                    f"corporate actions do not match manifest {manifest.dataset_id}"
                )
            return tuple(actions)
        except (DataIntegrityError, DataStorageError):
            raise
        except Exception as exc:
            raise DataStorageError(
                f"unable to read corporate actions {manifest.dataset_id}"
            ) from exc

    def manifests(self) -> tuple[DatasetManifest, ...]:
        if not self.manifest_directory.exists():
            return ()
        found: list[DatasetManifest] = []
        for path in sorted(self.manifest_directory.glob("*.json")):
            try:
                found.append(
                    DatasetManifest.from_dict(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
                )
            except Exception as exc:
                raise DataStorageError(f"invalid manifest file: {path}") from exc
        return tuple(found)

    def find_exact(
        self,
        *,
        provider: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        data_kind: DataKind,
    ) -> DatasetManifest | None:
        matches = [
            manifest
            for manifest in self.manifests()
            if manifest.provider == provider
            and manifest.symbol == symbol
            and manifest.timeframe == timeframe
            and manifest.requested_start == start.astimezone(timezone.utc)
            and manifest.requested_end == end.astimezone(timezone.utc)
            and manifest.data_kind is data_kind
        ]
        return max(matches, key=lambda item: item.downloaded_at, default=None)

    def find_latest(self, symbol: str, timeframe: str) -> DatasetManifest | None:
        matches = [
            manifest
            for manifest in self.manifests()
            if manifest.symbol == symbol
            and manifest.timeframe == timeframe
            and manifest.data_kind is not DataKind.CORPORATE_ACTIONS
        ]
        return max(matches, key=lambda item: item.downloaded_at, default=None)
