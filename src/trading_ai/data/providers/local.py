"""Offline adapter for explicitly supplied historical CSV or Parquet files."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Mapping

from trading_ai.data.base import DataProvider
from trading_ai.data.exceptions import (
    DataProviderError,
    DataUnavailableError,
    DataValidationError,
)
from trading_ai.data.models import (
    CorporateAction,
    InstrumentMetadata,
    MarketDataRequest,
    ProviderBar,
    ProviderBars,
)


_REQUIRED_COLUMNS = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)


class LocalHistoricalFileProvider(DataProvider):
    """Read caller-mapped historical files without network access.

    Paths are relative to ``root`` and each dataset must be explicitly mapped by
    ``(symbol, timeframe)``.  Timestamps must already carry a timezone; naive
    values are rejected rather than silently interpreted as UTC.  The regular
    Data Engine still owns normalization, quality checks, manifests and storage.
    """

    def __init__(
        self,
        *,
        root: Path | str,
        datasets: Mapping[tuple[str, str], Path | str],
        metadata_by_symbol: Mapping[str, InstrumentMetadata],
        actions_by_symbol: Mapping[str, tuple[CorporateAction, ...]] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.datasets = dict(datasets)
        self.metadata_by_symbol = dict(metadata_by_symbol)
        self.actions_by_symbol = dict(actions_by_symbol or {})
        if not self.datasets:
            raise ValueError("at least one local historical dataset is required")
        for key, configured_path in self.datasets.items():
            if len(key) != 2 or not all(str(value).strip() for value in key):
                raise ValueError("dataset keys must be non-empty (symbol, timeframe) pairs")
            self._resolve_path(configured_path)

    @property
    def name(self) -> str:
        return "local_historical_file"

    @property
    def version(self) -> str:
        return "1.0"

    def _resolve_path(self, configured_path: Path | str) -> Path:
        relative = Path(configured_path)
        if relative.is_absolute():
            raise DataValidationError("local dataset paths must be relative to the configured root")
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise DataValidationError("local dataset path escapes the configured root")
        if candidate.suffix.lower() not in {".csv", ".parquet"}:
            raise DataValidationError("local datasets must be CSV or Parquet files")
        return candidate

    def metadata(self, symbol: str) -> InstrumentMetadata:
        try:
            metadata = self.metadata_by_symbol[symbol]
        except KeyError as exc:
            raise DataUnavailableError(
                f"local historical metadata unavailable for {symbol}"
            ) from exc
        if metadata.symbol != symbol:
            raise DataValidationError("local metadata symbol does not match request")
        return metadata

    @staticmethod
    def _decimal(value, field_name: str) -> Decimal | None:
        import pandas as pd

        if pd.isna(value):
            return None
        try:
            return Decimal(str(value))
        except (ValueError, ArithmeticError) as exc:
            raise DataValidationError(
                f"invalid numeric value in local column {field_name!r}"
            ) from exc

    @staticmethod
    def _aware_timestamp(value) -> datetime:
        import pandas as pd

        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError("invalid local dataset timestamp") from exc
        converted = timestamp.to_pydatetime()
        if converted.tzinfo is None or converted.utcoffset() is None:
            raise DataValidationError(
                "local dataset timestamps must be timezone-aware"
            )
        return converted.astimezone(timezone.utc)

    def fetch_bars(self, request: MarketDataRequest) -> ProviderBars:
        key = (request.symbol, request.timeframe)
        try:
            configured_path = self.datasets[key]
        except KeyError as exc:
            raise DataUnavailableError(
                f"local bars unavailable for {request.symbol} {request.timeframe}"
            ) from exc
        path = self._resolve_path(configured_path)
        if not path.is_file():
            raise DataUnavailableError(
                f"configured local dataset is unavailable for {request.symbol} {request.timeframe}"
            )
        try:
            import pandas as pd

            frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
        except Exception as exc:
            raise DataProviderError(
                f"could not read local dataset for {request.symbol} {request.timeframe}"
            ) from exc
        columns = {str(column).strip().lower(): column for column in frame.columns}
        missing = sorted(_REQUIRED_COLUMNS - columns.keys())
        if missing:
            raise DataValidationError(
                "local dataset is missing required columns: " + ", ".join(missing)
            )
        bars: list[ProviderBar] = []
        for _, row in frame.iterrows():
            timestamp = self._aware_timestamp(row[columns["timestamp"]])
            if not request.start <= timestamp < request.end:
                continue
            adjusted_column = columns.get("adjusted_close") or columns.get("adj close")
            bars.append(
                ProviderBar(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    timestamp=timestamp,
                    open=self._decimal(row[columns["open"]], "open"),
                    high=self._decimal(row[columns["high"]], "high"),
                    low=self._decimal(row[columns["low"]], "low"),
                    close=self._decimal(row[columns["close"]], "close"),
                    volume=self._decimal(row[columns["volume"]], "volume"),
                    adjusted_close=(
                        self._decimal(row[adjusted_column], "adjusted_close")
                        if adjusted_column is not None
                        else None
                    ),
                    source=self.name,
                )
            )
        if not bars:
            raise DataUnavailableError(
                f"local dataset has no rows in the requested range for {request.symbol}"
            )
        return ProviderBars(
            bars=tuple(bars),
            metadata=self.metadata(request.symbol),
            warnings=(
                "Imported local historical data; provenance depends on the explicitly supplied source file.",
            ),
        )

    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[CorporateAction, ...]:
        return tuple(
            action
            for action in self.actions_by_symbol.get(symbol, ())
            if start <= action.timestamp < end
        )
