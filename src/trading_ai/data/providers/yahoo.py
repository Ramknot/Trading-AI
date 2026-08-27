"""Yahoo Finance historical adapter isolated behind DataProvider."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from zoneinfo import ZoneInfo

from trading_ai.data.base import DataProvider
from trading_ai.data.exceptions import (
    DataProviderError,
    DataProviderTemporaryError,
    DataUnavailableError,
)
from trading_ai.data.models import (
    CorporateAction,
    Dividend,
    InstrumentMetadata,
    MarketDataRequest,
    ProviderBar,
    ProviderBars,
    StockSplit,
)


_EXCHANGE_METADATA: dict[str, tuple[str, str]] = {
    "ASE": ("America/New_York", "NYSE"),
    "BTS": ("America/New_York", "NYSE"),
    "NCM": ("America/New_York", "NYSE"),
    "NGM": ("America/New_York", "NYSE"),
    "NMS": ("America/New_York", "NYSE"),
    "NYQ": ("America/New_York", "NYSE"),
    "PCX": ("America/New_York", "NYSE"),
    "PAR": ("Europe/Paris", "XPAR"),
}


def _decimal_or_none(value: Any) -> Decimal | None:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return Decimal(str(value))


def _is_temporary_error(error: Exception) -> bool:
    label = f"{type(error).__name__}: {error}".lower()
    return isinstance(error, (TimeoutError, ConnectionError)) or any(
        token in label
        for token in ("timeout", "temporarily", "connection", "rate limit", "429")
    )


class YahooFinanceProvider(DataProvider):
    """Development-only Yahoo adapter preserving raw and adjusted prices."""

    @property
    def name(self) -> str:
        return "yahoo"

    @property
    def version(self) -> str | None:
        try:
            return version("yfinance")
        except PackageNotFoundError:
            return None

    def metadata(self, symbol: str) -> InstrumentMetadata:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            metadata = ticker.get_history_metadata() or {}
            exchange = str(metadata.get("exchangeName") or "UNKNOWN").upper()
            source_timezone = str(metadata.get("exchangeTimezoneName") or "")
            exchange_mapping = _EXCHANGE_METADATA.get(exchange)
            if exchange_mapping is None:
                raise DataProviderError(
                    f"Yahoo exchange {exchange!r} has no configured market calendar"
                )
            mapped_timezone, calendar = exchange_mapping
            source_timezone = source_timezone or mapped_timezone
            currency = metadata.get("currency")
            return InstrumentMetadata(
                symbol=symbol,
                exchange=exchange,
                exchange_timezone=source_timezone,
                calendar=calendar,
                source=self.name,
                currency=str(currency) if currency else None,
            )
        except DataProviderError:
            raise
        except Exception as exc:  # provider boundary intentionally catches all
            if _is_temporary_error(exc):
                raise DataProviderTemporaryError(
                    f"temporary Yahoo metadata failure for {symbol}"
                ) from exc
            raise DataProviderError(f"Yahoo metadata failure for {symbol}") from exc

    def fetch_bars(self, request: MarketDataRequest) -> ProviderBars:
        if request.timeframe not in {"1h", "1d"}:
            raise DataProviderError(
                f"Yahoo native timeframe {request.timeframe!r} is unsupported"
            )
        try:
            import pandas as pd
            import yfinance as yf

            metadata = self.metadata(request.symbol)
            frame = yf.download(
                request.symbol,
                start=request.start,
                end=request.end,
                interval=request.timeframe,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if frame is None or frame.empty:
                raise DataUnavailableError(
                    f"Yahoo returned no bars for {request.symbol} {request.timeframe}"
                )
            if isinstance(frame.columns, pd.MultiIndex):
                if request.symbol in frame.columns.get_level_values(-1):
                    frame = frame.xs(request.symbol, axis=1, level=-1)
                else:
                    frame.columns = frame.columns.get_level_values(0)
            bars: list[ProviderBar] = []
            source_zone = ZoneInfo(metadata.exchange_timezone)
            for index, row in frame.iterrows():
                timestamp = index.to_pydatetime()
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=source_zone)
                timestamp = timestamp.astimezone(timezone.utc)
                bars.append(
                    ProviderBar(
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        timestamp=timestamp,
                        open=_decimal_or_none(row.get("Open")),
                        high=_decimal_or_none(row.get("High")),
                        low=_decimal_or_none(row.get("Low")),
                        close=_decimal_or_none(row.get("Close")),
                        volume=_decimal_or_none(row.get("Volume")),
                        adjusted_close=_decimal_or_none(row.get("Adj Close")),
                        source=self.name,
                    )
                )
            return ProviderBars(bars=tuple(bars), metadata=metadata)
        except (DataUnavailableError, DataProviderError):
            raise
        except Exception as exc:  # provider boundary intentionally catches all
            if _is_temporary_error(exc):
                raise DataProviderTemporaryError(
                    f"temporary Yahoo bars failure for {request.symbol}"
                ) from exc
            raise DataProviderError(
                f"Yahoo bars failure for {request.symbol} {request.timeframe}"
            ) from exc

    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[CorporateAction, ...]:
        try:
            import pandas as pd
            import yfinance as yf

            metadata = self.metadata(symbol)
            source_zone = ZoneInfo(metadata.exchange_timezone)
            frame = yf.Ticker(symbol).actions
            if frame is None or frame.empty:
                return ()
            actions: list[CorporateAction] = []
            for index, row in frame.iterrows():
                timestamp = index.to_pydatetime()
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=source_zone)
                timestamp = timestamp.astimezone(timezone.utc)
                if not start <= timestamp < end:
                    continue
                dividend = row.get("Dividends")
                split = row.get("Stock Splits")
                if dividend is not None and not pd.isna(dividend) and dividend > 0:
                    actions.append(
                        Dividend(symbol, timestamp, Decimal(str(dividend)), self.name)
                    )
                if split is not None and not pd.isna(split) and split > 0:
                    actions.append(
                        StockSplit(symbol, timestamp, Decimal(str(split)), self.name)
                    )
            return tuple(sorted(actions, key=lambda action: action.timestamp))
        except Exception as exc:  # provider boundary intentionally catches all
            if _is_temporary_error(exc):
                raise DataProviderTemporaryError(
                    f"temporary Yahoo corporate-actions failure for {symbol}"
                ) from exc
            raise DataProviderError(
                f"Yahoo corporate-actions failure for {symbol}"
            ) from exc
