"""Deterministic offline provider for unit tests and local diagnostics."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from trading_ai.data.base import DataProvider
from trading_ai.data.exceptions import DataUnavailableError
from trading_ai.data.models import (
    CorporateAction,
    InstrumentMetadata,
    MarketDataRequest,
    ProviderBar,
    ProviderBars,
)


class FakeDataProvider(DataProvider):
    """Return caller-supplied data and never access the network."""

    def __init__(
        self,
        *,
        datasets: dict[tuple[str, str], tuple[ProviderBar, ...]] | None = None,
        metadata_by_symbol: dict[str, InstrumentMetadata] | None = None,
        actions_by_symbol: dict[str, tuple[CorporateAction, ...]] | None = None,
        warnings_by_dataset: dict[tuple[str, str], tuple[str, ...]] | None = None,
    ) -> None:
        self.datasets = datasets or {}
        self.metadata_by_symbol = metadata_by_symbol or {}
        self.actions_by_symbol = actions_by_symbol or {}
        self.warnings_by_dataset = warnings_by_dataset or {}
        self.calls: defaultdict[str, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def version(self) -> str:
        return "1"

    def metadata(self, symbol: str) -> InstrumentMetadata:
        self.calls["metadata"] += 1
        try:
            return self.metadata_by_symbol[symbol]
        except KeyError as exc:
            raise DataUnavailableError(f"fake metadata unavailable for {symbol}") from exc

    def fetch_bars(self, request: MarketDataRequest) -> ProviderBars:
        self.calls["fetch_bars"] += 1
        key = (request.symbol, request.timeframe)
        if key not in self.datasets:
            raise DataUnavailableError(
                f"fake bars unavailable for {request.symbol} {request.timeframe}"
            )
        metadata = self.metadata(request.symbol)
        bars = tuple(
            bar
            for bar in self.datasets[key]
            if (
                bar.timestamp.tzinfo is None
                or bar.timestamp.utcoffset() is None
                or request.start <= bar.timestamp < request.end
            )
        )
        if not bars:
            raise DataUnavailableError(
                f"fake bars empty for {request.symbol} {request.timeframe}"
            )
        return ProviderBars(
            bars=bars,
            metadata=metadata,
            warnings=self.warnings_by_dataset.get(key, ()),
        )

    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[CorporateAction, ...]:
        self.calls["fetch_corporate_actions"] += 1
        return tuple(
            action
            for action in self.actions_by_symbol.get(symbol, ())
            if start <= action.timestamp < end
        )
