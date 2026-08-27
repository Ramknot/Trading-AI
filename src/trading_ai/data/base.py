"""Provider-independent historical market-data contract."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from trading_ai.core.models import MarketBar
from trading_ai.data.models import (
    CorporateAction,
    InstrumentMetadata,
    MarketDataRequest,
    ProviderBars,
)


class DataEngine(ABC):
    """Load normalized bars without coupling consumers to an implementation."""

    @abstractmethod
    def load_bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Return deterministic normalized bars for the requested assets."""


class DataProvider(ABC):
    """Translate one external source into Trading AI's neutral data models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable provider identifier used in manifests and storage paths."""

    @property
    @abstractmethod
    def version(self) -> str | None:
        """Provider library or adapter version when available."""

    @abstractmethod
    def fetch_bars(self, request: MarketDataRequest) -> ProviderBars:
        """Fetch untrusted provider rows for explicit normalization downstream."""

    @abstractmethod
    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[CorporateAction, ...]:
        """Fetch dividends and splits without applying them to prices."""

    @abstractmethod
    def metadata(self, symbol: str) -> InstrumentMetadata:
        """Return exchange, timezone, calendar, currency, and source metadata."""
