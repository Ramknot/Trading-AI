"""Data engine interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from trading_ai.core.models import MarketBar


class DataEngine(ABC):
    """Load normalized market bars without coupling callers to a provider."""

    @abstractmethod
    def load_bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Return normalized, time-ordered bars for the requested window."""
