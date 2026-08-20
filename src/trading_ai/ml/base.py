"""ML scorer interface."""

from abc import ABC, abstractmethod
from typing import Mapping

from trading_ai.core.models import Signal, TradingContext


class MLScorer(ABC):
    """Score a signal without portfolio, risk, execution, or broker authority."""

    @abstractmethod
    def score(
        self,
        context: TradingContext,
        signal: Signal,
        features: Mapping[str, float],
    ) -> float:
        """Return a normalized score; interpretation belongs to a strategy."""
