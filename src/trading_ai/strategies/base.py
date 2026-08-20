"""Strategy interface."""

from abc import ABC, abstractmethod
from typing import Sequence

from trading_ai.core.models import (
    MarketBar,
    PortfolioSnapshot,
    StrategyDecision,
    TradingContext,
)


class Strategy(ABC):
    """Produce opinions from data; never call execution or brokers directly."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable strategy identifier."""

    @abstractmethod
    def decide(
        self,
        context: TradingContext,
        market_data: Sequence[MarketBar],
        portfolio: PortfolioSnapshot,
    ) -> StrategyDecision:
        """Return a strategy decision with no execution authority."""
