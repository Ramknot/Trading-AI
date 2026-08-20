"""Backtester interface."""

from abc import ABC, abstractmethod
from typing import Sequence

from trading_ai.core.models import BacktestResult, MarketBar, TradingContext
from trading_ai.strategies.base import Strategy


class Backtester(ABC):
    """Evaluate a strategy against historical bars without broker access."""

    @abstractmethod
    def run(
        self,
        strategy: Strategy,
        market_data: Sequence[MarketBar],
        context: TradingContext,
    ) -> BacktestResult:
        """Run one deterministic backtest and return a typed result envelope."""
