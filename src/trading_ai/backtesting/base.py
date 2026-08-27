"""Provider- and broker-independent Backtester contract."""

from abc import ABC, abstractmethod
from typing import Sequence

from trading_ai.backtesting.models import BacktestConfig, BacktestDataset
from trading_ai.backtesting.strategy import BacktestStrategy
from trading_ai.core.models import BacktestResult, TradingContext


class Backtester(ABC):
    """Evaluate a strategy against validated inputs without network access."""

    @abstractmethod
    def run(
        self,
        strategy: BacktestStrategy,
        datasets: Sequence[BacktestDataset],
        context: TradingContext,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Run one deterministic chronological simulation."""
