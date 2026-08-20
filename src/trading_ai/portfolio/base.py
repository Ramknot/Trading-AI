"""Portfolio engine interface."""

from abc import ABC, abstractmethod

from trading_ai.core.models import ExecutionReceipt, PortfolioSnapshot, TradingContext


class PortfolioEngine(ABC):
    """Own portfolio state without direct strategy-to-broker coupling."""

    @abstractmethod
    def snapshot(self, context: TradingContext) -> PortfolioSnapshot:
        """Return an immutable point-in-time portfolio snapshot."""

    @abstractmethod
    def record_execution(
        self, context: TradingContext, receipt: ExecutionReceipt
    ) -> None:
        """Record an already guarded execution receipt."""
