"""Risk engine interface."""

from abc import ABC, abstractmethod

from trading_ai.core.models import (
    OrderRequest,
    PortfolioSnapshot,
    RiskDecision,
    TradingContext,
)


class RiskEngine(ABC):
    """Mandatory authorization gate between order proposals and execution."""

    @abstractmethod
    def evaluate(
        self,
        order: OrderRequest,
        portfolio: PortfolioSnapshot,
        context: TradingContext,
    ) -> RiskDecision:
        """Return an explicit APPROVE or REJECT for exactly one order."""
