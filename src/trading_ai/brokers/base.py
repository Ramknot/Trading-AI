"""Broker adapter interface."""

from abc import ABC, abstractmethod

from trading_ai.core.models import ExecutionReceipt, RiskApprovedOrder, TradingContext


class BrokerAdapter(ABC):
    """Transmit only risk-approved envelopes supplied by ExecutionEngine.

    Strategies, ML scorers, and portfolio components must never hold this adapter.
    """

    @abstractmethod
    def submit_approved(
        self, approved_order: RiskApprovedOrder, context: TradingContext
    ) -> ExecutionReceipt:
        """Submit an already authorized order to one broker implementation."""
