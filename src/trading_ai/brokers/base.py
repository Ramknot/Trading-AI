"""Provider-neutral broker adapter interface."""

from abc import ABC, abstractmethod
from typing import Any

from trading_ai.core.models import ExecutionReceipt, RiskApprovedOrder, TradingContext
from trading_ai.brokers.models import (
    BrokerAccountSnapshot,
    BrokerConnectionState,
    BrokerExecution,
    BrokerHealth,
    BrokerOrderRecord,
    BrokerPosition,
)


class BrokerAdapter(ABC):
    """Transmit only risk-approved envelopes supplied by ExecutionEngine.

    Strategies, ML scorers, and portfolio components must never hold this adapter.
    """

    @abstractmethod
    def submit_approved(
        self, approved_order: RiskApprovedOrder, context: TradingContext
    ) -> ExecutionReceipt:
        """Submit an already authorized order to one broker implementation."""

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    @property
    def connection_state(self) -> BrokerConnectionState:
        return BrokerConnectionState.DISCONNECTED

    def account_snapshot(self) -> BrokerAccountSnapshot:
        raise NotImplementedError

    def positions(self) -> tuple[BrokerPosition, ...]:
        return self.account_snapshot().positions

    def open_orders(self) -> tuple[BrokerOrderRecord, ...]:
        raise NotImplementedError

    def completed_orders(self) -> tuple[BrokerOrderRecord, ...]:
        raise NotImplementedError

    def executions(self) -> tuple[BrokerExecution, ...]:
        raise NotImplementedError

    def cancel_order(self, internal_order_id: str) -> None:
        raise NotImplementedError

    def sync_state(self) -> Any:
        raise NotImplementedError

    def health(self) -> BrokerHealth:
        raise NotImplementedError
