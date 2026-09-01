"""Crash recovery from immutable local Paper evidence, never resubmission."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_ai.brokers.base import BrokerAdapter
from trading_ai.brokers.models import (
    BrokerExecution,
    BrokerOrderRecord,
    BrokerOrderState,
    BrokerPosition,
)
from trading_ai.brokers.reconciliation import ReconciliationState
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.core.models import OrderSide, OrderType


class PaperRecoveryService:
    def __init__(self, store: LocalPaperStore) -> None:
        self.store = store

    def restore(self, session_id: str, broker: BrokerAdapter) -> ReconciliationState:
        payload = self.store.inspect(session_id)
        orders_by_key: dict[str, BrokerOrderRecord] = {}
        for row in payload["orders"]:
            item = self._order(row)
            previous = orders_by_key.get(item.client_order_key)
            if previous is None or item.updated_at > previous.updated_at:
                orders_by_key[item.client_order_key] = item
        executions = tuple(sorted((self._execution(row) for row in payload["executions"]), key=lambda x: x.exec_id))
        latest_snapshot = max(
            payload["snapshots"], key=lambda row: str(row.get("observed_at", "")), default=None
        )
        positions = ()
        cash = Decimal("0")
        if latest_snapshot is not None:
            cash = Decimal(str(latest_snapshot["cash"]))
            positions = tuple(
                sorted(
                    (
                        BrokerPosition(
                            symbol=str(row["symbol"]),
                            quantity=Decimal(str(row["quantity"])),
                            average_cost=Decimal(str(row["average_cost"])),
                            currency=str(row["currency"]),
                        )
                        for row in latest_snapshot.get("positions", ())
                    ),
                    key=lambda item: item.symbol,
                )
            )
        orders = tuple(sorted(orders_by_key.values(), key=lambda item: item.client_order_key))
        restore = getattr(broker, "restore_local_state", None)
        if callable(restore):
            restore(orders, executions)
        return ReconciliationState(cash, positions, orders, executions)

    @staticmethod
    def _order(row: dict[str, object]) -> BrokerOrderRecord:
        return BrokerOrderRecord(
            internal_order_id=str(row["internal_order_id"]),
            client_order_key=str(row["client_order_key"]),
            session_id=str(row["session_id"]),
            symbol=str(row["symbol"]),
            side=OrderSide(str(row["side"])),
            order_type=OrderType(str(row["order_type"])),
            quantity=Decimal(str(row["quantity"])),
            filled_quantity=Decimal(str(row["filled_quantity"])),
            state=BrokerOrderState(str(row["state"])),
            risk_decision_id=str(row["risk_decision_id"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            tif=str(row.get("tif", "DAY")),
            limit_price=(Decimal(str(row["limit_price"])) if row.get("limit_price") is not None else None),
            broker_order_id=(str(row["broker_order_id"]) if row.get("broker_order_id") is not None else None),
            perm_id=(str(row["perm_id"]) if row.get("perm_id") is not None else None),
            rejection_code=(str(row["rejection_code"]) if row.get("rejection_code") is not None else None),
            external=bool(row.get("external", False)),
        )

    @staticmethod
    def _execution(row: dict[str, object]) -> BrokerExecution:
        return BrokerExecution(
            exec_id=str(row["exec_id"]),
            internal_order_id=str(row["internal_order_id"]),
            client_order_key=str(row["client_order_key"]),
            broker_order_id=str(row["broker_order_id"]),
            perm_id=(str(row["perm_id"]) if row.get("perm_id") is not None else None),
            symbol=str(row["symbol"]),
            side=OrderSide(str(row["side"])),
            quantity=Decimal(str(row["quantity"])),
            price=Decimal(str(row["price"])),
            broker_timestamp=(
                datetime.fromisoformat(str(row["broker_timestamp"]))
                if row.get("broker_timestamp") is not None else None
            ),
            received_at=datetime.fromisoformat(str(row["received_at"])),
            broker_timestamp_raw=(
                str(row["broker_timestamp_raw"])
                if row.get("broker_timestamp_raw") is not None
                else None
            ),
            correction_of=(str(row["correction_of"]) if row.get("correction_of") is not None else None),
        )
