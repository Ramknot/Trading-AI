"""Stable client-order keys and duplicate-submission protection."""

from __future__ import annotations

from threading import RLock

from trading_ai.brokers.exceptions import ReconciliationRequiredError
from trading_ai.brokers.models import BrokerOrderRecord, BrokerOrderState
from trading_ai.core.hashing import stable_hash


def client_order_key(session_id: str, order_id: str, risk_decision_id: str) -> str:
    return "broker-order-" + stable_hash(
        {
            "session_id": session_id,
            "order_id": order_id,
            "risk_decision_id": risk_decision_id,
        }
    )[:32]


class IdempotencyRegistry:
    """Thread-safe mapping; UNKNOWN submissions are never retried blindly."""

    def __init__(self, records: tuple[BrokerOrderRecord, ...] = ()) -> None:
        self._lock = RLock()
        self._by_key = {record.client_order_key: record for record in records}

    def lookup(self, key: str) -> BrokerOrderRecord | None:
        with self._lock:
            return self._by_key.get(key)

    def register(self, record: BrokerOrderRecord) -> BrokerOrderRecord:
        with self._lock:
            existing = self._by_key.get(record.client_order_key)
            if existing is not None and existing.internal_order_id != record.internal_order_id:
                raise ReconciliationRequiredError("client order key collision")
            if existing is not None:
                return existing
            self._by_key[record.client_order_key] = record
            return record

    def update(self, record: BrokerOrderRecord) -> None:
        with self._lock:
            existing = self._by_key.get(record.client_order_key)
            if existing is None or existing.internal_order_id != record.internal_order_id:
                raise ReconciliationRequiredError("cannot update an unknown client order key")
            self._by_key[record.client_order_key] = record

    def assert_resubmission_safe(self, key: str) -> BrokerOrderRecord | None:
        existing = self.lookup(key)
        if existing is not None and existing.state in {
            BrokerOrderState.SUBMITTING,
            BrokerOrderState.UNKNOWN,
            BrokerOrderState.RECONCILIATION_REQUIRED,
        }:
            raise ReconciliationRequiredError(
                "submission outcome is uncertain; reconcile instead of retrying"
            )
        return existing
