"""Exact local-versus-broker reconciliation with external-activity detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_ai.brokers.models import (
    BrokerExecution,
    BrokerOrderRecord,
    BrokerPosition,
    ReconciliationStatus,
    _text,
    _utc,
)
from trading_ai.core.hashing import stable_hash


@dataclass(frozen=True, slots=True)
class ReconciliationState:
    cash: Decimal
    positions: tuple[BrokerPosition, ...]
    orders: tuple[BrokerOrderRecord, ...]
    executions: tuple[BrokerExecution, ...]

    def __post_init__(self) -> None:
        if not self.cash.is_finite():
            raise ValueError("reconciliation cash must be finite")


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    reconciliation_id: str
    session_id: str
    observed_at: datetime
    status: ReconciliationStatus
    differences: tuple[str, ...]
    external_activity: tuple[str, ...]
    local_state_hash: str
    broker_state_hash: str

    def __post_init__(self) -> None:
        _text(self.reconciliation_id, "reconciliation_id")
        _text(self.session_id, "session_id")
        _utc(self.observed_at, "observed_at")
        if self.differences != tuple(sorted(set(self.differences))):
            raise ValueError("differences must be sorted and unique")
        if self.external_activity != tuple(sorted(set(self.external_activity))):
            raise ValueError("external_activity must be sorted and unique")

    @property
    def reconciliation_hash(self) -> str:
        return stable_hash(self)


class BrokerReconciler:
    def __init__(self, *, cash_tolerance: Decimal = Decimal("0.01")) -> None:
        if cash_tolerance < Decimal("0"):
            raise ValueError("cash_tolerance must not be negative")
        self.cash_tolerance = cash_tolerance

    def reconcile(
        self,
        *,
        session_id: str,
        observed_at: datetime,
        local: ReconciliationState,
        broker: ReconciliationState,
    ) -> ReconciliationResult:
        differences: set[str] = set()
        external: set[str] = set()
        if abs(local.cash - broker.cash) > self.cash_tolerance:
            differences.add("CASH_MISMATCH")
        if broker.cash < Decimal("0"):
            differences.add("NEGATIVE_BROKER_CASH")
        local_positions = {item.symbol: item.quantity for item in local.positions}
        broker_positions = {item.symbol: item.quantity for item in broker.positions}
        if local_positions != broker_positions:
            differences.add("POSITION_MISMATCH")
        local_order_keys = {item.client_order_key for item in local.orders}
        for item in broker.orders:
            if item.external or item.client_order_key not in local_order_keys:
                external.add(f"ORDER:{item.broker_order_id or item.internal_order_id}")
        local_exec_ids = {item.exec_id for item in local.executions}
        broker_exec_ids = {item.exec_id for item in broker.executions}
        for item in broker.executions:
            if item.exec_id not in local_exec_ids:
                external.add(f"EXECUTION:{item.exec_id}")
        if local_exec_ids - broker_exec_ids:
            differences.add("BROKER_EXECUTION_MISSING")
        if external:
            differences.add("EXTERNAL_BROKER_ACTIVITY")
        local_keys = {
            (item.client_order_key, item.state.value, str(item.filled_quantity))
            for item in local.orders
        }
        broker_keys = {
            (item.client_order_key, item.state.value, str(item.filled_quantity))
            for item in broker.orders
            if not item.external
        }
        if local_keys != broker_keys:
            differences.add("ORDER_STATE_MISMATCH")
        critical = bool(
            {
                "CASH_MISMATCH",
                "NEGATIVE_BROKER_CASH",
                "POSITION_MISMATCH",
                "EXTERNAL_BROKER_ACTIVITY",
                "BROKER_EXECUTION_MISSING",
                "ORDER_STATE_MISMATCH",
            }
            & differences
        )
        if critical:
            status = ReconciliationStatus.CRITICAL_DRIFT
        elif differences:
            status = ReconciliationStatus.DRIFT
        else:
            status = ReconciliationStatus.IN_SYNC
        payload = {
            "session_id": session_id,
            "observed_at": observed_at,
            "local": stable_hash(local),
            "broker": stable_hash(broker),
            "differences": sorted(differences),
            "external": sorted(external),
        }
        return ReconciliationResult(
            reconciliation_id="reconcile-" + stable_hash(payload)[:24],
            session_id=session_id,
            observed_at=observed_at,
            status=status,
            differences=tuple(sorted(differences)),
            external_activity=tuple(sorted(external)),
            local_state_hash=stable_hash(local),
            broker_state_hash=stable_hash(broker),
        )
