"""Deterministic order, session, and emergency-halt state machines."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from trading_ai.brokers.exceptions import BrokerStateTransitionError
from trading_ai.brokers.models import (
    BrokerOrderRecord,
    BrokerOrderState,
    PaperEmergencyHalt,
    PaperSessionState,
)


_ORDER_TRANSITIONS: dict[BrokerOrderState, frozenset[BrokerOrderState]] = {
    BrokerOrderState.CREATED: frozenset(
        {BrokerOrderState.SUBMITTING, BrokerOrderState.REJECTED}
    ),
    BrokerOrderState.SUBMITTING: frozenset(
        {
            BrokerOrderState.SUBMITTED,
            BrokerOrderState.ACKNOWLEDGED,
            BrokerOrderState.PARTIALLY_FILLED,
            BrokerOrderState.FILLED,
            BrokerOrderState.CANCELLED,
            BrokerOrderState.REJECTED,
            BrokerOrderState.UNKNOWN,
            BrokerOrderState.RECONCILIATION_REQUIRED,
        }
    ),
    BrokerOrderState.SUBMITTED: frozenset(
        {
            BrokerOrderState.ACKNOWLEDGED,
            BrokerOrderState.PARTIALLY_FILLED,
            BrokerOrderState.FILLED,
            BrokerOrderState.CANCEL_REQUESTED,
            BrokerOrderState.CANCELLED,
            BrokerOrderState.REJECTED,
            BrokerOrderState.UNKNOWN,
            BrokerOrderState.RECONCILIATION_REQUIRED,
        }
    ),
    BrokerOrderState.ACKNOWLEDGED: frozenset(
        {
            BrokerOrderState.PARTIALLY_FILLED,
            BrokerOrderState.FILLED,
            BrokerOrderState.CANCEL_REQUESTED,
            BrokerOrderState.CANCELLED,
            BrokerOrderState.REJECTED,
            BrokerOrderState.UNKNOWN,
            BrokerOrderState.RECONCILIATION_REQUIRED,
        }
    ),
    BrokerOrderState.PARTIALLY_FILLED: frozenset(
        {
            BrokerOrderState.PARTIALLY_FILLED,
            BrokerOrderState.FILLED,
            BrokerOrderState.CANCEL_REQUESTED,
            BrokerOrderState.CANCELLED,
            BrokerOrderState.UNKNOWN,
            BrokerOrderState.RECONCILIATION_REQUIRED,
        }
    ),
    BrokerOrderState.CANCEL_REQUESTED: frozenset(
        {
            BrokerOrderState.CANCELLED,
            BrokerOrderState.PARTIALLY_FILLED,
            BrokerOrderState.FILLED,
            BrokerOrderState.UNKNOWN,
            BrokerOrderState.RECONCILIATION_REQUIRED,
        }
    ),
    BrokerOrderState.UNKNOWN: frozenset({BrokerOrderState.RECONCILIATION_REQUIRED}),
    BrokerOrderState.RECONCILIATION_REQUIRED: frozenset(
        {
            BrokerOrderState.SUBMITTED,
            BrokerOrderState.ACKNOWLEDGED,
            BrokerOrderState.PARTIALLY_FILLED,
            BrokerOrderState.FILLED,
            BrokerOrderState.CANCELLED,
            BrokerOrderState.REJECTED,
        }
    ),
    BrokerOrderState.FILLED: frozenset(),
    BrokerOrderState.CANCELLED: frozenset(),
    BrokerOrderState.REJECTED: frozenset(),
}


_SESSION_TRANSITIONS: dict[PaperSessionState, frozenset[PaperSessionState]] = {
    PaperSessionState.DISCONNECTED: frozenset({PaperSessionState.CONNECTING}),
    PaperSessionState.CONNECTING: frozenset(
        {PaperSessionState.CONNECTED_UNVERIFIED, PaperSessionState.DISCONNECTED}
    ),
    PaperSessionState.CONNECTED_UNVERIFIED: frozenset(
        {
            PaperSessionState.PAPER_VERIFIED,
            PaperSessionState.HALTED,
            PaperSessionState.DEGRADED,
            PaperSessionState.STOPPING,
        }
    ),
    PaperSessionState.PAPER_VERIFIED: frozenset(
        {PaperSessionState.RECONCILING, PaperSessionState.HALTED, PaperSessionState.STOPPING}
    ),
    PaperSessionState.RECONCILING: frozenset(
        {
            PaperSessionState.READY,
            PaperSessionState.DEGRADED,
            PaperSessionState.HALTED,
            PaperSessionState.STOPPING,
        }
    ),
    PaperSessionState.READY: frozenset(
        {
            PaperSessionState.RECONCILING,
            PaperSessionState.DEGRADED,
            PaperSessionState.HALTED,
            PaperSessionState.STOPPING,
        }
    ),
    PaperSessionState.DEGRADED: frozenset(
        {
            PaperSessionState.RECONCILING,
            PaperSessionState.HALTED,
            PaperSessionState.STOPPING,
        }
    ),
    PaperSessionState.HALTED: frozenset(
        {PaperSessionState.RECONCILING, PaperSessionState.STOPPING}
    ),
    PaperSessionState.STOPPING: frozenset({PaperSessionState.DISCONNECTED}),
}


class BrokerOrderStateMachine:
    @staticmethod
    def transition(
        order: BrokerOrderRecord,
        state: BrokerOrderState,
        *,
        updated_at: datetime,
        filled_quantity=None,
        broker_order_id: str | None = None,
        perm_id: str | None = None,
        rejection_code: str | None = None,
    ) -> BrokerOrderRecord:
        if state not in _ORDER_TRANSITIONS[order.state]:
            raise BrokerStateTransitionError(
                f"invalid broker order transition {order.state.value}->{state.value}"
            )
        values = {
            "state": state,
            "updated_at": updated_at,
            "broker_order_id": broker_order_id or order.broker_order_id,
            "perm_id": perm_id or order.perm_id,
            "rejection_code": rejection_code or order.rejection_code,
        }
        if filled_quantity is not None:
            values["filled_quantity"] = filled_quantity
        return replace(order, **values)


class PaperSessionStateMachine:
    def __init__(self) -> None:
        self._state = PaperSessionState.DISCONNECTED

    @property
    def state(self) -> PaperSessionState:
        return self._state

    def transition(self, state: PaperSessionState) -> PaperSessionState:
        if state not in _SESSION_TRANSITIONS[self._state]:
            raise BrokerStateTransitionError(
                f"invalid Paper session transition {self._state.value}->{state.value}"
            )
        self._state = state
        return state


class PaperEmergencyHaltController:
    def __init__(self, initial: PaperEmergencyHalt) -> None:
        self._halt = initial

    @property
    def snapshot(self) -> PaperEmergencyHalt:
        return self._halt

    def halt(self, reason: str, *, at: datetime) -> PaperEmergencyHalt:
        self._halt = PaperEmergencyHalt(True, reason, at)
        return self._halt

    def reset(self, reason: str, *, at: datetime) -> PaperEmergencyHalt:
        self._halt = PaperEmergencyHalt(False, reason, at)
        return self._halt
