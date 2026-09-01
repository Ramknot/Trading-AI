"""Deterministic fault-injectable broker used only by offline tests and smokes."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from trading_ai.brokers.base import BrokerAdapter
from trading_ai.brokers.exceptions import (
    BrokerUnavailableError,
    PaperExecutionLockedError,
    ReconciliationRequiredError,
)
from trading_ai.brokers.idempotency import IdempotencyRegistry, client_order_key
from trading_ai.brokers.ledger import PaperLedger
from trading_ai.brokers.models import (
    BrokerAccountIdentity,
    BrokerAccountSnapshot,
    BrokerCommissionReport,
    BrokerConnectionState,
    BrokerEnvironment,
    BrokerEvent,
    BrokerEventType,
    BrokerExecution,
    BrokerHealth,
    BrokerOrderRecord,
    BrokerOrderState,
    BrokerPosition,
    CommissionKnowledge,
    PaperSessionState,
)
from trading_ai.brokers.reconciliation import ReconciliationState
from trading_ai.brokers.state import BrokerOrderStateMachine
from trading_ai.core.models import ExecutionReceipt, RiskApprovedOrder, TradingContext


@dataclass(frozen=True, slots=True)
class FakeBrokerScenario:
    reject: bool = False
    disconnect_before_submit: bool = False
    disconnect_during_submit: bool = False
    disconnect_after_submit: bool = False
    partial_fill_quantities: tuple[Decimal, ...] = ()
    fill_price: Decimal = Decimal("100")
    commission: Decimal | None = Decimal("1")
    duplicate_execution_callbacks: bool = False
    execution_correction_price: Decimal | None = None
    stale: bool = False


class FakeBroker(BrokerAdapter):
    """No production registry exposes this deterministic simulation broker."""

    name = "fake-paper-broker"
    version = "1.0"

    def __init__(
        self,
        *,
        session_id: str,
        account: BrokerAccountIdentity,
        cash: Decimal = Decimal("100000"),
        positions: tuple[BrokerPosition, ...] = (),
        scenario: FakeBrokerScenario = FakeBrokerScenario(),
        start_time: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) -> None:
        if start_time.tzinfo is None or start_time.utcoffset() != timedelta(0):
            raise ValueError("FakeBroker start_time must be UTC-aware")
        self.session_id = session_id
        self.account = account
        self.cash = cash
        self.net_liquidation = cash
        self._positions = {item.symbol: item for item in positions}
        self._ledger = PaperLedger(
            cash=cash,
            base_currency=account.base_currency,
            positions=positions,
        )
        self._last_prices: dict[str, Decimal] = {
            item.symbol: item.average_cost for item in positions
        }
        self.scenario = scenario
        self._connection = BrokerConnectionState.DISCONNECTED
        self._orders: dict[str, BrokerOrderRecord] = {}
        self._executions: dict[str, BrokerExecution] = {}
        self._commissions: dict[str, BrokerCommissionReport] = {}
        self._events: list[BrokerEvent] = []
        self._event_counter = 0
        self._clock_counter = 0
        self._start_time = start_time
        self._registry = IdempotencyRegistry()
        self.transmission_count = 0
        self.reconnect_count = 0

    @property
    def connection_state(self) -> BrokerConnectionState:
        return self._connection

    @property
    def commission_reports(self) -> tuple[BrokerCommissionReport, ...]:
        return tuple(
            self._commissions.get(
                key,
                BrokerCommissionReport(
                    exec_id=key,
                    status=CommissionKnowledge.UNAVAILABLE,
                    received_at=self._executions[key].received_at,
                ),
            )
            for key in sorted(self._executions)
        )

    @property
    def broker_events(self) -> tuple[BrokerEvent, ...]:
        return tuple(self._events)

    def _now(self) -> datetime:
        value = self._start_time + timedelta(microseconds=self._clock_counter)
        self._clock_counter += 1
        return value

    def _emit(
        self,
        event_type: BrokerEventType,
        *,
        related_ids: tuple[tuple[str, str], ...] = (),
        payload: dict[str, object] | None = None,
    ) -> None:
        self._event_counter += 1
        now = self._now()
        self._events.append(
            BrokerEvent(
                event_id=f"fake-event-{self._event_counter:06d}",
                session_id=self.session_id,
                event_type=event_type,
                received_at=now,
                broker_timestamp=now,
                source=self.name,
                source_version=self.version,
                related_ids=tuple(sorted(related_ids)),
                payload_json=json.dumps(
                    payload or {}, sort_keys=True, separators=(",", ":")
                ),
            )
        )

    def connect(self) -> None:
        if self._connection is not BrokerConnectionState.DISCONNECTED:
            raise BrokerUnavailableError("fake broker already connected")
        self._connection = BrokerConnectionState.CONNECTED
        self._emit(BrokerEventType.CONNECTED)

    def reconnect(self) -> None:
        self.reconnect_count += 1
        self._emit(BrokerEventType.RECONNECTING)
        self._connection = BrokerConnectionState.CONNECTED
        self._emit(BrokerEventType.CONNECTED)

    def disconnect(self) -> None:
        self._connection = BrokerConnectionState.DISCONNECTED
        self._emit(BrokerEventType.DISCONNECTED)

    def submit_approved(
        self, approved_order: RiskApprovedOrder, context: TradingContext
    ) -> ExecutionReceipt:
        del approved_order, context
        raise PaperExecutionLockedError(
            "FakeBroker direct submission is disabled; use PaperExecutionBoundary"
        )

    def transmit_approved(
        self, approved_order: RiskApprovedOrder, context: TradingContext
    ) -> ExecutionReceipt:
        del context
        if self.scenario.disconnect_before_submit:
            self.disconnect()
        if self._connection is not BrokerConnectionState.CONNECTED:
            raise BrokerUnavailableError("fake broker unavailable before submit")
        order = approved_order.order
        key = client_order_key(
            self.session_id, order.order_id, approved_order.risk_decision.decision_id
        )
        existing = self._registry.assert_resubmission_safe(key)
        if existing is not None:
            return ExecutionReceipt(
                order_id=order.order_id,
                broker_order_id=str(existing.broker_order_id),
                accepted_at=existing.updated_at,
            )
        now = self._now()
        broker_id = f"fake-{len(self._orders) + 1}"
        record = BrokerOrderRecord(
            internal_order_id=order.order_id,
            client_order_key=key,
            session_id=self.session_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            filled_quantity=Decimal("0"),
            state=BrokerOrderState.CREATED,
            risk_decision_id=approved_order.risk_decision.decision_id,
            created_at=now,
            updated_at=now,
            limit_price=order.limit_price,
            broker_order_id=broker_id,
        )
        record = BrokerOrderStateMachine.transition(
            record, BrokerOrderState.SUBMITTING, updated_at=now
        )
        self._registry.register(record)
        self._orders[key] = record
        self.transmission_count += 1
        self._emit(
            BrokerEventType.ORDER_SUBMITTED,
            related_ids=(("client_order_key", key), ("order_id", order.order_id)),
        )
        if self.scenario.disconnect_during_submit:
            uncertain = BrokerOrderStateMachine.transition(
                record,
                BrokerOrderState.RECONCILIATION_REQUIRED,
                updated_at=self._now(),
            )
            self._replace(uncertain)
            self.disconnect()
            raise ReconciliationRequiredError("fake disconnect during submit")
        if self.scenario.reject:
            rejected = BrokerOrderStateMachine.transition(
                record,
                BrokerOrderState.REJECTED,
                updated_at=self._now(),
                rejection_code="FAKE_REJECT",
            )
            self._replace(rejected)
            self._emit(
                BrokerEventType.ORDER_REJECT,
                related_ids=(("client_order_key", key),),
                payload={"code": "FAKE_REJECT"},
            )
            return ExecutionReceipt(order.order_id, broker_id, rejected.updated_at)
        submitted = BrokerOrderStateMachine.transition(
            record, BrokerOrderState.SUBMITTED, updated_at=self._now()
        )
        acknowledged = BrokerOrderStateMachine.transition(
            submitted,
            BrokerOrderState.ACKNOWLEDGED,
            updated_at=self._now(),
        )
        self._replace(acknowledged)
        self._emit(
            BrokerEventType.ORDER_ACK,
            related_ids=(("client_order_key", key), ("broker_order_id", broker_id)),
        )
        quantities = self.scenario.partial_fill_quantities or (order.quantity,)
        cumulative = Decimal("0")
        for index, quantity in enumerate(quantities, start=1):
            if quantity <= Decimal("0") or cumulative + quantity > order.quantity:
                raise ValueError("fake partial fills exceed the Risk-approved quantity")
            exec_id = f"exec-{broker_id}-{index}"
            execution = BrokerExecution(
                exec_id=exec_id,
                internal_order_id=order.order_id,
                client_order_key=key,
                broker_order_id=broker_id,
                perm_id=f"perm-{broker_id}",
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=self.scenario.fill_price,
                broker_timestamp=self._now(),
                received_at=self._now(),
            )
            is_new = exec_id not in self._executions
            self._executions.setdefault(exec_id, execution)
            if is_new:
                self._ledger.apply_execution(execution)
                self._last_prices[execution.symbol] = execution.price
                self._sync_account_from_ledger()
            if self.scenario.duplicate_execution_callbacks:
                self._executions.setdefault(exec_id, execution)
            cumulative += quantity
            target = (
                BrokerOrderState.FILLED
                if cumulative == order.quantity
                else BrokerOrderState.PARTIALLY_FILLED
            )
            acknowledged = BrokerOrderStateMachine.transition(
                acknowledged,
                target,
                updated_at=self._now(),
                filled_quantity=cumulative,
            )
            self._replace(acknowledged)
            self._emit(
                BrokerEventType.FILL
                if target is BrokerOrderState.FILLED
                else BrokerEventType.PARTIAL_FILL,
                related_ids=(
                    ("broker_order_id", broker_id),
                    ("client_order_key", key),
                    ("exec_id", exec_id),
                ),
                payload={"quantity": str(quantity), "price": str(execution.price)},
            )
        if self.scenario.execution_correction_price is not None and self._executions:
            original = self._executions[sorted(self._executions)[-1]]
            corrected = replace(
                original,
                exec_id=original.exec_id + ".1",
                price=self.scenario.execution_correction_price,
                correction_of=original.exec_id,
                received_at=self._now(),
            )
            self._executions[corrected.exec_id] = corrected
            self._ledger.apply_execution(corrected)
            self._last_prices[corrected.symbol] = corrected.price
            self._sync_account_from_ledger()
            self._emit(
                BrokerEventType.EXECUTION_CORRECTION,
                related_ids=(
                    ("correction_of", original.exec_id),
                    ("exec_id", corrected.exec_id),
                ),
            )
        corrected_execution_ids = {
            execution.correction_of
            for execution in self._executions.values()
            if execution.correction_of is not None
        }
        for execution in self._executions.values():
            if execution.exec_id in corrected_execution_ids:
                continue
            if execution.client_order_key == key and self.scenario.commission is not None:
                self._commissions[execution.exec_id] = BrokerCommissionReport(
                    exec_id=execution.exec_id,
                    status=CommissionKnowledge.KNOWN,
                    received_at=self._now(),
                    amount=self.scenario.commission,
                    currency=self.account.base_currency,
                )
                self._ledger.apply_commission(self._commissions[execution.exec_id])
                self._sync_account_from_ledger()
                self._emit(
                    BrokerEventType.COMMISSION_REPORT,
                    related_ids=(("exec_id", execution.exec_id),),
                    payload={
                        "amount": str(self.scenario.commission),
                        "currency": self.account.base_currency,
                    },
                )
        if self.scenario.disconnect_after_submit:
            self.disconnect()
        return ExecutionReceipt(order.order_id, broker_id, acknowledged.updated_at)

    def _replace(self, record: BrokerOrderRecord) -> None:
        self._orders[record.client_order_key] = record
        self._registry.update(record)

    def _sync_account_from_ledger(self) -> None:
        snapshot = self._ledger.snapshot
        self.cash = snapshot.cash
        self._positions = {item.symbol: item for item in snapshot.positions}
        market_value = sum(
            (
                position.quantity
                * self._last_prices.get(position.symbol, position.average_cost)
                for position in snapshot.positions
            ),
            Decimal("0"),
        )
        self.net_liquidation = self.cash + market_value

    def account_snapshot(self) -> BrokerAccountSnapshot:
        if self._connection is not BrokerConnectionState.CONNECTED:
            raise BrokerUnavailableError("fake broker is disconnected")
        return BrokerAccountSnapshot(
            observed_at=self._now(),
            account=self.account,
            cash=self.cash,
            net_liquidation=self.net_liquidation,
            positions=self.positions(),
        )

    def positions(self) -> tuple[BrokerPosition, ...]:
        return tuple(self._positions[key] for key in sorted(self._positions))

    def open_orders(self) -> tuple[BrokerOrderRecord, ...]:
        terminal = {BrokerOrderState.FILLED, BrokerOrderState.CANCELLED, BrokerOrderState.REJECTED}
        return tuple(
            sorted((item for item in self._orders.values() if item.state not in terminal), key=lambda x: x.client_order_key)
        )

    def completed_orders(self) -> tuple[BrokerOrderRecord, ...]:
        terminal = {BrokerOrderState.FILLED, BrokerOrderState.CANCELLED, BrokerOrderState.REJECTED}
        return tuple(
            sorted((item for item in self._orders.values() if item.state in terminal), key=lambda x: x.client_order_key)
        )

    def executions(self) -> tuple[BrokerExecution, ...]:
        return tuple(self._executions[key] for key in sorted(self._executions))

    def cancel_order(self, internal_order_id: str) -> None:
        matches = [item for item in self._orders.values() if item.internal_order_id == internal_order_id]
        if len(matches) != 1 or matches[0].external:
            raise ReconciliationRequiredError("cannot cancel an unknown or external fake order")
        requested = BrokerOrderStateMachine.transition(
            matches[0], BrokerOrderState.CANCEL_REQUESTED, updated_at=self._now()
        )
        cancelled = BrokerOrderStateMachine.transition(
            requested, BrokerOrderState.CANCELLED, updated_at=self._now()
        )
        self._replace(cancelled)
        self._emit(
            BrokerEventType.ORDER_CANCEL,
            related_ids=(("client_order_key", cancelled.client_order_key),),
        )

    def sync_state(self) -> ReconciliationState:
        snapshot = self.account_snapshot()
        return ReconciliationState(
            cash=snapshot.cash,
            positions=snapshot.positions,
            orders=tuple(sorted(self._orders.values(), key=lambda item: item.client_order_key)),
            executions=self.executions(),
        )

    def health(self) -> BrokerHealth:
        now = self._now()
        return BrokerHealth(
            observed_at=now,
            connection_state=(BrokerConnectionState.STALE if self.scenario.stale else self._connection),
            session_state=(PaperSessionState.DEGRADED if self.scenario.stale else PaperSessionState.CONNECTED_UNVERIFIED),
            stale=self.scenario.stale,
            last_heartbeat_at=now,
            critical_errors=(),
        )

    def inject_external_order(self, record: BrokerOrderRecord) -> None:
        if not record.external:
            raise ValueError("injected external activity must be explicitly marked external")
        self._orders[record.client_order_key] = record
        self._emit(
            BrokerEventType.EXTERNAL_BROKER_ACTIVITY,
            related_ids=(("broker_order_id", str(record.broker_order_id)),),
        )

    def restore_local_state(
        self,
        orders: tuple[BrokerOrderRecord, ...],
        executions: tuple[BrokerExecution, ...],
    ) -> None:
        self._orders = {item.client_order_key: item for item in orders}
        self._executions = {item.exec_id: item for item in executions}
        self._registry = IdempotencyRegistry(orders)
