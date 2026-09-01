"""IBKR TWS Paper adapter behind the Lot 9 Paper execution boundary."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from trading_ai.brokers.base import BrokerAdapter
from trading_ai.brokers.config import IBKRPaperConfig
from trading_ai.brokers.exceptions import (
    BrokerUnavailableError,
    ContractResolutionError,
    PaperExecutionLockedError,
    ReconciliationRequiredError,
)
from trading_ai.brokers.idempotency import IdempotencyRegistry, client_order_key
from trading_ai.brokers.ibkr.client import IBKRClientPort, OfficialIBAPIClient
from trading_ai.brokers.ibkr.contracts import (
    IBKRContractCandidate,
    IBKRContractResolver,
)
from trading_ai.brokers.ibkr.errors import normalize_ibkr_error
from trading_ai.brokers.ibkr.events import normalize_callback_event, parse_ibkr_timestamp
from trading_ai.brokers.ibkr.orders import contract_payload, to_ibkr_order
from trading_ai.brokers.ibkr.versioning import (
    IBKR_ADAPTER_NAME,
    IBKR_ADAPTER_VERSION,
    validate_sdk_version,
)
from trading_ai.brokers.models import (
    BrokerAccountIdentity,
    BrokerAccountSnapshot,
    BrokerCommissionReport,
    BrokerConnectionState,
    BrokerEnvironment,
    BrokerEvent,
    BrokerExecution,
    BrokerHealth,
    BrokerOrderRecord,
    BrokerOrderState,
    BrokerPosition,
    CommissionKnowledge,
    PaperMode,
    PaperSessionState,
)
from trading_ai.brokers.reconciliation import ReconciliationState
from trading_ai.brokers.state import BrokerOrderStateMachine
from trading_ai.core.hashing import stable_hash
from trading_ai.core.models import (
    ExecutionEnvironment,
    ExecutionReceipt,
    OrderSide,
    OrderType,
    RiskApprovedOrder,
    TradingContext,
    TradingProfileName,
)


class IBKRPaperAdapter(BrokerAdapter):
    """Asynchronous Paper adapter; direct submission is deliberately blocked."""

    name = IBKR_ADAPTER_NAME
    version = IBKR_ADAPTER_VERSION

    def __init__(
        self,
        config: IBKRPaperConfig,
        resolver: IBKRContractResolver,
        *,
        session_id: str,
        client: IBKRClientPort | None = None,
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.session_id = session_id
        self._lock = threading.RLock()
        self._state = BrokerConnectionState.DISCONNECTED
        self._events: list[BrokerEvent] = []
        self._account_values: dict[str, tuple[str, str]] = {}
        self._positions: dict[str, BrokerPosition] = {}
        self._orders: dict[str, BrokerOrderRecord] = {}
        self._order_by_broker_id: dict[str, str] = {}
        self._executions: dict[str, BrokerExecution] = {}
        self._commissions: dict[str, BrokerCommissionReport] = {}
        self._contract_candidates: dict[int, list[IBKRContractCandidate]] = {}
        self._account_identity: BrokerAccountIdentity | None = None
        self._raw_account_id: str | None = None
        self._next_request_id = 10000
        self._last_heartbeat: datetime | None = None
        self._clock_drift_seconds: float | None = None
        self._critical_errors: set[str] = set()
        self._idempotency = IdempotencyRegistry()
        self._state_ready = threading.Event()
        self._sync_pending: set[str] = set()
        self._sync_active = False
        self._sync_seen_order_keys: set[str] = set()
        self._sync_seen_execution_ids: set[str] = set()
        self._client = client or OfficialIBAPIClient(
            self._on_callback,
            max_messages_per_second=config.max_messages_per_second,
            expected_sdk_version=config.official_sdk_version,
        )

    @property
    def connection_state(self) -> BrokerConnectionState:
        return self._state

    @property
    def account_identity(self) -> BrokerAccountIdentity | None:
        return self._account_identity

    @property
    def broker_events(self) -> tuple[BrokerEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def commission_reports(self) -> tuple[BrokerCommissionReport, ...]:
        with self._lock:
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
    def sdk_version(self) -> str | None:
        return self._client.sdk_version

    @property
    def server_version(self) -> str | None:
        return self._client.server_version

    def connect(self) -> None:
        with self._lock:
            if self._state is not BrokerConnectionState.DISCONNECTED:
                raise BrokerUnavailableError("IBKR adapter is already connecting or connected")
            self._state = BrokerConnectionState.CONNECTING
        try:
            self._client.connect(
                self.config.host,
                self.config.port,
                self.config.client_id,
                self.config.request_timeout_seconds,
            )
            validate_sdk_version(self.config.official_sdk_version, self._client.sdk_version)
            deadline = datetime.now(timezone.utc).timestamp() + self.config.request_timeout_seconds
            while not self._client.account_ids and datetime.now(timezone.utc).timestamp() < deadline:
                threading.Event().wait(0.01)
            self._identify_account(self._client.account_ids)
            with self._lock:
                self._state = BrokerConnectionState.CONNECTED
                self._last_heartbeat = datetime.now(timezone.utc)
        except Exception:
            self._client.disconnect()
            with self._lock:
                self._state = BrokerConnectionState.DISCONNECTED
            raise

    def _identify_account(self, account_ids: tuple[str, ...]) -> None:
        if len(account_ids) != 1:
            raise BrokerUnavailableError(
                "Lot 9 requires exactly one explicitly allowlisted IBKR account"
            )
        raw = account_ids[0]
        salt = os.environ.get(self.config.account_hash_salt_env)
        if not salt:
            raise BrokerUnavailableError(
                f"missing account hash salt environment variable {self.config.account_hash_salt_env}"
            )
        digest = hashlib.sha256(f"{salt}:{raw}".encode("utf-8")).hexdigest()
        normalized_id = raw.upper()
        paper_pattern = normalized_id.startswith("DU")
        live_pattern = normalized_id.startswith("U") and not paper_pattern
        allowed = digest in self.config.allowed_account_hashes
        environment = (
            BrokerEnvironment.PAPER
            if paper_pattern
            else BrokerEnvironment.LIVE
            if live_pattern
            else BrokerEnvironment.UNKNOWN
        )
        self._raw_account_id = raw
        self._account_identity = BrokerAccountIdentity(
            broker="IBKR",
            account_hash=digest,
            account_masked="IBKR-****" + raw[-4:],
            environment=environment,
            base_currency="UNKNOWN",
            capabilities=tuple(sorted(("ACCOUNT", "CANCEL", "EXECUTIONS", "ORDERS", "POSITIONS"))),
            environment_verified=bool(paper_pattern and allowed),
            environment_evidence="IBKR_PAPER_ID_PATTERN_PLUS_SALTED_LOCAL_ALLOWLIST",
        )

    def disconnect(self) -> None:
        with self._lock:
            self._state = BrokerConnectionState.DISCONNECTED
        self._client.disconnect()
        self._raw_account_id = None

    def reconnect(self) -> None:
        with self._lock:
            self._state = BrokerConnectionState.RECONNECTING
            self._events.append(
                normalize_callback_event(
                    session_id=self.session_id,
                    kind="RECONNECTING",
                    payload={},
                    source_version=self.version,
                )
            )
        self._client.disconnect()
        with self._lock:
            self._state = BrokerConnectionState.DISCONNECTED
        self.connect()

    def submit_approved(
        self, approved_order: RiskApprovedOrder, context: TradingContext
    ) -> ExecutionReceipt:
        del approved_order, context
        raise PaperExecutionLockedError(
            "IBKRPaperAdapter cannot be called directly; use PaperExecutionBoundary"
        )

    def transmit_approved(
        self, approved_order: RiskApprovedOrder, context: TradingContext
    ) -> ExecutionReceipt:
        """Called only by a validated PaperExecutionBoundary."""

        if context.environment is not ExecutionEnvironment.PAPER:
            raise PaperExecutionLockedError("IBKR transport accepts PAPER context only")
        if context.profile is not TradingProfileName.BALANCED:
            raise PaperExecutionLockedError("IBKR transport keeps aggressive locked")
        self._assert_transport_armed()
        if self._state is not BrokerConnectionState.CONNECTED or not self._client.connected:
            raise BrokerUnavailableError("IBKR adapter is not connected")
        if self._raw_account_id is None or self._account_identity is None:
            raise BrokerUnavailableError("IBKR account has not been verified")
        order = approved_order.order
        key = client_order_key(
            self.session_id, order.order_id, approved_order.risk_decision.decision_id
        )
        existing = self._idempotency.assert_resubmission_safe(key)
        if existing is not None:
            if existing.broker_order_id is None:
                raise ReconciliationRequiredError("existing order lacks a broker mapping")
            return ExecutionReceipt(
                order_id=order.order_id,
                broker_order_id=existing.broker_order_id,
                accepted_at=existing.updated_at,
            )
        contract = self.resolver.resolve(order.symbol)
        broker_id = self._client.next_order_id
        if broker_id is None:
            raise BrokerUnavailableError("IBKR nextValidId is unavailable")
        now = datetime.now(timezone.utc)
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
            tif=self.config.tif,
            limit_price=order.limit_price,
            broker_order_id=str(broker_id),
        )
        record = BrokerOrderStateMachine.transition(
            record, BrokerOrderState.SUBMITTING, updated_at=now
        )
        self._idempotency.register(record)
        with self._lock:
            self._orders[key] = record
            self._order_by_broker_id[str(broker_id)] = key
        try:
            self._client.place_order(
                broker_id,
                contract_payload(contract),
                to_ibkr_order(approved_order, tif=self.config.tif, transmit=True),
                account_id=self._raw_account_id,
                client_order_key=key,
            )
        except Exception:
            uncertain = BrokerOrderStateMachine.transition(
                record,
                BrokerOrderState.RECONCILIATION_REQUIRED,
                updated_at=datetime.now(timezone.utc),
            )
            self._replace_order(uncertain)
            raise ReconciliationRequiredError(
                "IBKR submit outcome is unknown; blind retry is forbidden"
            )
        with self._lock:
            self._events.append(
                normalize_callback_event(
                    session_id=self.session_id,
                    kind="ORDER_SUBMITTED",
                    payload={
                        "broker_order_id": str(broker_id),
                        "client_order_key": key,
                    },
                    source_version=self.version,
                )
            )
        with self._lock:
            current = self._orders[key]
            if current.state is BrokerOrderState.SUBMITTING:
                current = BrokerOrderStateMachine.transition(
                    current,
                    BrokerOrderState.SUBMITTED,
                    updated_at=datetime.now(timezone.utc),
                )
                self._replace_order(current)
        return ExecutionReceipt(
            order_id=order.order_id,
            broker_order_id=str(broker_id),
            accepted_at=current.updated_at,
        )

    def cancel_order(self, internal_order_id: str) -> None:
        self._assert_transport_armed()
        matches = [item for item in self._orders.values() if item.internal_order_id == internal_order_id]
        if len(matches) != 1 or matches[0].external or matches[0].broker_order_id is None:
            raise ReconciliationRequiredError("cancel is limited to one mapped system Paper order")
        record = matches[0]
        requested = BrokerOrderStateMachine.transition(
            record,
            BrokerOrderState.CANCEL_REQUESTED,
            updated_at=datetime.now(timezone.utc),
        )
        self._replace_order(requested)
        self._client.cancel_order(int(record.broker_order_id))

    def _assert_transport_armed(self) -> None:
        """Defense in depth: no Lot 9 configuration can satisfy this check."""

        identity = self._account_identity
        if (
            self.config.mode is not PaperMode.PAPER_EXECUTION_ARMED
            or not self.config.paper_execution_armed
        ):
            raise PaperExecutionLockedError(
                "IBKR Paper transport is not armed in Lot 9"
            )
        if (
            identity is None
            or identity.environment is not BrokerEnvironment.PAPER
            or not identity.environment_verified
            or identity.account_hash not in self.config.allowed_account_hashes
        ):
            raise PaperExecutionLockedError(
                "IBKR transport account is not a verified allowlisted Paper account"
            )

    def request_contract_resolution(self, symbol: str) -> int:
        spec = self.resolver.configured(symbol)
        request_id = self._next_request_id
        self._next_request_id += 1
        self._contract_candidates[request_id] = []
        payload: dict[str, object] = {
            "symbol": spec.broker_symbol,
            "secType": spec.sec_type,
            "exchange": spec.exchange,
            "primaryExchange": spec.primary_exchange,
            "currency": spec.currency,
        }
        if spec.con_id is not None:
            payload["conId"] = spec.con_id
        self._client.request_contract_details(request_id, payload)
        return request_id

    def finalize_contract_resolution(self, symbol: str, request_id: int) -> IBKRContractCandidate:
        return self.resolver.resolve(symbol, tuple(self._contract_candidates.get(request_id, ())))

    def sync_state(self) -> ReconciliationState:
        if self._state is not BrokerConnectionState.CONNECTED:
            raise BrokerUnavailableError("cannot synchronize a disconnected IBKR adapter")
        with self._lock:
            self._state_ready.clear()
            self._sync_pending = {
                "ACCOUNT_SUMMARY_END",
                "POSITIONS_END",
                "OPEN_ORDERS_END",
                "COMPLETED_ORDERS_END",
                "EXECUTIONS_END",
            }
            self._sync_active = True
            self._sync_seen_order_keys.clear()
            self._sync_seen_execution_ids.clear()
            self._account_values.clear()
            self._positions.clear()
        try:
            self._client.request_state()
            if not self._state_ready.wait(self.config.request_timeout_seconds):
                raise BrokerUnavailableError("IBKR state synchronization timed out")
            snapshot = self.account_snapshot()
            with self._lock:
                orders = tuple(
                    self._orders[key]
                    for key in sorted(self._sync_seen_order_keys)
                    if key in self._orders
                )
                executions = tuple(
                    self._executions[key]
                    for key in sorted(self._sync_seen_execution_ids)
                    if key in self._executions
                )
                self._critical_errors.difference_update(
                    {
                        "IBKR_CONNECTIVITY_LOST",
                        "IBKR_CONNECTIVITY_RESTORED_DATA_LOST",
                        "IBKR_SOCKET_PORT_RESET",
                        "IBKR_SERVER_CONNECTIVITY_BROKEN",
                        "IBKR_NOT_CONNECTED",
                    }
                )
        finally:
            with self._lock:
                self._sync_active = False
        return ReconciliationState(
            cash=snapshot.cash,
            positions=snapshot.positions,
            orders=orders,
            executions=executions,
        )

    def account_snapshot(self) -> BrokerAccountSnapshot:
        account = self._account_identity
        if account is None:
            raise BrokerUnavailableError("IBKR account identity is unavailable")
        cash = self._decimal_account_value("TotalCashValue")
        net_liquidation = self._decimal_account_value("NetLiquidation")
        currency = (
            self._account_values.get("NetLiquidation", ("", ""))[1]
            or self._account_values.get("TotalCashValue", ("", ""))[1]
            or account.base_currency
        )
        if currency and currency != account.base_currency:
            account = replace(account, base_currency=currency)
            self._account_identity = account
        return BrokerAccountSnapshot(
            observed_at=datetime.now(timezone.utc),
            account=account,
            cash=cash,
            net_liquidation=net_liquidation,
            positions=self.positions(),
        )

    def _decimal_account_value(self, name: str) -> Decimal:
        try:
            return Decimal(self._account_values[name][0])
        except (KeyError, InvalidOperation) as exc:
            raise BrokerUnavailableError(f"IBKR account value {name} is unavailable") from exc

    def positions(self) -> tuple[BrokerPosition, ...]:
        with self._lock:
            return tuple(self._positions[key] for key in sorted(self._positions))

    def open_orders(self) -> tuple[BrokerOrderRecord, ...]:
        terminal = {BrokerOrderState.FILLED, BrokerOrderState.CANCELLED, BrokerOrderState.REJECTED}
        with self._lock:
            return tuple(
                sorted((item for item in self._orders.values() if item.state not in terminal), key=lambda x: x.client_order_key)
            )

    def completed_orders(self) -> tuple[BrokerOrderRecord, ...]:
        terminal = {BrokerOrderState.FILLED, BrokerOrderState.CANCELLED, BrokerOrderState.REJECTED}
        with self._lock:
            return tuple(
                sorted((item for item in self._orders.values() if item.state in terminal), key=lambda x: x.client_order_key)
            )

    def executions(self) -> tuple[BrokerExecution, ...]:
        with self._lock:
            return tuple(self._executions[key] for key in sorted(self._executions))

    def health(self) -> BrokerHealth:
        now = datetime.now(timezone.utc)
        stale = (
            self._last_heartbeat is None
            or (now - self._last_heartbeat).total_seconds() > self.config.heartbeat_timeout_seconds
        )
        connection = BrokerConnectionState.STALE if stale and self._state is BrokerConnectionState.CONNECTED else self._state
        return BrokerHealth(
            observed_at=now,
            connection_state=connection,
            session_state=(
                PaperSessionState.DEGRADED
                if stale else PaperSessionState.CONNECTED_UNVERIFIED
            ),
            stale=stale,
            last_heartbeat_at=self._last_heartbeat,
            critical_errors=tuple(sorted(self._critical_errors)),
            clock_drift_seconds=self._clock_drift_seconds,
        )

    def heartbeat(self) -> None:
        self._client.request_current_time()

    def _replace_order(self, record: BrokerOrderRecord) -> None:
        with self._lock:
            self._orders[record.client_order_key] = record
            self._idempotency.update(record)

    def _on_callback(self, kind: str, payload: dict[str, object]) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._last_heartbeat = now
            event_payload = dict(payload)
            if kind == "EXECUTION":
                exec_id = str(payload.get("exec_id", ""))
                root = exec_id.rsplit(".", 1)[0]
                prior = next(
                    (
                        item.exec_id
                        for item in self._executions.values()
                        if item.exec_id.rsplit(".", 1)[0] == root
                    ),
                    None,
                )
                if prior is not None:
                    event_payload["correction_of"] = prior
                key = str(payload.get("client_order_key", ""))
                order = self._orders.get(key)
                if order is not None and prior is None:
                    cumulative = sum(
                        (
                            item.quantity
                            for item in self._executions.values()
                            if item.client_order_key == key
                            and item.correction_of is None
                        ),
                        Decimal("0"),
                    ) + Decimal(str(payload.get("quantity", "0")))
                    event_payload["is_partial"] = cumulative < order.quantity
            event = normalize_callback_event(
                session_id=self.session_id,
                kind=kind,
                payload=event_payload,
                source_version=self.version,
                received_at=now,
            )
            self._events.append(event)
            if kind == "ACCOUNT_SUMMARY":
                self._account_values[str(payload["tag"])] = (
                    str(payload["value"]), str(payload.get("currency", ""))
                )
            elif kind == "POSITION":
                symbol = self._configured_symbol_for_callback(payload)
                quantity = Decimal(str(payload["quantity"]))
                if quantity == Decimal("0"):
                    self._positions.pop(symbol, None)
                else:
                    self._positions[symbol] = BrokerPosition(
                        symbol=symbol,
                        quantity=quantity,
                        average_cost=Decimal(str(payload["average_cost"])),
                        currency=str(payload["currency"]),
                    )
            elif kind == "OPEN_ORDER":
                self._ingest_open_order(payload, now)
                self._mark_sync_order_seen(payload)
            elif kind == "COMPLETED_ORDER":
                self._ingest_completed_order(payload, now)
                self._mark_sync_order_seen(payload)
            elif kind == "ORDER_STATUS":
                self._ingest_order_status(payload, now)
            elif kind == "EXECUTION":
                self._ingest_execution(event_payload, now)
                if self._sync_active and str(payload["exec_id"]) in self._executions:
                    self._sync_seen_execution_ids.add(str(payload["exec_id"]))
            elif kind == "COMMISSION_REPORT":
                self._ingest_commission(payload, now)
            elif kind == "CONTRACT_DETAILS":
                request_id = int(payload["request_id"])
                self._contract_candidates.setdefault(request_id, []).append(
                    IBKRContractCandidate(
                        con_id=int(payload["con_id"]),
                        symbol=str(payload["symbol"]),
                        sec_type=str(payload["sec_type"]),
                        exchange=str(payload["exchange"]),
                        primary_exchange=str(payload["primary_exchange"]),
                        currency=str(payload["currency"]),
                        local_symbol=str(payload["local_symbol"]),
                    )
                )
            elif kind == "CURRENT_TIME":
                self._last_heartbeat = now
                broker_time = datetime.fromtimestamp(int(payload["epoch"]), timezone.utc)
                self._clock_drift_seconds = abs((now - broker_time).total_seconds())
                if self._clock_drift_seconds > self.config.max_clock_drift_seconds:
                    self._critical_errors.add("BROKER_CLOCK_DRIFT")
            elif kind == "ERROR":
                normalized = normalize_ibkr_error(int(payload["code"]))
                if normalized.connectivity_lost:
                    self._state = BrokerConnectionState.STALE
                if normalized.reconciliation_required:
                    self._critical_errors.add(normalized.stable_code)
            elif kind == "DISCONNECTED":
                self._state = BrokerConnectionState.DISCONNECTED
            if kind in self._sync_pending:
                self._sync_pending.discard(kind)
                if not self._sync_pending:
                    self._state_ready.set()

    def _mark_sync_order_seen(self, payload: dict[str, object]) -> None:
        if not self._sync_active:
            return
        key = str(payload.get("client_order_key", ""))
        if key not in self._orders:
            key = self._order_by_broker_id.get(
                str(payload["broker_order_id"]), key
            )
        if key in self._orders:
            self._sync_seen_order_keys.add(key)

    def _configured_symbol_for_callback(self, payload: dict[str, object]) -> str:
        return self.resolver.configured_symbol_for(
            str(payload["symbol"]), str(payload["currency"])
        )

    def _callback_order_type(self, payload: dict[str, object]) -> OrderType | None:
        raw = str(payload.get("order_type", ""))
        if raw == "MKT":
            return OrderType.MARKET
        if raw == "LMT":
            return OrderType.LIMIT
        self._critical_errors.add("UNSUPPORTED_BROKER_ORDER_TYPE")
        return None

    def _validate_broker_order_echo(
        self,
        record: BrokerOrderRecord,
        payload: dict[str, object],
        *,
        symbol: str,
        order_type: OrderType,
    ) -> bool:
        try:
            matches = (
                record.symbol == symbol
                and record.side is OrderSide(str(payload["side"]))
                and record.order_type is order_type
                and record.quantity == Decimal(str(payload["quantity"]))
                and str(payload.get("tif", "")) == record.tif
            )
        except (KeyError, InvalidOperation, ValueError):
            matches = False
        if not matches:
            self._critical_errors.add("BROKER_ORDER_ECHO_MISMATCH")
        return matches

    def _ingest_open_order(self, payload: dict[str, object], now: datetime) -> None:
        key = str(payload.get("client_order_key", ""))
        broker_id = str(payload["broker_order_id"])
        symbol = self._configured_symbol_for_callback(payload)
        order_type = self._callback_order_type(payload)
        if order_type is None:
            return
        existing = self._orders.get(key)
        if existing is None:
            external_key = key or f"external-{broker_id}"
            limit_raw = Decimal(str(payload.get("limit_price", "0")))
            record = BrokerOrderRecord(
                internal_order_id=f"external-{broker_id}",
                client_order_key=external_key,
                session_id=self.session_id,
                symbol=symbol,
                side=OrderSide(str(payload["side"])),
                order_type=order_type,
                quantity=Decimal(str(payload["quantity"])),
                filled_quantity=Decimal("0"),
                state=BrokerOrderState.ACKNOWLEDGED,
                risk_decision_id="EXTERNAL_BROKER_ACTIVITY",
                created_at=now,
                updated_at=now,
                tif=str(payload.get("tif", "DAY")),
                limit_price=(limit_raw if order_type is OrderType.LIMIT else None),
                broker_order_id=broker_id,
                perm_id=str(payload.get("perm_id") or "") or None,
                external=True,
            )
            self._orders[external_key] = record
            self._order_by_broker_id[broker_id] = external_key
            self._critical_errors.add("EXTERNAL_BROKER_ACTIVITY")
            return
        if not self._validate_broker_order_echo(
            existing, payload, symbol=symbol, order_type=order_type
        ):
            return
        if existing.state in {BrokerOrderState.SUBMITTING, BrokerOrderState.SUBMITTED}:
            updated = BrokerOrderStateMachine.transition(
                existing,
                BrokerOrderState.ACKNOWLEDGED,
                updated_at=now,
                broker_order_id=broker_id,
                perm_id=str(payload.get("perm_id") or "") or None,
            )
            self._replace_order(updated)

    def _ingest_order_status(self, payload: dict[str, object], now: datetime) -> None:
        key = self._order_by_broker_id.get(str(payload["broker_order_id"]))
        if key is None or key not in self._orders:
            self._critical_errors.add("UNKNOWN_BROKER_ORDER")
            return
        record = self._orders[key]
        status = str(payload["status"])
        filled = Decimal(str(payload["filled"]))
        target = {
            "Submitted": BrokerOrderState.ACKNOWLEDGED,
            "PreSubmitted": BrokerOrderState.ACKNOWLEDGED,
            "Filled": BrokerOrderState.FILLED,
            "Cancelled": BrokerOrderState.CANCELLED,
            "ApiCancelled": BrokerOrderState.CANCELLED,
            "Inactive": BrokerOrderState.REJECTED,
            "PendingCancel": BrokerOrderState.CANCEL_REQUESTED,
        }.get(status)
        if target is None and filled > Decimal("0"):
            target = BrokerOrderState.PARTIALLY_FILLED
        if target is None or target == record.state:
            return
        try:
            updated = BrokerOrderStateMachine.transition(
                record, target, updated_at=now, filled_quantity=filled
            )
        except Exception:
            self._critical_errors.add("INVALID_ORDER_STATE_TRANSITION")
            return
        self._replace_order(updated)

    def _ingest_completed_order(
        self, payload: dict[str, object], now: datetime
    ) -> None:
        key = str(payload.get("client_order_key", ""))
        broker_id = str(payload["broker_order_id"])
        symbol = self._configured_symbol_for_callback(payload)
        order_type = self._callback_order_type(payload)
        if order_type is None:
            return
        status = str(payload.get("state", ""))
        target = {
            "Filled": BrokerOrderState.FILLED,
            "Cancelled": BrokerOrderState.CANCELLED,
            "ApiCancelled": BrokerOrderState.CANCELLED,
            "Inactive": BrokerOrderState.REJECTED,
        }.get(status, BrokerOrderState.UNKNOWN)
        existing = self._orders.get(key)
        if existing is not None:
            if not self._validate_broker_order_echo(
                existing, payload, symbol=symbol, order_type=order_type
            ):
                return
            if existing.state is target:
                return
            filled = existing.quantity if target is BrokerOrderState.FILLED else existing.filled_quantity
            try:
                updated = BrokerOrderStateMachine.transition(
                    existing,
                    target,
                    updated_at=now,
                    filled_quantity=filled,
                    broker_order_id=broker_id,
                    perm_id=str(payload.get("perm_id") or "") or None,
                )
            except Exception:
                self._critical_errors.add("INVALID_COMPLETED_ORDER_TRANSITION")
                return
            self._replace_order(updated)
            return
        external_key = key or f"external-{broker_id}"
        quantity = Decimal(str(payload["quantity"]))
        limit_raw = Decimal(str(payload.get("limit_price", "0")))
        self._orders[external_key] = BrokerOrderRecord(
            internal_order_id=f"external-{broker_id}",
            client_order_key=external_key,
            session_id=self.session_id,
            symbol=symbol,
            side=OrderSide(str(payload["side"])),
            order_type=order_type,
            quantity=quantity,
            filled_quantity=quantity if target is BrokerOrderState.FILLED else Decimal("0"),
            state=target,
            risk_decision_id="EXTERNAL_BROKER_ACTIVITY",
            created_at=now,
            updated_at=now,
            tif=str(payload.get("tif", "DAY")),
            limit_price=(limit_raw if order_type is OrderType.LIMIT else None),
            broker_order_id=broker_id,
            perm_id=str(payload.get("perm_id") or "") or None,
            external=True,
        )
        self._order_by_broker_id[broker_id] = external_key
        self._critical_errors.add("EXTERNAL_BROKER_ACTIVITY")

    def _ingest_execution(self, payload: dict[str, object], now: datetime) -> None:
        exec_id = str(payload["exec_id"])
        if exec_id in self._executions:
            return
        key = str(payload.get("client_order_key", ""))
        if key not in self._orders:
            key = self._order_by_broker_id.get(
                str(payload["broker_order_id"]), key
            )
        if key not in self._orders:
            side_raw = str(payload.get("side", ""))
            if side_raw in {"BOT", "BUY"}:
                side = OrderSide.BUY
            elif side_raw in {"SLD", "SELL"}:
                side = OrderSide.SELL
            else:
                self._critical_errors.add("UNKNOWN_BROKER_EXECUTION_SIDE")
                return
            base = exec_id.rsplit(".", 1)[0]
            prior = next(
                (
                    item.exec_id
                    for item in self._executions.values()
                    if item.exec_id.rsplit(".", 1)[0] == base
                ),
                None,
            )
            external_key = key or f"external-execution-{payload['broker_order_id']}"
            self._executions[exec_id] = BrokerExecution(
                exec_id=exec_id,
                internal_order_id=f"external-{payload['broker_order_id']}",
                client_order_key=external_key,
                broker_order_id=str(payload["broker_order_id"]),
                perm_id=str(payload.get("perm_id") or "") or None,
                symbol=self._configured_symbol_for_callback(payload),
                side=side,
                quantity=Decimal(str(payload["quantity"])),
                price=Decimal(str(payload["price"])),
                broker_timestamp=parse_ibkr_timestamp(payload.get("time")),
                received_at=now,
                broker_timestamp_raw=(
                    str(payload["time"]) if payload.get("time") else None
                ),
                correction_of=prior,
            )
            self._critical_errors.add("EXTERNAL_BROKER_ACTIVITY")
            if prior is not None:
                self._critical_errors.add(
                    "EXECUTION_CORRECTION_REQUIRES_RECONCILIATION"
                )
            return
        record = self._orders[key]
        if self._configured_symbol_for_callback(payload) != record.symbol:
            self._critical_errors.add("BROKER_EXECUTION_CONTRACT_MISMATCH")
            return
        base = exec_id.rsplit(".", 1)[0]
        prior = next((item.exec_id for item in self._executions.values() if item.exec_id.rsplit(".", 1)[0] == base), None)
        execution = BrokerExecution(
            exec_id=exec_id,
            internal_order_id=record.internal_order_id,
            client_order_key=key,
            broker_order_id=str(payload["broker_order_id"]),
            perm_id=str(payload.get("perm_id") or "") or None,
            symbol=record.symbol,
            side=record.side,
            quantity=Decimal(str(payload["quantity"])),
            price=Decimal(str(payload["price"])),
            broker_timestamp=parse_ibkr_timestamp(payload.get("time")),
            received_at=now,
            broker_timestamp_raw=(
                str(payload["time"]) if payload.get("time") else None
            ),
            correction_of=prior,
        )
        self._executions[exec_id] = execution
        if prior is not None:
            self._critical_errors.add(
                "EXECUTION_CORRECTION_REQUIRES_RECONCILIATION"
            )
        cumulative = sum(
            (item.quantity for item in self._executions.values() if item.client_order_key == key and item.correction_of is None),
            Decimal("0"),
        )
        if cumulative > record.quantity:
            self._critical_errors.add("EXECUTION_QUANTITY_EXCEEDS_APPROVAL")
        elif prior is None:
            target = (
                BrokerOrderState.FILLED
                if cumulative == record.quantity
                else BrokerOrderState.PARTIALLY_FILLED
            )
            if target is not record.state:
                try:
                    self._replace_order(
                        BrokerOrderStateMachine.transition(
                            record,
                            target,
                            updated_at=now,
                            filled_quantity=cumulative,
                        )
                    )
                except Exception:
                    self._critical_errors.add("INVALID_EXECUTION_STATE_TRANSITION")

    def _ingest_commission(self, payload: dict[str, object], now: datetime) -> None:
        exec_id = str(payload["exec_id"])
        if exec_id not in self._executions:
            self._critical_errors.add("COMMISSION_WITHOUT_EXECUTION")
            return
        report = BrokerCommissionReport(
            exec_id=exec_id,
            status=CommissionKnowledge.KNOWN,
            received_at=now,
            amount=Decimal(str(payload["commission"])),
            currency=str(payload["currency"]),
        )
        existing = self._commissions.get(exec_id)
        if existing is not None:
            if (
                existing.status is report.status
                and existing.amount == report.amount
                and existing.currency == report.currency
            ):
                return
            self._critical_errors.add("COMMISSION_CORRECTION_REQUIRES_REVIEW")
            return
        self._commissions[exec_id] = report

    def restore_local_state(
        self,
        orders: tuple[BrokerOrderRecord, ...],
        executions: tuple[BrokerExecution, ...],
    ) -> None:
        """Restore only local mappings; startup still re-reads and reconciles TWS."""

        with self._lock:
            self._orders = {item.client_order_key: item for item in orders}
            self._order_by_broker_id = {
                str(item.broker_order_id): item.client_order_key
                for item in orders
                if item.broker_order_id is not None
            }
            self._executions = {item.exec_id: item for item in executions}
            self._idempotency = IdempotencyRegistry(orders)
