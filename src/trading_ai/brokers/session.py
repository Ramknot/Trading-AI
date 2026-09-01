"""Read-only Lot 9 Paper-session lifecycle and startup reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trading_ai.brokers.base import BrokerAdapter
from trading_ai.brokers.exceptions import (
    BrokerConfigurationError,
    BrokerUnavailableError,
    ReconciliationRequiredError,
)
from trading_ai.brokers.models import (
    BrokerAccountIdentity,
    BrokerConnectionState,
    DataFreshnessStatus,
    PaperEmergencyHalt,
    PaperMode,
    PaperSafetySnapshot,
    PaperDecisionEnvelope,
    PaperOutcomeEnvelope,
    PaperSessionManifest,
    PaperSessionState,
    ReconciliationStatus,
)
from trading_ai.brokers.paper_guard import (
    DenyPaperSubmissionAuthorization,
    PaperAccountGuard,
    PaperExecutionBoundary,
    PaperSubmissionAuthorization,
)
from trading_ai.brokers.reconciliation import (
    BrokerReconciler,
    ReconciliationResult,
    ReconciliationState,
)
from trading_ai.brokers.state import PaperEmergencyHaltController, PaperSessionStateMachine
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.core.hashing import stable_hash


class PaperTradingSession:
    """Coordinates verification/reconciliation; Lot 9 never arms execution."""

    def __init__(
        self,
        broker: BrokerAdapter,
        *,
        session_id: str,
        mode: PaperMode,
        allowed_account_hashes: tuple[str, ...],
        config_hashes: tuple[tuple[str, str], ...],
        code_sha: str,
        ml_model_ids: tuple[str, ...] = (),
        store: LocalPaperStore | None = None,
        authorization: PaperSubmissionAuthorization | None = None,
        data_freshness: DataFreshnessStatus = DataFreshnessStatus.UNKNOWN,
        risk_health_ok: bool = False,
        market_session_open: bool | None = None,
    ) -> None:
        if mode is PaperMode.PAPER_EXECUTION_ARMED:
            raise BrokerConfigurationError(
                "Lot 9 sessions cannot start in PAPER_EXECUTION_ARMED mode"
            )
        self.broker = broker
        self.session_id = session_id
        self.mode = mode
        self._config_hashes = tuple(sorted(config_hashes))
        self.code_sha = code_sha
        self.ml_model_ids = tuple(sorted(ml_model_ids))
        self.store = store or LocalPaperStore()
        self._state_machine = PaperSessionStateMachine()
        now = datetime.now(timezone.utc)
        self._emergency = PaperEmergencyHaltController(
            PaperEmergencyHalt(False, "INITIALIZED_NOT_ARMED", now)
        )
        self._guard = PaperAccountGuard(allowed_account_hashes)
        self._authorization = authorization or DenyPaperSubmissionAuthorization()
        self._reconciler = BrokerReconciler()
        self._reconciliation: ReconciliationResult | None = None
        self._account: BrokerAccountIdentity | None = None
        self._data_freshness = data_freshness
        self._risk_health_ok = risk_health_ok
        self._market_session_open = market_session_open
        self._configs_frozen = False
        self.boundary = PaperExecutionBoundary(
            broker,
            safety_snapshot=self.safety_snapshot,
            account_guard=self._guard,
            authorization=self._authorization,
            after_attempt=self._persist_broker_state,
        )

    @property
    def state(self) -> PaperSessionState:
        return self._state_machine.state

    @property
    def reconciliation(self) -> ReconciliationResult | None:
        return self._reconciliation

    def start(self, *, local_state: ReconciliationState | None = None) -> PaperSessionState:
        self._state_machine.transition(PaperSessionState.CONNECTING)
        try:
            self.broker.connect()
            self._state_machine.transition(PaperSessionState.CONNECTED_UNVERIFIED)
            identity = getattr(self.broker, "account_identity", None)
            if identity is None:
                identity = self.broker.account_snapshot().account
            self._account = identity
            self._guard.verify(self.safety_snapshot())
            self._state_machine.transition(PaperSessionState.PAPER_VERIFIED)
            self._freeze_manifest()
            if self.mode is PaperMode.CONNECTIVITY_CHECK:
                self.broker.sync_state()
                self._account = self.broker.account_snapshot().account
                self._persist_broker_state()
                self._state_machine.transition(PaperSessionState.STOPPING)
                self.broker.disconnect()
                self._state_machine.transition(PaperSessionState.DISCONNECTED)
                return self.state
            self._state_machine.transition(PaperSessionState.RECONCILING)
            if local_state is None:
                self._state_machine.transition(PaperSessionState.DEGRADED)
                return self.state
            self._perform_reconciliation(local_state)
            return self.state
        except Exception:
            if self.state not in {PaperSessionState.DISCONNECTED, PaperSessionState.HALTED}:
                try:
                    self._state_machine.transition(PaperSessionState.HALTED)
                except Exception:
                    pass
            self._emergency.halt("STARTUP_VERIFICATION_FAILED", at=datetime.now(timezone.utc))
            try:
                self.broker.disconnect()
            except Exception:
                pass
            raise

    def _freeze_manifest(self) -> None:
        if self._account is None:
            raise BrokerUnavailableError("cannot freeze a Paper session without account identity")
        manifest = PaperSessionManifest(
            session_id=self.session_id,
            created_at=datetime.now(timezone.utc),
            code_sha=self.code_sha,
            mode=self.mode,
            broker_adapter_name=str(getattr(self.broker, "name", type(self.broker).__name__)),
            broker_adapter_version=str(getattr(self.broker, "version", "unknown")),
            official_sdk_version=getattr(self.broker, "sdk_version", None),
            server_version=getattr(self.broker, "server_version", None),
            account_hash=self._account.account_hash,
            account_masked=self._account.account_masked,
            config_hashes=self._config_hashes,
            ml_model_ids=self.ml_model_ids,
            paper_execution_armed=False,
        )
        self.store.create_session(manifest)
        self._configs_frozen = True

    def _perform_reconciliation(self, local_state: ReconciliationState) -> None:
        broker_state = self.broker.sync_state()
        now = datetime.now(timezone.utc)
        result = self._reconciler.reconcile(
            session_id=self.session_id,
            observed_at=now,
            local=local_state,
            broker=broker_state,
        )
        self._reconciliation = result
        self.store.append(
            self.session_id,
            "reconciliation",
            result,
            record_id=result.reconciliation_id,
        )
        self._persist_broker_state()
        health = self.broker.health()
        fully_healthy = (
            result.status is ReconciliationStatus.IN_SYNC
            and not health.stale
            and not health.critical_errors
            and self._data_freshness is DataFreshnessStatus.FRESH
            and self._risk_health_ok
            and not self._emergency.snapshot.active
        )
        if fully_healthy:
            self._state_machine.transition(PaperSessionState.READY)
        elif result.status is ReconciliationStatus.IN_SYNC:
            self._state_machine.transition(PaperSessionState.DEGRADED)
        elif result.status is ReconciliationStatus.DRIFT:
            self._state_machine.transition(PaperSessionState.DEGRADED)
        else:
            self._state_machine.transition(PaperSessionState.HALTED)
            self._emergency.halt(
                "CRITICAL_RECONCILIATION_DRIFT", at=datetime.now(timezone.utc)
            )

    def reconcile(self, local_state: ReconciliationState) -> ReconciliationResult:
        if self.state not in {
            PaperSessionState.READY,
            PaperSessionState.DEGRADED,
            PaperSessionState.HALTED,
            PaperSessionState.PAPER_VERIFIED,
        }:
            raise ReconciliationRequiredError("session is not connected for reconciliation")
        self._state_machine.transition(PaperSessionState.RECONCILING)
        self._perform_reconciliation(local_state)
        assert self._reconciliation is not None
        return self._reconciliation

    def reconnect(self, *, local_state: ReconciliationState) -> ReconciliationResult:
        if self.broker.connection_state is not BrokerConnectionState.CONNECTED:
            reconnect = getattr(self.broker, "reconnect", None)
            if callable(reconnect):
                reconnect()
            else:
                if self.broker.connection_state is not BrokerConnectionState.DISCONNECTED:
                    self.broker.disconnect()
                self.broker.connect()
        if self.state is PaperSessionState.DISCONNECTED:
            self._state_machine.transition(PaperSessionState.CONNECTING)
            self._state_machine.transition(PaperSessionState.CONNECTED_UNVERIFIED)
            self._state_machine.transition(PaperSessionState.PAPER_VERIFIED)
        return self.reconcile(local_state)

    def assert_frozen_configs(self, current: tuple[tuple[str, str], ...]) -> None:
        if tuple(sorted(current)) != self._config_hashes:
            self._emergency.halt("CONFIG_HASH_CHANGED", at=datetime.now(timezone.utc))
            if self.state not in {PaperSessionState.HALTED, PaperSessionState.STOPPING}:
                self._state_machine.transition(PaperSessionState.HALTED)
            raise BrokerConfigurationError(
                "Paper configuration changed during a frozen session; restart is required"
            )

    def set_data_freshness(self, status: DataFreshnessStatus) -> None:
        self._data_freshness = status
        if status is not DataFreshnessStatus.FRESH and self.state is PaperSessionState.READY:
            self._state_machine.transition(PaperSessionState.DEGRADED)

    def set_component_health(
        self,
        *,
        data_freshness: DataFreshnessStatus,
        risk_health_ok: bool,
        market_session_open: bool | None,
    ) -> None:
        self._data_freshness = data_freshness
        self._risk_health_ok = risk_health_ok
        self._market_session_open = market_session_open
        if (
            self.state is PaperSessionState.READY
            and (
                data_freshness is not DataFreshnessStatus.FRESH
                or not risk_health_ok
                or market_session_open is not True
            )
        ):
            self._state_machine.transition(PaperSessionState.DEGRADED)

    def reset_emergency_halt(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("emergency halt reset requires an explicit reason")
        self._emergency.reset(reason, at=datetime.now(timezone.utc))

    def emergency_halt(self, reason: str) -> None:
        self._emergency.halt(reason, at=datetime.now(timezone.utc))
        if self.state not in {PaperSessionState.HALTED, PaperSessionState.STOPPING, PaperSessionState.DISCONNECTED}:
            self._state_machine.transition(PaperSessionState.HALTED)

    def safety_snapshot(self) -> PaperSafetySnapshot:
        broker_health = self.broker.health()
        reconciliation_hash = (
            self._reconciliation.reconciliation_hash
            if self._reconciliation is not None
            else stable_hash({"session_id": self.session_id, "reconciliation": "UNAVAILABLE"})
        )
        return PaperSafetySnapshot(
            session_id=self.session_id,
            observed_at=datetime.now(timezone.utc),
            mode=self.mode,
            session_state=self.state,
            account=self._account,
            connection_state=broker_health.connection_state,
            reconciliation_status=(
                self._reconciliation.status
                if self._reconciliation is not None
                else ReconciliationStatus.UNKNOWN
            ),
            reconciliation_hash=reconciliation_hash,
            data_freshness=self._data_freshness,
            risk_health_ok=self._risk_health_ok,
            market_session_open=self._market_session_open,
            configs_frozen=self._configs_frozen,
            emergency_halt=self._emergency.snapshot,
        )

    def _persist_broker_state(self) -> None:
        try:
            snapshot = self.broker.account_snapshot()
        except BrokerUnavailableError:
            snapshot = None
        if snapshot is not None:
            self.store.append(
                self.session_id,
                "snapshots",
                snapshot,
                record_id="account-" + stable_hash(snapshot)[:24],
            )
        for order in (*self.broker.open_orders(), *self.broker.completed_orders()):
            record_id = "order-" + stable_hash(order)[:24]
            self.store.append(self.session_id, "orders", order, record_id=record_id)
        for execution in self.broker.executions():
            self.store.append(
                self.session_id,
                "executions",
                execution,
                record_id="execution-" + stable_hash(execution)[:24],
            )
        for commission in getattr(self.broker, "commission_reports", ()):
            self.store.append(
                self.session_id,
                "commissions",
                commission,
                record_id="commission-" + stable_hash(commission)[:24],
            )
        for event in getattr(self.broker, "broker_events", ()):
            self.store.append(
                self.session_id,
                "events",
                event,
                record_id=event.event_id,
            )

    def disconnect(self) -> None:
        if self.state is PaperSessionState.DISCONNECTED:
            return
        if self.state is not PaperSessionState.STOPPING:
            self._state_machine.transition(PaperSessionState.STOPPING)
        self.broker.disconnect()
        self._state_machine.transition(PaperSessionState.DISCONNECTED)

    def record_decision_envelope(self, envelope: PaperDecisionEnvelope) -> None:
        if envelope.session_id != self.session_id:
            raise ValueError("decision envelope belongs to another Paper session")
        self.store.append(
            self.session_id,
            "decisions",
            envelope,
            record_id=envelope.envelope_id,
        )

    def record_outcome_envelope(self, envelope: PaperOutcomeEnvelope) -> None:
        if envelope.session_id != self.session_id:
            raise ValueError("outcome envelope belongs to another Paper session")
        self.store.append(
            self.session_id,
            "outcomes",
            envelope,
            record_id=envelope.envelope_id,
        )
