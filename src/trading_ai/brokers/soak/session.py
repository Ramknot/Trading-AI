"""Persistent read-only observer. It has no execution boundary or order API."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from trading_ai.brokers.config import IBKRPaperConfig
from trading_ai.brokers.exceptions import BrokerConfigurationError, BrokerUnavailableError
from trading_ai.brokers.models import (
    BrokerEnvironment, BrokerConnectionState, BrokerAccountSnapshot, BrokerHealth, CommissionKnowledge,
    PaperMode, PaperSessionManifest, ReconciliationStatus,
)
from trading_ai.brokers.soak.models import (
    SoakConfig, SoakState, ReadOnlySnapshot, PaperReadOnlySoakReport,
)
from trading_ai.brokers.soak.reconciliation import reconcile_observations
from trading_ai.brokers.soak.gates import PaperReadOnlyReconciliationGate, Lot10ReadinessGate
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.brokers.reconciliation import ReconciliationState
from trading_ai.core.hashing import stable_hash, to_primitive
from trading_ai.monitoring.models import MonitoringEvent, MonitoringEventType


class ReadOnlyBrokerPort(Protocol):
    """Intentionally excludes every transmit/cancel operation."""
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def sync_state(self) -> ReconciliationState: ...
    def account_snapshot(self) -> BrokerAccountSnapshot: ...
    def health(self) -> BrokerHealth: ...
    def heartbeat(self) -> None: ...
    @property
    def last_server_time_at(self) -> datetime | None: ...


class SoakFailure(Exception):
    """Only stable, non-sensitive reason codes are persisted."""


class PaperReadOnlySession:
    def __init__(self, broker: ReadOnlyBrokerPort, *, broker_config: IBKRPaperConfig,
                 config: SoakConfig, session_id: str, code_sha: str,
                 store: LocalPaperStore, previous_session_id: str | None = None,
                 monitoring_store=None, now=None, monotonic=None, sleep=None):
        if broker_config.mode is not PaperMode.PAPER_READ_ONLY or broker_config.paper_execution_armed:
            raise BrokerConfigurationError("soak requires unarmed PAPER_READ_ONLY")
        if not broker_config.connectable:
            raise BrokerConfigurationError("soak requires an explicit local account allowlist")
        if config.heartbeat_seconds >= broker_config.heartbeat_timeout_seconds:
            raise BrokerConfigurationError("heartbeat cadence must be below stale timeout")
        if config.clock_warning_seconds >= broker_config.max_clock_drift_seconds:
            raise BrokerConfigurationError("clock warning must be below failure threshold")
        # Validate paths and refuse reuse before any connection.
        if store._session(session_id).exists():
            raise BrokerConfigurationError("use a new session ID; continuity cannot be invented")
        if previous_session_id:
            store.verify(previous_session_id)
        self.broker = broker
        self.broker_config = broker_config
        self.config = config
        self._frozen_config_hash = config.config_hash
        self.session_id = session_id
        self.code_sha = code_sha
        self.store = store
        self.previous_session_id = previous_session_id
        self.monitoring_store = monitoring_store
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self.sleep = sleep or time.sleep
        self.state = SoakState.STARTING
        self.snapshots = []
        self.reconciliations = []
        self.failures = set()
        self.warnings = set()
        self.reconnects = self.disconnects = self.stale_events = 0
        self.reconnect_duration = self.max_heartbeat_gap = 0.0
        self._sequence = 0
        self._pending_events = []
        self._event_cursor = 0
        self._callback_errors = 0
        self._seen_activity = set()
        self._verified = False
        self._baseline = None
        self._first_observation = self._last_observation = None
        self._last_heartbeat_mono = None
        self._run_started = False

    def _event(self, kind, **payload):
        self._sequence += 1
        stamp = self.now()
        row = {"event_id": f"soak-{self._sequence:08d}", "session_id": self.session_id,
               "timestamp": stamp, "event_type": kind, "state": self.state.value,
               "source": "paper-read-only-session", "source_version": "1.0", **payload}
        self._pending_events.append(row)
        if self.monitoring_store is not None:
            self.monitoring_store.append_event(MonitoringEvent(
                event_id=self.session_id + "-" + row["event_id"], timestamp=stamp,
                event_type=MonitoringEventType.PAPER_READ_ONLY_SOAK,
                run_id=self.session_id, session_id=self.session_id,
                source_component="paper-read-only-session", component_version="1.0",
                payload_json=json.dumps(to_primitive(row), sort_keys=True), status=kind,
            ))

    def _flush_events(self):
        if self._pending_events:
            self.store.append(self.session_id, "soak_events", {"events": tuple(self._pending_events)},
                              record_id=self._pending_events[-1]["event_id"])
            self._pending_events.clear()

    def _transition(self, state):
        self.state = state
        self._event("SESSION_STATE_CHANGED")

    def _verify_identity(self, account):
        if (account.environment is not BrokerEnvironment.PAPER or not account.environment_verified
            or account.account_hash not in self.broker_config.allowed_account_hashes):
            raise SoakFailure("PAPER_ACCOUNT_VERIFICATION_FAILED")
        if self._baseline and account.account_hash != self._baseline.account.account.account_hash:
            raise SoakFailure("ACCOUNT_CHANGED")
        self._verified = True

    def _verify_connected_identity(self):
        identity = getattr(self.broker, "account_identity", None)
        if identity is None:
            identity = self.broker.account_snapshot().account
        self._verify_identity(identity)

    def _heartbeat(self):
        if self.config.config_hash != self._frozen_config_hash:
            raise SoakFailure("FROZEN_CONFIG_CHANGED")
        started = self.monotonic()
        previous = self.broker.last_server_time_at
        self.broker.heartbeat()
        while True:
            health = self.broker.health()
            if health.critical_errors:
                serious = set(health.critical_errors) - {
                    "EXTERNAL_BROKER_ACTIVITY", "IBKR_CONNECTIVITY_LOST",
                    "IBKR_CONNECTIVITY_RESTORED_DATA_LOST", "IBKR_SOCKET_PORT_RESET",
                    "IBKR_SERVER_CONNECTIVITY_BROKEN", "IBKR_NOT_CONNECTED",
                }
                if serious:
                    self._callback_errors += 1
                    raise SoakFailure("BROKER_CRITICAL_ERROR")
            if (not health.stale and health.connection_state is BrokerConnectionState.CONNECTED
                and self.broker.last_server_time_at is not None and self.broker.last_server_time_at != previous):
                break
            if self.monotonic() - started >= self.broker_config.request_timeout_seconds:
                self.stale_events += 1
                self._transition(SoakState.DEGRADED)
                self._event("SESSION_STALE")
                raise BrokerUnavailableError("heartbeat unavailable")
            self.sleep(min(0.2, self.broker_config.request_timeout_seconds))
        instant = self.monotonic()
        if self._last_heartbeat_mono is not None:
            self.max_heartbeat_gap = max(self.max_heartbeat_gap, instant - self._last_heartbeat_mono)
        self._last_heartbeat_mono = instant
        if health.clock_drift_seconds is None:
            self.warnings.add("SERVER_TIME_UNAVAILABLE")
        elif health.clock_drift_seconds > self.broker_config.max_clock_drift_seconds:
            raise SoakFailure("BROKER_CLOCK_DRIFT")
        elif health.clock_drift_seconds > self.config.clock_warning_seconds:
            self.warnings.add("CLOCK_DRIFT_WARNING")
        self._event("HEARTBEAT_OK", health=to_primitive(health))

    def _capture(self, *, final=False):
        if getattr(self.broker, "config", self.broker_config) != self.broker_config:
            raise SoakFailure("FROZEN_CONFIG_CHANGED")
        if not final:
            self._transition(SoakState.RECONCILING)
        begin = self.monotonic()
        state = self.broker.sync_state()  # Must wait for all five completion callbacks.
        account = self.broker.account_snapshot()
        self._verify_identity(account.account)
        self._heartbeat()
        health = self.broker.health()
        commissions = tuple(getattr(self.broker, "commission_reports", ()))
        missing = []
        if account.account.base_currency in {"", "UNKNOWN", "BASE"}:
            missing.append("BASE_CURRENCY_UNAVAILABLE")
        known_commissions = {x.exec_id for x in commissions if x.status is CommissionKnowledge.KNOWN}
        if {x.exec_id for x in state.executions} - known_commissions:
            missing.append("COMMISSIONS_UNAVAILABLE")
        if health.clock_drift_seconds is None:
            missing.append("SERVER_TIME_UNAVAILABLE")
        snapshot = ReadOnlySnapshot(
            self.session_id, len(self.snapshots), self.now(), account, state, commissions,
            health, str(getattr(self.broker, "version", "UNAVAILABLE")),
            getattr(self.broker, "sdk_version", None), getattr(self.broker, "server_version", None),
            self.monotonic() - begin, tuple(missing),
        )
        if self._baseline is None:
            self._baseline = snapshot
            self.store.append(self.session_id, "soak_baseline", snapshot, record_id="bootstrap")
        reconciliation = reconcile_observations(self._baseline, snapshot, self.config)
        events = tuple(getattr(self.broker, "broker_events", ()))[self._event_cursor:]
        self._event_cursor += len(events)
        self._callback_errors += sum(x.event_type.value == "ERROR" for x in events)
        # A transient external event cannot be hidden by returning to the baseline before a poll.
        activity = [x for x in events if x.event_type.value in {
            "EXTERNAL_BROKER_ACTIVITY", "FILL", "PARTIAL_FILL", "EXECUTION_CORRECTION"
        }]
        # Refreshing the same execution after later partial fills must not turn
        # historical callbacks into new economic activity. Numeric changes and
        # new/corrected exec IDs still produce a different fingerprint.
        activity_keys = {stable_hash((
            "EXECUTION" if dict(x.related_ids).get("exec_id") else x.event_type.value,
            x.related_ids,
            {k: v for k, v in x.payload.items() if k not in {"request_id", "is_partial", "correction_of"}},
        )) for x in activity}
        if self.snapshots and activity_keys - self._seen_activity:
            reconciliation = replace(reconciliation, status=ReconciliationStatus.CRITICAL_DRIFT,
                external_activity=True, reasons=tuple(sorted(set(reconciliation.reasons) | {"EXTERNAL_BROKER_ACTIVITY"})))
        self._seen_activity.update(activity_keys)
        self.store.append(self.session_id, "soak_snapshots", {
            "snapshot_id": snapshot.snapshot_id, "snapshot": snapshot,
            "reconciliation": reconciliation, "broker_events": events,
            "progress": {
                "observed_seconds": (self.monotonic() - self._first_observation
                                     if self._first_observation is not None else 0.0),
                "uptime_seconds": max(0.0, self.monotonic() - self._first_observation - self.reconnect_duration)
                                  if self._first_observation is not None else 0.0,
                "reconnects": self.reconnects,
                "disconnect_count": self.disconnects,
                "snapshots_count": len(self.snapshots) + 1,
                "drift_events": sum(r.status is not ReconciliationStatus.IN_SYNC
                                    for r in (*self.reconciliations, reconciliation)),
            },
        }, record_id=f"snapshot-{snapshot.sequence:08d}")
        self.snapshots.append(snapshot)
        self.reconciliations.append(reconciliation)
        if self._first_observation is None:
            self._first_observation = self.monotonic()
        self._last_observation = self.monotonic()
        self._event("SNAPSHOT_CAPTURED", snapshot_id=snapshot.snapshot_id)
        self.warnings.update(missing)
        if reconciliation.external_activity:
            self._event("EXTERNAL_BROKER_ACTIVITY", reasons=reconciliation.reasons)
        if reconciliation.status is ReconciliationStatus.CRITICAL_DRIFT:
            self._event("DRIFT_DETECTED", reconciliation=reconciliation)
            raise SoakFailure("CRITICAL_RECONCILIATION_DRIFT")
        if reconciliation.status is not ReconciliationStatus.IN_SYNC:
            self.warnings.update(reconciliation.reasons or ("SNAPSHOT_INCOMPLETE",))
            if not final:
                self._transition(SoakState.DEGRADED)
            self._event("DRIFT_DETECTED", reconciliation=reconciliation)
            if len(self.snapshots) == 1:
                raise SoakFailure("INITIAL_RECONCILIATION_NOT_IN_SYNC")
        else:
            self._event("RECONCILIATION_IN_SYNC", snapshot_id=snapshot.snapshot_id)
            if not final:
                self._transition(SoakState.SOAK_RUNNING)
        self._flush_events()

    def _recover(self):
        self.disconnects += 1
        self._transition(SoakState.DEGRADED)
        self.warnings.add("CONNECTION_INTERRUPTED")
        begin = self.monotonic()
        for attempt in range(self.config.reconnect_attempts):
            self._event("RECONNECT_STARTED", attempt=attempt + 1)
            try:
                self.broker.disconnect()
                self.sleep(self.config.reconnect_delay_seconds)
                self.broker.connect()
                self._verify_connected_identity()
                self._transition(SoakState.PAPER_VERIFIED)
                self._capture()
                if self.reconciliations[-1].status is not ReconciliationStatus.IN_SYNC:
                    raise SoakFailure("RECONNECT_RECONCILIATION_FAILED")
                self.reconnects += 1
                self.reconnect_duration += self.monotonic() - begin
                self._event("RECONNECT_COMPLETED")
                return
            except BrokerUnavailableError:
                continue
        self.reconnect_duration += self.monotonic() - begin
        raise SoakFailure("RECONNECT_EXHAUSTED")

    def run(self) -> PaperReadOnlySoakReport:
        try:
            return self._observe()
        finally:
            # Even an evidence-store failure must close the socket. Do not let
            # reporting exceptions bypass shutdown.
            if getattr(self.broker, "connection_state", None) is not BrokerConnectionState.DISCONNECTED:
                self.broker.disconnect()

    def _observe(self) -> PaperReadOnlySoakReport:
        if self._run_started:
            raise BrokerConfigurationError("a soak session can run only once")
        self._run_started = True
        start, began = self.now(), self.monotonic()
        # Startup failures also get an auditable manifest; identity is only trusted in verified snapshots.
        self.store.create_session(PaperSessionManifest(
            self.session_id, start, self.code_sha, PaperMode.PAPER_READ_ONLY,
            str(getattr(self.broker, "name", "UNAVAILABLE")), str(getattr(self.broker, "version", "UNAVAILABLE")),
            getattr(self.broker, "sdk_version", None), getattr(self.broker, "server_version", None),
            stable_hash("UNVERIFIED"), "UNAVAILABLE_UNTIL_VERIFIED",
            tuple(sorted((("broker", self.broker_config.config_hash), ("soak", self.config.config_hash)))), (),
        ))
        self.store.append(self.session_id, "soak_config", {
            "config": self.config, "previous_session_id": self.previous_session_id,
            "continuity_claimed": False,
        }, record_id="frozen")
        self._event("READ_ONLY_SOAK_STARTED")
        try:
            self._transition(SoakState.CONNECTING)
            self.broker.connect()
            self._verify_connected_identity()
            self._transition(SoakState.PAPER_VERIFIED)
            self._capture()
            # Requested observation duration begins after complete bootstrap, not before connection.
            deadline = self.monotonic() + self.config.duration_seconds
            next_snapshot = self.monotonic() + self.config.snapshot_seconds
            while self.monotonic() < deadline:
                self.sleep(min(self.config.heartbeat_seconds, deadline - self.monotonic()))
                try:
                    self._heartbeat()
                    if self.monotonic() >= next_snapshot:
                        self._capture()
                        next_snapshot = self.monotonic() + self.config.snapshot_seconds
                except BrokerUnavailableError:
                    self._recover()
                    next_snapshot = self.monotonic() + self.config.snapshot_seconds
        except KeyboardInterrupt:
            self.warnings.add("USER_INTERRUPTED")
        except SoakFailure as exc:
            self.failures.add(str(exc))
            self._transition(SoakState.HALTED)
        except Exception:
            # Never persist arbitrary SDK exception text or account identifiers.
            self.failures.add("READ_ONLY_OBSERVATION_FAILED")
            self._transition(SoakState.HALTED)
        finally:
            self._transition(SoakState.STOPPING)
            if self._baseline is not None and not self.failures:
                try:
                    self._capture(final=True)
                except Exception:
                    self.failures.add("FINAL_SNAPSHOT_FAILED")
            try:
                self.broker.disconnect()
            except Exception:
                self.failures.add("DISCONNECT_FAILED")
            self._transition(SoakState.FAILED if self.failures else SoakState.COMPLETED)
        end = self.now()
        elapsed = self.monotonic() - began
        observed = (self._last_observation - self._first_observation
                    if self._first_observation is not None else 0.0)
        initial = self.reconciliations[0].status.value if self.reconciliations else "UNKNOWN"
        final = self.reconciliations[-1].status.value if self.reconciliations else "UNKNOWN"
        self._flush_events()
        self.store.verify(self.session_id)  # Corruption cannot be blessed by appending a PASS.
        gate = PaperReadOnlyReconciliationGate().evaluate(
            config=self.config, observed_seconds=observed, initial=initial, final=final,
            verified=self._verified, integrity=True, failures=tuple(sorted(self.failures)),
            warnings=tuple(sorted(self.warnings)),
        )
        drifts = [s.health.clock_drift_seconds for s in self.snapshots if s.health.clock_drift_seconds is not None]
        report = PaperReadOnlySoakReport(
            self.session_id, start, end, elapsed, observed, max(0.0, observed - self.reconnect_duration),
            len(self.snapshots), self.reconnects, self.disconnects, self.reconnect_duration,
            self.stale_events, self.max_heartbeat_gap,
            max((s.latency_seconds for s in self.snapshots), default=0.0), self._callback_errors,
            sum(r.status is not ReconciliationStatus.IN_SYNC for r in self.reconciliations),
            sum(r.status is ReconciliationStatus.CRITICAL_DRIFT for r in self.reconciliations),
            sum(r.external_activity for r in self.reconciliations), max(drifts) if drifts else None,
            sum(not s.missing for s in self.snapshots) / len(self.snapshots) if self.snapshots else 0.0,
            "ERROR" if gate.status == "FAIL" else "WARNING" if self.warnings else "HEALTHY",
            initial, final, self._verified, "VERIFIED", self.state, gate, self.config.config_hash,
            self.previous_session_id, tuple(sorted(self.warnings)),
        )
        self.store.append(self.session_id, "soak_reports", report, record_id="final")
        readiness = Lot10ReadinessGate().evaluate(
            report, lot9_done=True, connectivity_pass=self._verified,
            evidence_integrity=True,
            real_broker_evidence=getattr(self.broker, "name", "") == "ibkr-tws-paper",
        )
        self.store.append(self.session_id, "soak_readiness", {
            "report_hash": report.report_hash, "review": readiness,
        }, record_id="lot10")
        self._event("READ_ONLY_SOAK_COMPLETED", report_hash=report.report_hash, gate=gate)
        self._flush_events()
        return report
