"""Accelerated offline mechanics only. No gateway, wall-clock soak or orders."""
import ast
import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_ai.brokers.config import load_ibkr_paper_config
from trading_ai.brokers.exceptions import BrokerConfigurationError, BrokerIntegrityError
from trading_ai.brokers.fake_soak import FakeReadOnlyBroker
from trading_ai.brokers.models import (
    BrokerAccountIdentity, BrokerEnvironment, BrokerPosition, BrokerConnectionState,
    BrokerOrderRecord, BrokerOrderState, BrokerExecution, PaperMode,
)
from trading_ai.brokers.soak.models import SoakConfig, SoakState, load_soak_config
from trading_ai.brokers.soak.session import PaperReadOnlySession
from trading_ai.brokers.soak.gates import PaperReadOnlyReconciliationGate, Lot10ReadinessGate
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.core.models import OrderSide, OrderType
from trading_ai.monitoring.dashboard import create_dashboard_app
from trading_ai.monitoring.paper import LocalPaperMonitoringReader
from trading_ai.monitoring.soak import soak_view
from trading_ai.cli import main


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def now(self):
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self.seconds)

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


def setup(tmp_path, *, duration=60, scheduled=(), positions=(), identity=None, session_id="soak-test", previous=None):
    clock = Clock()
    identity = identity or BrokerAccountIdentity("FAKE", "a" * 64, "FAKE-****", BrokerEnvironment.PAPER,
                                                "EUR", (), True, "OFFLINE_FIXTURE")
    broker = FakeReadOnlyBroker(clock=clock, session_id=session_id, account=identity,
                               scheduled=scheduled, positions=positions)
    cfg = replace(load_ibkr_paper_config(allow_example=True), allowed_account_hashes=("a" * 64,),
                  request_timeout_seconds=1)
    store = LocalPaperStore(tmp_path / "paper")
    session = PaperReadOnlySession(broker, broker_config=cfg, config=SoakConfig(duration_seconds=duration),
                                  session_id=session_id, code_sha="test-code", store=store,
                                  now=clock.now, monotonic=clock.monotonic, sleep=clock.sleep,
                                  previous_session_id=previous)
    return session, broker, store, clock


def run(session, broker):
    result = session.run()
    assert broker.submit_order_calls == broker.cancel_order_calls == 0
    assert broker.transmission_count == 0
    assert broker.connection_state is BrokerConnectionState.DISCONNECTED
    assert result.paper_execution_armed is False and result.live_hard_locked
    return result


def test_empty_account_full_hour_mechanics_pass(tmp_path):
    session, broker, store, _ = setup(tmp_path, duration=3600)
    report = run(session, broker)
    assert report.gate.status == "PASS"
    assert report.gate.evidence_level == "READ_ONLY_SOAK_PASS"
    assert report.observed_seconds == 3600
    assert report.reconciliation_initial == report.reconciliation_final == "IN_SYNC"
    assert report.snapshot_completeness == 1
    assert report.snapshots_count == 122
    assert report.uptime_seconds == 3600
    payload = store.inspect(session.session_id)
    assert payload["soak_baseline"][0]["ownership"] == "BROKER_BOOTSTRAP_READ_ONLY"
    assert payload["soak_readiness"][0]["review"]["status"] == "INSUFFICIENT_EVIDENCE"
    with pytest.raises(FrozenInstanceError):
        report.reconnects = 20


@pytest.mark.parametrize("duration,status,level", [(60, "INSUFFICIENT_DURATION", "NO_PASS"),
                                                  (900, "PASS", "READ_ONLY_SMOKE_PASS")])
def test_duration_levels_never_conflate_smoke_and_soak(tmp_path, duration, status, level):
    session, broker, _, _ = setup(tmp_path, duration=duration)
    report = run(session, broker)
    assert (report.gate.status, report.gate.evidence_level) == (status, level)


def test_manual_initial_positions_are_external_bootstrap_not_strategy_owned(tmp_path):
    session, broker, store, _ = setup(tmp_path, positions=(BrokerPosition("TEST", Decimal(2), Decimal(50), "EUR"),))
    report = run(session, broker)
    assert report.reconciliation_initial == report.reconciliation_final == "IN_SYNC"
    payload = store.inspect(session.session_id)
    assert payload["soak_baseline"][0]["account"]["positions"][0]["symbol"] == "TEST"
    assert payload["decisions"] == []


@pytest.mark.parametrize("kind", ["cash", "position", "currency", "execution"])
def test_critical_drift_never_absorbed(tmp_path, kind):
    def inject(b):
        if kind == "cash":
            b.cash -= Decimal("1")
        elif kind == "position":
            b._positions["TEST"] = BrokerPosition("TEST", Decimal(1), Decimal(20), "EUR")
        elif kind == "currency":
            b.account = replace(b.account, base_currency="USD")
        else:
            b._executions["e1"] = BrokerExecution("e1", "manual", "external", "1", None, "TEST",
                OrderSide.BUY, Decimal(1), Decimal(10), b.clock.now(), b.clock.now())
    session, broker, _, _ = setup(tmp_path, scheduled=((30, inject),))
    report = run(session, broker)
    assert report.gate.status == "FAIL"
    assert report.critical_drift == 1


def manual_order(clock):
    return BrokerOrderRecord("external-1", "external-1", "manual", "TEST", OrderSide.BUY, OrderType.LIMIT,
        Decimal(1), Decimal(0), BrokerOrderState.ACKNOWLEDGED, "EXTERNAL_BROKER_ACTIVITY",
        clock.now(), clock.now(), limit_price=Decimal(10), broker_order_id="1", external=True)


def test_external_order_activity_is_recorded_and_fail_closed(tmp_path):
    session, broker, store, clock = setup(tmp_path, scheduled=((30, lambda b: b.inject_external_order(manual_order(b.clock))),))
    report = run(session, broker)
    assert report.gate.status == "FAIL" and report.external_activity
    assert "EXTERNAL_BROKER_ACTIVITY" in json.dumps(store.inspect(session.session_id), default=str)


def test_initial_external_order_not_a_false_mismatch(tmp_path):
    session, broker, _, clock = setup(tmp_path)
    broker.inject_external_order(manual_order(clock))
    report = run(session, broker)
    assert report.reconciliation_initial == "IN_SYNC"
    assert not report.external_activity


def test_equity_change_without_trade_is_explicit_warning(tmp_path):
    session, broker, _, _ = setup(tmp_path, scheduled=((30, lambda b: setattr(b, "net_liquidation", b.cash + 2)),))
    report = run(session, broker)
    assert "EQUITY_CHANGED_MARK_TO_MARKET_UNVERIFIED" in report.warnings
    assert report.reconciliation_final == "DRIFT"
    assert report.gate.status != "PASS"


def test_reconnect_requires_full_sync_and_records_gap(tmp_path):
    session, broker, store, _ = setup(tmp_path, scheduled=((30, lambda b: b.disconnect()),))
    report = run(session, broker)
    assert report.reconnects == report.disconnect_count == 1
    assert broker.connect_calls == 2 and broker.sync_calls >= 3
    assert report.reconciliation_final == "IN_SYNC"
    assert report.gate.status == "WARNING"
    assert report.reconnect_duration_seconds >= 10
    events = [e for r in store.inspect(session.session_id)["soak_events"] for e in r["events"]]
    assert any(e["event_type"] == "RECONNECT_COMPLETED" for e in events)


def test_reconnect_exhaustion_fails(tmp_path):
    def disconnect(b):
        b.disconnect()
        b.connect_fails = True
    session, broker, _, _ = setup(tmp_path, scheduled=((30, disconnect),))
    report = run(session, broker)
    assert report.gate.status == "FAIL"
    assert "RECONNECT_EXHAUSTED" in report.gate.reasons
    assert broker.connect_calls == 4


def test_wrong_account_after_reconnect_immediate_fail(tmp_path):
    def change(b):
        b.account = replace(b.account, account_hash="b" * 64)
        b.disconnect()
    session, broker, _, _ = setup(tmp_path, scheduled=((30, change),))
    report = run(session, broker)
    assert report.gate.status == "FAIL" and broker.connect_calls == 2


@pytest.mark.parametrize("environment", [BrokerEnvironment.LIVE, BrokerEnvironment.UNKNOWN])
def test_live_unknown_never_start_observation(tmp_path, environment):
    identity = BrokerAccountIdentity("FAKE", "a" * 64, "MASKED", environment, "EUR", (), False, "TEST")
    session, broker, _, _ = setup(tmp_path, identity=identity)
    report = run(session, broker)
    assert report.gate.status == "FAIL" and broker.sync_calls == 0


@pytest.mark.parametrize("attribute", ["reader_stale", "reader_failed"])
def test_reader_stale_or_failed_never_healthy(tmp_path, attribute):
    session, broker, _, _ = setup(tmp_path, scheduled=((30, lambda b: setattr(b, attribute, True)),))
    report = run(session, broker)
    assert report.gate.status == "FAIL"
    assert report.health_status == "ERROR"


@pytest.mark.parametrize("drift,expected", [(0.2, "INSUFFICIENT_DURATION"), (3, "WARNING"), (6, "FAIL")])
def test_server_clock_thresholds(tmp_path, drift, expected):
    session, broker, _, _ = setup(tmp_path)
    broker.clock_drift = drift
    report = run(session, broker)
    assert report.gate.status == expected


@pytest.mark.parametrize("field,value", [("snapshot_seconds", 1), ("duration_seconds", float("nan")),
    ("duration_seconds", -1), ("reconnect_attempts", 100), ("cash_tolerance", Decimal("NaN")),
    ("smoke_minimum_seconds", 30), ("soak_minimum_seconds", 60)])
def test_config_refuses_invalid_or_misleading_evidence(field, value):
    with pytest.raises(ValueError):
        SoakConfig(**{field: value})


def test_profiles_are_frozen_and_hash_stable():
    for name in ("smoke", "soak"):
        cfg = load_soak_config(f"config/brokers/read_only_{name}.toml")
        assert cfg.config_hash == load_soak_config(f"config/brokers/read_only_{name}.toml").config_hash


def test_mode_refusal_before_connection(tmp_path):
    session, broker, store, _ = setup(tmp_path)
    with pytest.raises(BrokerConfigurationError):
        PaperReadOnlySession(broker, broker_config=replace(session.broker_config, mode=PaperMode.CONNECTIVITY_CHECK),
            config=session.config, session_id="bad-mode", code_sha="test", store=store)
    for kwargs in ({"mode": PaperMode.PAPER_EXECUTION_ARMED}, {"paper_execution_armed": True}):
        with pytest.raises(BrokerConfigurationError):
            replace(session.broker_config, **kwargs)
    assert broker.connect_calls == 0


@pytest.mark.parametrize("bad_id", ["../x", "/absolute", "C:\\x", "bad.id"])
def test_path_rejected_before_connection(tmp_path, bad_id):
    with pytest.raises(BrokerIntegrityError):
        setup(tmp_path, session_id=bad_id)


@pytest.mark.parametrize("tamper", ["corrupt", "extra", "unmanifested"])
def test_tamper_invalidates_gate_in_api(tmp_path, tamper):
    session, broker, store, _ = setup(tmp_path)
    run(session, broker)
    directory = tmp_path / "paper" / session.session_id
    if tamper == "corrupt":
        (directory / "soak_reports" / "final.json").write_text("{}")
    else:
        (directory / ("extra.json" if tamper == "extra" else "hidden.bin")).write_bytes(b"bad")
    with pytest.raises(BrokerIntegrityError):
        store.verify(session.session_id)
    view = soak_view(LocalPaperMonitoringReader(tmp_path / "paper"), session.session_id)
    assert view["gate"]["status"] == "FAIL" and view["report"] is None


def test_restart_links_but_never_merges_duration(tmp_path):
    session, broker, _, _ = setup(tmp_path)
    run(session, broker)
    with pytest.raises(BrokerConfigurationError):
        setup(tmp_path)
    second, b, store, _ = setup(tmp_path, session_id="second", previous=session.session_id)
    report = run(second, b)
    assert report.previous_session_id == session.session_id
    assert report.observed_seconds == 60 and b.sync_calls >= 2
    assert store.inspect("second")["soak_config"][0]["continuity_claimed"] is False


def test_missing_commission_is_unavailable_not_zero(tmp_path):
    session, broker, _, clock = setup(tmp_path)
    broker._executions["e"] = BrokerExecution("e", "external", "external", "1", None,
        "TEST", OrderSide.BUY, Decimal(1), Decimal(10), clock.now(), clock.now())
    report = run(session, broker)
    assert report.gate.status == "FAIL"
    assert "COMMISSIONS_UNAVAILABLE" in report.warnings
    assert session.snapshots[0].commissions[0].amount is None


def test_soak_cli_api_ui_are_read_only(tmp_path, capsys):
    session, broker, _, _ = setup(tmp_path)
    run(session, broker)
    for command in ("read-only-status", "read-only-inspect", "read-only-report"):
        assert main(["paper", command, "--session-id", session.session_id, "--data-root", str(tmp_path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["integrity"] == "VERIFIED"
    assert main(["paper", "read-only-list", "--data-root", str(tmp_path), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    client = TestClient(create_dashboard_app(data_root=tmp_path))
    for route in ("latest", "reconciliation", "report"):
        response = client.get("/api/v1/broker/soak/" + route, params={"session_id": session.session_id})
        assert response.status_code == 200 and response.json()["integrity"] == "VERIFIED"
        assert client.post("/api/v1/broker/soak/" + route).status_code == 405
    assert len(client.get("/api/v1/broker/soak").json()["sessions"]) == 1
    assert "Read-only soak" in client.get("/").text
    assert main(["paper", "read-only-run", "--config", "config/brokers/ibkr_paper.example.toml",
                 "--session-id", "no-network", "--data-root", str(tmp_path), "--json"]) == 2


def test_architecture_session_cannot_reach_execution():
    tree = ast.parse(Path("src/trading_ai/brokers/soak/session.py").read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"submit_approved", "transmit_approved", "place_order", "cancel_order", "fit", "partial_fit"}


def test_gate_never_accepts_execution_or_bad_integrity():
    inputs = dict(config=SoakConfig(), observed_seconds=3600, initial="IN_SYNC", final="IN_SYNC",
                  verified=True, integrity=True, failures=(), warnings=())
    for patch in ({"armed": True}, {"submit_calls": 1}, {"cancel_calls": 1}, {"integrity": False}):
        assert PaperReadOnlyReconciliationGate().evaluate(**(inputs | patch)).status == "FAIL"


def test_lot10_gate_does_not_arm_and_requires_real_evidence(tmp_path):
    session, broker, _, _ = setup(tmp_path)
    report = run(session, broker)
    gate = Lot10ReadinessGate()
    inputs = dict(lot9_done=True, connectivity_pass=True, evidence_integrity=True, real_broker_evidence=True)
    assert gate.evaluate(report, **inputs).status == "INSUFFICIENT_EVIDENCE"
    successful = replace(report, observed_seconds=3600,
                         gate=replace(report.gate, status="PASS", evidence_level="READ_ONLY_SOAK_PASS"))
    ready = gate.evaluate(successful, **inputs)
    assert ready.status == "READY_FOR_HUMAN_REVIEW" and not ready.auto_arms_execution
    assert gate.evaluate(successful, **(inputs | {"evidence_integrity": False})).status == "NOT_READY"
    assert gate.evaluate(successful, **(inputs | {"real_broker_evidence": False})).status == "INSUFFICIENT_EVIDENCE"
    assert gate.evaluate(replace(successful, observed_seconds=60), **inputs).status == "INSUFFICIENT_EVIDENCE"
    assert gate.evaluate(replace(successful, gate=replace(successful.gate, status="WARNING")), **inputs).status == "INSUFFICIENT_EVIDENCE"


def test_clean_interrupt_final_snapshot_disconnect(tmp_path):
    session, broker, store, clock = setup(tmp_path)
    original_sleep = clock.sleep
    def interrupt(seconds):
        original_sleep(seconds)
        if clock.seconds == 20:
            raise KeyboardInterrupt
    session.sleep = interrupt
    report = run(session, broker)
    assert "USER_INTERRUPTED" in report.warnings
    assert report.gate.status == "WARNING" and report.observed_seconds == 20
    assert report.reconciliation_final == "IN_SYNC"
    events = [e for b in store.inspect(session.session_id)["soak_events"] for e in b["events"]]
    stopping = next(i for i, e in enumerate(events) if e["state"] == "STOPPING")
    assert all(e["state"] != "SOAK_RUNNING" for e in events[stopping:])


def test_store_failure_always_disconnects(tmp_path, monkeypatch):
    session, broker, store, _ = setup(tmp_path)
    original = store.append
    def broken(session_id, category, value, **kwargs):
        if category == "soak_snapshots":
            raise BrokerIntegrityError("fixture persistence failure")
        return original(session_id, category, value, **kwargs)
    monkeypatch.setattr(store, "append", broken)
    report = run(session, broker)
    assert report.gate.status == "FAIL"


def test_frozen_config_change_halts(tmp_path):
    session, broker, _, _ = setup(tmp_path)
    broker.scheduled.append((30, lambda b: setattr(session, "config", replace(session.config, snapshot_seconds=60))))
    assert "FROZEN_CONFIG_CHANGED" in run(session, broker).gate.reasons


def test_events_are_mirrored_to_monitoring(tmp_path):
    from trading_ai.monitoring.store import SQLiteMonitoringStore
    session, broker, _, _ = setup(tmp_path)
    monitor = SQLiteMonitoringStore(tmp_path / "monitoring.db")
    session.monitoring_store = monitor
    run(session, broker)
    events = monitor.list_events(session.session_id)
    assert any(e.status == "READ_ONLY_SOAK_COMPLETED" for e in events)
    assert all(e.session_id == session.session_id for e in events)


def test_unavailable_time_never_healthy(tmp_path):
    session, broker, _, _ = setup(tmp_path)
    broker.clock_drift = None
    report = run(session, broker)
    assert report.gate.status == "FAIL" and report.snapshot_completeness == 0


def test_identical_inputs_reports_deterministic(tmp_path):
    a, ba, _, _ = setup(tmp_path / "a")
    b, bb, _, _ = setup(tmp_path / "b")
    assert run(a, ba).report_hash == run(b, bb).report_hash


@pytest.mark.parametrize("quantity_changed", [False, True])
def test_repeated_execution_callback_not_new_activity_but_changed_quantity_is(tmp_path, quantity_changed):
    from trading_ai.brokers.models import BrokerCommissionReport, CommissionKnowledge, BrokerEventType
    session, broker, _, clock = setup(tmp_path)
    broker._executions["e"] = BrokerExecution("e", "external", "external", "1", None,
        "TEST", OrderSide.BUY, Decimal(1), Decimal(10), clock.now(), clock.now())
    broker._commissions["e"] = BrokerCommissionReport("e", CommissionKnowledge.KNOWN, clock.now(), Decimal(1), "EUR")
    broker._emit(BrokerEventType.PARTIAL_FILL, related_ids=(("exec_id", "e"),),
                 payload={"quantity": "1", "is_partial": True, "request_id": 9002})
    broker.scheduled.append((30, lambda b: b._emit(BrokerEventType.FILL, related_ids=(("exec_id", "e"),),
        payload={"quantity": "2" if quantity_changed else "1", "is_partial": False, "request_id": 9004})))
    report = run(session, broker)
    assert report.gate.status == ("FAIL" if quantity_changed else "INSUFFICIENT_DURATION")
