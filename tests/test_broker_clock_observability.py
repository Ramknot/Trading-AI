"""Clock mechanics, never real TWS connectivity or trading evidence."""
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from trading_ai.brokers.clock import HeartbeatClockSample, ClockRequestTracker, clock_summary
from trading_ai.brokers.exceptions import BrokerUnavailableError
from trading_ai.brokers.soak.gates import PaperReadOnlyReconciliationGate
from trading_ai.brokers.soak.models import SoakConfig
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.cli import main
from trading_ai.monitoring.dashboard import create_dashboard_app
from trading_ai.monitoring.paper import LocalPaperMonitoringReader
from trading_ai.monitoring.soak import soak_view
from test_paper_read_only_soak import Clock, setup, run


EPOCH = 1767225600
BASE = datetime.fromtimestamp(EPOCH, timezone.utc)


def sample(*, offset=2.4, rtt=.4, sequence=1):
    return HeartbeatClockSample("fixture", str(sequence), BASE + timedelta(seconds=offset-rtt),
        BASE + timedelta(seconds=offset), 10, 10+rtt, EPOCH)


def persisted_samples(store, session_id):
    return [s for batch in store.inspect(session_id)["soak_clock_samples"] for s in batch["samples"]]


def test_rtt_midpoint_resolution_uncertainty_and_immutability():
    s = sample()
    assert s.round_trip_ms == pytest.approx(400)
    assert s.raw_server_time_offset_seconds == s.raw_offset_seconds == 2.4
    assert s.midpoint_offset_estimate_seconds == 2.2
    assert s.offset_uncertainty_seconds == pytest.approx(1.2)
    assert s.certain_clock_offset_seconds == pytest.approx(1.0)
    assert s.server_timestamp_resolution_seconds == 1
    assert s.source == "IBKR_CURRENT_TIME" and s.server_time_utc == BASE
    assert s.sample_id == sample().sample_id
    with pytest.raises(FrozenInstanceError):
        s.server_epoch = 0


@pytest.mark.parametrize("end", [9, float("inf"), float("nan"), 1e308])
def test_invalid_monotonic_rtt_fails(end):
    with pytest.raises(ValueError, match="CLOCK_RTT_INVALID"):
        replace(sample(), receive_monotonic=end)


def test_wall_clock_jump_never_changes_monotonic_rtt():
    s = replace(sample(), received_at_utc=BASE + timedelta(seconds=102.4))
    assert s.round_trip_ms == pytest.approx(400)
    assert s.wall_clock_step_seconds == pytest.approx(100)
    assert s.offset_uncertainty_seconds == pytest.approx(51.2)
    assert s.raw_server_time_offset_seconds == 102.4


@pytest.mark.parametrize("kw", [{"server_epoch": float(EPOCH)}, {"server_timestamp_resolution_seconds": .001},
                               {"requested_at_utc": BASE.replace(tzinfo=None)}])
def test_no_fictitious_precision_or_naive_timestamp(kw):
    with pytest.raises(ValueError):
        replace(sample(), **kw)


def tracker():
    clock = Clock()
    return ClockRequestTracker("fixture", timeout_seconds=1, now=clock.now, monotonic=clock.monotonic), clock


def test_concurrent_requests_poison_channel_no_misattributed_sample():
    t, _ = tracker()
    t.begin()
    with pytest.raises(BrokerUnavailableError):
        t.begin()
    assert t.receive(EPOCH) is None and t.latest is None and t.error


def test_duplicate_response_no_second_sample_and_pending_duplicate_is_ambiguous():
    t, c = tracker()
    rid = t.begin()
    first = t.receive(EPOCH)
    assert first.request_id == rid
    assert t.receive(EPOCH) is None and t.latest is first and t.error is None
    c.sleep(10)
    t.begin()
    assert t.receive(EPOCH) is None and t.error == "CLOCK_RESPONSE_AMBIGUOUS"
    assert t.receive(EPOCH+10) is None and t.latest is first


def test_late_response_after_timeout_cannot_be_reassigned():
    t, c = tracker()
    t.begin()
    c.sleep(2)
    assert t.receive(EPOCH+2) is None and t.error == "CLOCK_RESPONSE_EXPIRED"
    with pytest.raises(BrokerUnavailableError):
        t.begin()
    assert t.receive(EPOCH+3) is None
    t.reset_connection()
    new_id = t.begin()
    assert t.receive(EPOCH+2).request_id == new_id


@pytest.mark.parametrize("epoch", [None, "not-epoch", float("nan")])
def test_invalid_callback_never_fabricates_measurement(epoch):
    t, _ = tracker()
    t.begin()
    assert t.receive(epoch) is None and t.error == "CLOCK_MEASUREMENT_INVALID"


def test_unsolicited_current_time_is_explicitly_unknown():
    t, _ = tracker()
    assert t.receive(EPOCH) is None and t.error == "CLOCK_RESPONSE_UNSOLICITED"


@pytest.mark.parametrize("peak_offset,warning", [(2.4, False), (3.4, True)])
def test_transient_heartbeat_vs_snapshot_regression(tmp_path, peak_offset, warning):
    session, broker, store, _ = setup(tmp_path, scheduled=(
        (10, lambda b: setattr(b, "clock_drift", peak_offset)),
        (20, lambda b: setattr(b, "clock_drift", 1.8))))
    broker.clock_drift = 1.8
    report = run(session, broker)
    assert max(s.health.clock_drift_seconds for s in session.snapshots) == 1.8
    assert report.clock_drift_max_seconds == report.clock_raw_offset_max_seconds == peak_offset
    assert report.clock_certain_offset_max_seconds == pytest.approx(peak_offset - 1)
    assert report.clock_certain_offset_current_seconds == pytest.approx(.8)
    assert report.gate.status == ("WARNING" if warning else "INSUFFICIENT_DURATION")
    assert report.clock_warning_count == int(warning)
    samples = persisted_samples(store, session.session_id)
    if warning:
        peak = next(s for s in samples if s["sample_id"] == report.clock_warning_max_sample_id)
        assert peak["certain_clock_offset_seconds"] == report.clock_warning_max_value_seconds == pytest.approx(peak_offset - 1)
        assert peak["received_at_utc"] == report.clock_warning_first_at.isoformat()
        assert report.clock_warning_first_at == report.clock_warning_last_at
    assert len(samples) == report.clock_sample_count > len(session.snapshots)
    assert report.clock_gate_metric == "certain_clock_offset_seconds"
    assert report.clock_warning_threshold_seconds == 2 and report.clock_hard_threshold_seconds == 5
    with pytest.raises(ValueError, match="same evaluated samples"):
        replace(report, clock_drift_max_seconds=1.8)
    if warning:
        with pytest.raises(ValueError, match="provenance"):
            replace(report, clock_warning_max_sample_id=None)


@pytest.mark.parametrize("offset,warning,hard", [(2.09, False, False), (3, False, False),
    (3.4, True, False), (6, True, False), (6.1, True, True)])
def test_thresholds_unchanged_on_certain_metric_persisted_before_snapshot(tmp_path, offset, warning, hard):
    session, broker, store, _ = setup(tmp_path, duration=10)
    broker.clock_drift = offset
    report = run(session, broker)
    assert report.clock_raw_offset_max_seconds == offset
    assert report.clock_certain_offset_max_seconds == pytest.approx(offset - 1)
    assert bool(report.clock_warning_count) == warning
    assert ("CLOCK_DRIFT_WARNING" in report.warnings) == warning
    samples = persisted_samples(store, session.session_id)
    assert samples and samples[0]["raw_server_time_offset_seconds"] == offset
    if hard:
        assert "BROKER_CLOCK_DRIFT" in report.gate.reasons and report.gate.status == "FAIL"
        assert report.clock_hard_failure_sample_id == samples[0]["sample_id"]
        assert report.snapshots_count == 0
    elif warning:
        assert report.gate.status == "WARNING"


def test_percentiles_are_nearest_rank_and_diagnostic_only():
    values = tuple(sample(offset=i/10, rtt=i/100, sequence=i) for i in range(1, 21))
    summary = clock_summary(values, 2, 5)
    assert summary["clock_raw_offset_p95_seconds"] == 1.9
    assert summary["clock_rtt_p95_ms"] == pytest.approx(190)
    assert summary["clock_certain_offset_p95_seconds"] == pytest.approx(.71)
    assert summary["clock_warning_count"] == 0
    assert clock_summary((), 2, 5)["clock_raw_offset_max_seconds"] is None


def test_gate_refuses_warning_without_sample_provenance():
    result = PaperReadOnlyReconciliationGate().evaluate(config=SoakConfig(), observed_seconds=3600,
        initial="IN_SYNC", final="IN_SYNC", verified=True, integrity=True, failures=(),
        warnings=("CLOCK_DRIFT_WARNING",))
    assert result.status == "FAIL" and "CLOCK_PROVENANCE_MISSING" in result.reasons


def test_fake_jitter_quantized_epoch_keeps_all_metrics_traceable(tmp_path):
    session, broker, store, _ = setup(tmp_path, scheduled=((10, lambda b: setattr(b, "clock_rtt_seconds", .8)),))
    broker.server_clock_offset_seconds = .3
    broker.clock_rtt_seconds = .2
    report = run(session, broker)
    samples = persisted_samples(store, session.session_id)
    assert report.clock_rtt_max_ms == pytest.approx(800)
    assert report.clock_raw_offset_max_seconds == max(s["raw_server_time_offset_seconds"] for s in samples)
    for s in samples:
        assert type(s["server_epoch"]) is int and s["server_timestamp_resolution_seconds"] == 1
        assert s["round_trip_ms"] == pytest.approx((s["receive_monotonic"] - s["request_monotonic"]) * 1000)
        assert s["offset_uncertainty_seconds"] >= 1


def test_cli_api_show_persisted_clock_metrics_read_only(tmp_path, capsys):
    session, broker, _, _ = setup(tmp_path, duration=10)
    run(session, broker)
    assert main(["paper", "read-only-report", "--session-id", session.session_id,
                 "--data-root", str(tmp_path), "--json"]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view["clock"]["status"] == "AVAILABLE" and view["clock_samples"]
    client = TestClient(create_dashboard_app(data_root=tmp_path))
    result = client.get("/api/v1/broker/soak/report", params={"session_id": session.session_id})
    assert result.status_code == 200
    assert result.json()["clock"]["clock_warning_threshold_seconds"] == 2
    assert client.post("/api/v1/broker/soak/report").status_code == 405


@pytest.mark.parametrize("legacy_samples", [False, True])
def test_legacy_report_inspection_does_not_invent_or_rewrite_samples(tmp_path, legacy_samples):
    session, broker, store, _ = setup(tmp_path / "new", duration=10)
    run(session, broker)
    payload = store.inspect(session.session_id)
    legacy = LocalPaperStore(tmp_path / "legacy" / "paper")
    # Construct an independent historical-format fixture, never edit a real bundle.
    from trading_ai.brokers.models import PaperSessionManifest
    manifest = dict(payload["session"])
    manifest["created_at"] = datetime.fromisoformat(manifest["created_at"])
    manifest["config_hashes"] = tuple(tuple(x) for x in manifest["config_hashes"])
    manifest["ml_model_ids"] = tuple(manifest["ml_model_ids"])
    legacy.create_session(PaperSessionManifest(**manifest))
    old_clock = {}
    if legacy_samples:
        from trading_ai.core.hashing import to_primitive
        old_sample = to_primitive(sample(offset=2.09, rtt=0))
        del old_sample["certain_clock_offset_seconds"]
        legacy.append(session.session_id, "soak_clock_samples", {
            "schema_version": "1.1", "samples": [old_sample]}, record_id="old")
        old_clock = {"clock_sample_count": 1, "clock_gate_metric": "raw_server_time_offset_seconds"}
    legacy.append(session.session_id, "soak_reports", {
        "state": "COMPLETED", "clock_drift_max_seconds": 1.879262,
        "gate": {"status": "WARNING", "evidence_level": "NO_PASS", "reasons": ["CLOCK_DRIFT_WARNING"]},
        **old_clock,
    }, record_id="final")
    directory = tmp_path / "legacy" / "paper" / session.session_id
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*.json")}
    view = soak_view(LocalPaperMonitoringReader(tmp_path / "legacy" / "paper"), session.session_id)
    assert view["clock"]["status"] == ("AVAILABLE" if legacy_samples else "UNAVAILABLE")
    assert all("certain_clock_offset_seconds" not in s for s in view["clock_samples"])
    assert "clock_warning_count" not in view["clock"]
    assert "clock_certain_offset_max_seconds" not in view["clock"]
    assert view["gate"]["status"] == "WARNING" and view["report"]["clock_drift_max_seconds"] == 1.879262
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*.json")}


def test_raw_peak_and_gate_peak_are_distinct_with_exact_provenance():
    raw_peak = sample(offset=10, rtt=8, sequence=1)  # midpoint=6, uncertainty=5
    gate_peak = sample(offset=7, rtt=0, sequence=2)  # certain=6
    summary = clock_summary((raw_peak, sample(offset=6.5, rtt=0, sequence=3), gate_peak), 2, 5)
    assert summary["clock_raw_offset_max_seconds"] == 10
    assert raw_peak.certain_clock_offset_seconds == 1
    assert summary["clock_certain_offset_max_seconds"] == 6
    assert summary["clock_warning_max_value_seconds"] == 6
    assert summary["clock_peak_sample_id"] == summary["clock_warning_max_sample_id"] == gate_peak.sample_id
    assert summary["clock_hard_failure_sample_id"] == gate_peak.sample_id


def test_high_raw_uncertainty_read_only_gate_does_not_change_adapter_health(tmp_path):
    session, broker, store, _ = setup(tmp_path, duration=10)
    broker.clock_drift, broker.clock_rtt_seconds = 8, 8
    original_health = broker.health
    broker.health = lambda: replace(original_health(), critical_errors=("BROKER_CLOCK_DRIFT",))
    report = run(session, broker)
    assert report.clock_raw_offset_max_seconds == 8
    assert report.clock_certain_offset_max_seconds == 0
    assert report.clock_warning_count == report.clock_hard_failure_count == 0
    assert "CLOCK_DRIFT_WARNING" not in report.warnings and report.gate.status != "FAIL"
    assert all(s["offset_uncertainty_seconds"] == 5 for s in persisted_samples(store, session.session_id))
    assert session.snapshots[0].health.critical_errors == ("BROKER_CLOCK_DRIFT",)


def test_legacy_raw_alarm_without_matched_sample_still_fails_closed(tmp_path):
    from trading_ai.brokers.soak.session import SoakFailure
    session, broker, _, _ = setup(tmp_path)
    with pytest.raises(SoakFailure, match="BROKER_CRITICAL_ERROR"):
        session._check_callback_health(replace(broker.health(), critical_errors=("BROKER_CLOCK_DRIFT",)))


def test_clock_batch_tampering_fails_integrity(tmp_path):
    session, broker, _, _ = setup(tmp_path, duration=10)
    run(session, broker)
    artifact = next((tmp_path / "paper" / session.session_id / "soak_clock_samples").glob("*.json"))
    artifact.write_text("{}", encoding="utf-8")
    view = soak_view(LocalPaperMonitoringReader(tmp_path / "paper"), session.session_id)
    assert view["integrity"] == "ERROR" and view["gate"]["status"] == "FAIL"


def test_unfinished_session_clock_view_uses_recorded_event_not_snapshot(tmp_path):
    session, broker, store, _ = setup(tmp_path / "new", duration=10)
    run(session, broker)
    payload = store.inspect(session.session_id)
    from trading_ai.brokers.models import PaperSessionManifest
    manifest = dict(payload["session"])
    manifest["created_at"] = datetime.fromisoformat(manifest["created_at"])
    manifest["config_hashes"] = tuple(tuple(x) for x in manifest["config_hashes"])
    manifest["ml_model_ids"] = tuple(manifest["ml_model_ids"])
    incomplete = LocalPaperStore(tmp_path / "unfinished" / "paper")
    incomplete.create_session(PaperSessionManifest(**manifest))
    for category in ("soak_clock_samples", "soak_events"):
        for i, batch in enumerate(payload[category]):
            incomplete.append(session.session_id, category, batch, record_id=str(i))
    view = soak_view(LocalPaperMonitoringReader(incomplete.root), session.session_id)
    assert view["report"] is None
    assert view["clock"]["clock_sample_count"] == len(view["clock_samples"])
    assert view["clock"]["clock_raw_offset_max_seconds"] == session._clock_summary()["clock_raw_offset_max_seconds"]


def test_adapter_single_flight_request_and_duplicate_callback_without_socket():
    from trading_ai.brokers.config import load_ibkr_paper_config
    from trading_ai.brokers.ibkr.adapter import IBKRPaperAdapter
    from trading_ai.brokers.ibkr.contracts import IBKRContractResolver, load_contract_specs
    from test_ibkr_adapter_infrastructure import StubIBKRClient
    config = load_ibkr_paper_config(allow_example=True)
    client = StubIBKRClient()
    adapter = IBKRPaperAdapter(config, IBKRContractResolver(load_contract_specs(config.contract_config)),
                              session_id="clock-adapter", client=client)
    assert not client.connected
    rid = adapter.heartbeat()
    epoch = int(datetime.now(timezone.utc).timestamp())
    adapter._on_callback("CURRENT_TIME", {"epoch": epoch})
    first = adapter.latest_clock_sample
    assert first.request_id == rid and first.round_trip_ms >= 0
    adapter._on_callback("CURRENT_TIME", {"epoch": epoch})
    assert adapter.latest_clock_sample is first and adapter.clock_measurement_error is None
    adapter.heartbeat()
    adapter._on_callback("CURRENT_TIME", {"epoch": epoch})
    assert adapter.clock_measurement_error == "CLOCK_RESPONSE_AMBIGUOUS"
    assert adapter.health().clock_drift_seconds is None
    assert client.placed == [] and not client.connected


def test_monitoring_store_carries_produced_clock_metrics(tmp_path):
    from trading_ai.monitoring.store import SQLiteMonitoringStore
    session, broker, _, _ = setup(tmp_path, duration=10)
    session.monitoring_store = SQLiteMonitoringStore(tmp_path / "monitoring.db")
    report = run(session, broker)
    events = session.monitoring_store.list_events(session.session_id, status="CLOCK_SAMPLE_RECORDED")
    assert len(events) == report.clock_sample_count
    recorded = json.loads(events[-1].payload_json)
    assert recorded["clock"]["clock_raw_offset_max_seconds"] == report.clock_raw_offset_max_seconds
    assert recorded["clock"]["clock_rtt_max_ms"] == report.clock_rtt_max_ms
    assert recorded["clock"]["clock_warning_count"] == report.clock_warning_count


def test_reusing_recent_sample_does_not_hide_reader_failure(tmp_path):
    session, broker, store, _ = setup(tmp_path)
    original_sync = broker.sync_state
    def fail_reader_after_sync():
        state = original_sync()
        if broker.sync_calls == 2:
            broker.reader_failed = True
        return state
    broker.sync_state = fail_reader_after_sync
    report = run(session, broker)
    assert report.gate.status == "FAIL" and "BROKER_CRITICAL_ERROR" in report.gate.reasons
    assert report.broker_callback_error_count == 1
    assert persisted_samples(store, session.session_id)


def test_late_ambiguous_callback_during_shutdown_is_not_hidden(tmp_path):
    session, broker, store, _ = setup(tmp_path, duration=10)
    original_disconnect = broker.disconnect
    def disconnect_with_late_reply():
        original_disconnect()
        broker._clock_tracker.receive(EPOCH + 99)
    broker.disconnect = disconnect_with_late_reply
    report = run(session, broker)
    assert report.gate.status == "WARNING"
    assert "CLOCK_MEASUREMENT_UNKNOWN" in report.warnings
    events = [e for batch in store.inspect(session.session_id)["soak_events"] for e in batch["events"]]
    assert any(e["event_type"] == "CLOCK_MEASUREMENT_DISCARDED" and e.get("phase") == "SHUTDOWN" for e in events)
