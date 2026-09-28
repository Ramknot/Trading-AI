"""Render soak evidence without importing brokers or recalculating gates."""
from datetime import datetime, timezone

from trading_ai.monitoring.exceptions import MonitoringIntegrityError


def soak_view(reader, session_id):
    try:
        payload = reader.inspect(session_id)
    except MonitoringIntegrityError:
        return {"session_id": session_id, "integrity": "ERROR", "gate": {"status": "FAIL"},
                "status": "ERROR", "report": None}
    snapshots = payload.get("soak_snapshots", [])
    events = [event for batch in payload.get("soak_events", []) for event in batch["events"]]
    reports = payload.get("soak_reports", [])
    readiness = payload.get("soak_readiness", [])
    latest = snapshots[-1] if snapshots else None
    report = reports[-1] if reports else None
    last_event = events[-1] if events else None
    samples = [s for batch in payload.get("soak_clock_samples", []) for s in batch["samples"]]
    clock_events = [e for e in events if e["event_type"] == "CLOCK_SAMPLE_RECORDED"]
    clock_source = report if report and report.get("clock_sample_count") is not None else (
        clock_events[-1]["clock"] if clock_events else {})
    clock = {k: v for k, v in clock_source.items() if k.startswith("clock_")}
    clock["status"] = "AVAILABLE" if samples else "UNAVAILABLE"
    clock["limitation"] = (None if samples else
        "No persisted heartbeat clock samples. Historical gate is unchanged; snapshot maxima cannot reconstruct heartbeat peaks.")
    freshness = "UNAVAILABLE"
    if last_event:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(last_event["timestamp"])).total_seconds()
        freshness = "COMPLETED" if report else "STALE" if age > 120 else "RECENT"
    return {"session_id": session_id, "integrity": "VERIFIED",
            "status": report["state"] if report else last_event["state"] if last_event else "UNAVAILABLE",
            "observation_freshness": freshness, "latest": latest,
            "reconciliation": latest["reconciliation"] if latest else None,
            "report": report, "gate": report["gate"] if report else {"status": "UNAVAILABLE"},
            "lot10_readiness": readiness[-1] if readiness else {"status": "INSUFFICIENT_EVIDENCE"},
            "clock": clock, "clock_samples": samples,
            "events": events, "paper_execution_armed": False, "live_hard_locked": True}
