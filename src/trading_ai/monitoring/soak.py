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
            "events": events, "paper_execution_armed": False, "live_hard_locked": True}
