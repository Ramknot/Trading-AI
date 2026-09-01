"""Broker-neutral, read-only access to local Paper observability evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from trading_ai.core.hashing import stable_hash
from trading_ai.monitoring.base import EventMonitoringSource
from trading_ai.monitoring.exceptions import MonitoringIntegrityError
from trading_ai.monitoring.models import MonitoringEvent, MonitoringEventType
from trading_ai.monitoring.security import redact_sensitive


_PAPER_STORE_SCHEMA_VERSION = "1.0"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_CATEGORIES = (
    "events",
    "orders",
    "executions",
    "commissions",
    "snapshots",
    "reconciliation",
    "decisions",
    "outcomes",
    "audits",
)


class PaperEvidenceReader(Protocol):
    """Small observability port; it deliberately knows no BrokerAdapter."""

    def list_sessions(self) -> tuple[dict[str, Any], ...]: ...

    def inspect(self, session_id: str) -> dict[str, Any]: ...


class LocalPaperMonitoringReader:
    """Verify and render immutable Paper artifacts without broker imports.

    The writer remains owned by the broker package. This separate reader keeps
    Dashboard/Monitoring read-only and prevents the UI from acquiring a trading
    mutation dependency.
    """

    def __init__(self, root: Path | str = Path("data_local/paper")) -> None:
        self.root = Path(root)

    def _session(self, session_id: str) -> Path:
        if not _SAFE_ID.fullmatch(session_id):
            raise MonitoringIntegrityError("invalid Paper session identifier")
        root = self.root.resolve()
        target = (self.root / session_id).resolve()
        if root not in target.parents:
            raise MonitoringIntegrityError("Paper session path escapes local store")
        return target

    @staticmethod
    def _sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def verify(self, session_id: str) -> dict[str, Any]:
        directory = self._session(session_id)
        try:
            manifest = json.loads(
                (directory / "checksums.json").read_text(encoding="utf-8")
            )
            if manifest["schema_version"] != _PAPER_STORE_SCHEMA_VERSION:
                raise MonitoringIntegrityError("unsupported Paper store schema")
            declared = manifest["files"]
            if not isinstance(declared, dict):
                raise MonitoringIntegrityError("invalid Paper checksum manifest")
            for name, expected in declared.items():
                path = (directory / str(name)).resolve()
                if directory.resolve() not in path.parents or not path.is_file():
                    raise MonitoringIntegrityError(
                        "Paper checksum references an invalid path"
                    )
                if self._sha256(path.read_bytes()) != expected:
                    raise MonitoringIntegrityError(
                        f"Paper artifact checksum mismatch: {name}"
                    )
            actual = {
                str(path.relative_to(directory)).replace("\\", "/")
                for path in directory.rglob("*.json")
                if path.name != "checksums.json"
            }
            if actual != set(declared):
                raise MonitoringIntegrityError(
                    "Paper store contains unmanifested JSON artifacts"
                )
            return manifest
        except MonitoringIntegrityError:
            raise
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise MonitoringIntegrityError("invalid Paper monitoring evidence") from exc

    def list_sessions(self) -> tuple[dict[str, Any], ...]:
        if not self.root.is_dir():
            return ()
        rows: list[dict[str, Any]] = []
        for directory in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                continue
            try:
                self.verify(directory.name)
                manifest = json.loads(
                    (directory / "session_manifest.json").read_text(encoding="utf-8")
                )
                rows.append(
                    redact_sensitive(
                        {
                            "session_id": directory.name,
                            "mode": manifest["mode"],
                            "account_masked": manifest["account_masked"],
                            "paper_execution_armed": manifest[
                                "paper_execution_armed"
                            ],
                            "integrity": "VERIFIED",
                        }
                    )
                )
            except MonitoringIntegrityError:
                rows.append({"session_id": directory.name, "integrity": "ERROR"})
        return tuple(rows)

    def inspect(self, session_id: str) -> dict[str, Any]:
        self.verify(session_id)
        directory = self._session(session_id)
        try:
            session = json.loads(
                (directory / "session_manifest.json").read_text(encoding="utf-8")
            )
            records: dict[str, list[dict[str, Any]]] = {}
            for category in _CATEGORIES:
                category_path = directory / category
                records[category] = (
                    [
                        json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted(category_path.glob("*.json"))
                    ]
                    if category_path.is_dir()
                    else []
                )
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise MonitoringIntegrityError("Paper evidence cannot be decoded") from exc
        payload = {
            "schema_version": _PAPER_STORE_SCHEMA_VERSION,
            "integrity": "VERIFIED",
            "session": session,
            **records,
        }
        payload["evidence_hash"] = stable_hash({"session": session, **records})
        return redact_sensitive(payload)

    def replay_summary(self, session_id: str) -> dict[str, Any]:
        payload = self.inspect(session_id)
        deterministic = {
            "session": payload["session"],
            "events": payload["events"],
            "orders": payload["orders"],
            "executions": payload["executions"],
        }
        return {
            "session_id": session_id,
            "event_count": len(payload["events"]),
            "order_count": len(payload["orders"]),
            "execution_count": len(payload["executions"]),
            "replay_hash": stable_hash(deterministic),
            "broker_fills_reproduced": False,
        }

    def shadow_audit_summary(self, session_id: str) -> dict[str, Any]:
        payload = self.inspect(session_id)
        persisted = payload["audits"]
        return {
            "session_id": session_id,
            "decision_envelopes": len(payload["decisions"]),
            "outcome_envelopes": len(payload["outcomes"]),
            "persisted_audits": persisted,
            "status": "UNAVAILABLE" if not persisted else "PERSISTED",
            "read_only": True,
        }


class PaperMonitoringSource(EventMonitoringSource):
    def __init__(self, store: PaperEvidenceReader, session_id: str) -> None:
        self.store = store
        self.session_id = session_id

    def events_after(self, cursor: str | None = None) -> tuple[MonitoringEvent, ...]:
        payload = self.store.inspect(self.session_id)
        rows = payload["events"]
        if cursor is not None:
            rows = [row for row in rows if str(row.get("event_id", "")) > cursor]
        events = []
        for row in rows:
            related = tuple(sorted(tuple(item) for item in row.get("related_ids", ())))
            events.append(
                MonitoringEvent(
                    event_id=str(row["event_id"]),
                    timestamp=datetime.fromisoformat(str(row["received_at"])),
                    event_type=MonitoringEventType.BROKER_EVENT,
                    run_id=self.session_id,
                    session_id=self.session_id,
                    source_component=str(row.get("source", "BROKER")),
                    component_version=str(row.get("source_version", "UNKNOWN")),
                    related_ids=related,
                    payload_json=json.dumps(row.get("payload_json", {}))
                    if isinstance(row.get("payload_json"), dict)
                    else str(row.get("payload_json", "{}")),
                    status=str(row.get("event_type", "UNKNOWN")),
                )
            )
        return tuple(sorted(events, key=lambda item: (item.timestamp, item.event_id)))
