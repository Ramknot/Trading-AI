"""Small local SQLite store for reusable monitoring events and snapshots."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from trading_ai.monitoring.base import MonitoringStore
from trading_ai.monitoring.exceptions import MonitoringError
from trading_ai.monitoring.models import (
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSnapshot,
    SystemStatus,
)


class SQLiteMonitoringStore(MonitoringStore):
    """Local-only observability persistence using the Python standard library."""

    def __init__(self, path: Path | str = Path("data_local/monitoring/monitoring.db")):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS monitoring_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        source_component TEXT NOT NULL,
                        component_version TEXT NOT NULL,
                        related_ids_json TEXT NOT NULL,
                        provenance_json TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        symbol TEXT,
                        strategy_name TEXT,
                        status TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_monitoring_events_run_time
                        ON monitoring_events(run_id, timestamp, event_id);
                    CREATE TABLE IF NOT EXISTS monitoring_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        source_schema_version TEXT NOT NULL,
                        source_fingerprint TEXT NOT NULL,
                        sections_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_monitoring_snapshots_run
                        ON monitoring_snapshots(run_id, timestamp);
                    """
                )
        except sqlite3.Error as exc:
            raise MonitoringError("unable to initialize local monitoring store") from exc

    def append_event(self, event: MonitoringEvent) -> None:
        self.append_events((event,))

    def append_events(self, events: tuple[MonitoringEvent, ...]) -> None:
        if not events:
            return
        try:
            with self._connect() as connection:
                for event in events:
                    cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO monitoring_events (
                        event_id, timestamp, event_type, run_id, session_id,
                        source_component, component_version, related_ids_json,
                        provenance_json, payload_json, symbol, strategy_name, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                        event.event_id,
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        event.run_id,
                        event.session_id,
                        event.source_component,
                        event.component_version,
                        json.dumps(event.related_ids, separators=(",", ":")),
                        json.dumps(event.provenance, separators=(",", ":")),
                        event.payload_json,
                        event.symbol,
                        event.strategy_name,
                        event.status,
                        ),
                    )
                    if cursor.rowcount == 0:
                        existing = connection.execute(
                            "SELECT * FROM monitoring_events WHERE event_id = ?",
                            (event.event_id,),
                        ).fetchone()
                        if existing is None or self._event_from_row(existing) != event:
                            raise MonitoringError(
                                "monitoring event IDs are immutable and unique"
                            )
        except sqlite3.Error as exc:
            raise MonitoringError("unable to append monitoring event") from exc

    def list_events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> tuple[MonitoringEvent, ...]:
        if limit < 1 or limit > 5000:
            raise ValueError("event limit must be in [1, 5000]")
        clauses = ["run_id = ?"]
        values: list[object] = [run_id]
        for column, value in (
            ("event_type", event_type),
            ("symbol", symbol),
            ("strategy_name", strategy_name),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        values.append(limit)
        query = (
            "SELECT * FROM monitoring_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY timestamp, event_id LIMIT ?"
        )
        try:
            with self._connect() as connection:
                rows = connection.execute(query, values).fetchall()
        except sqlite3.Error as exc:
            raise MonitoringError("unable to read monitoring events") from exc
        return tuple(self._event_from_row(row) for row in rows)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> MonitoringEvent:
        return MonitoringEvent(
            event_id=row["event_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            event_type=MonitoringEventType(row["event_type"]),
            run_id=row["run_id"],
            session_id=row["session_id"],
            source_component=row["source_component"],
            component_version=row["component_version"],
            related_ids=tuple(tuple(item) for item in json.loads(row["related_ids_json"])),
            provenance=tuple(tuple(item) for item in json.loads(row["provenance_json"])),
            payload_json=row["payload_json"],
            symbol=row["symbol"],
            strategy_name=row["strategy_name"],
            status=row["status"],
        )

    def save_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO monitoring_snapshots (
                        snapshot_id, run_id, timestamp, mode, status,
                        source_schema_version, source_fingerprint, sections_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_id) DO UPDATE SET
                        sections_json=excluded.sections_json,
                        status=excluded.status
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.run_id,
                        snapshot.timestamp.isoformat(),
                        snapshot.mode,
                        snapshot.status.value,
                        snapshot.source_schema_version,
                        snapshot.source_fingerprint,
                        snapshot.sections_json,
                    ),
                )
        except sqlite3.Error as exc:
            raise MonitoringError("unable to save monitoring snapshot") from exc

    def load_snapshot(self, snapshot_id: str) -> MonitoringSnapshot | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM monitoring_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise MonitoringError("unable to load monitoring snapshot") from exc
        if row is None:
            return None
        return MonitoringSnapshot(
            snapshot_id=row["snapshot_id"],
            run_id=row["run_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            mode=row["mode"],
            status=SystemStatus(row["status"]),
            source_schema_version=row["source_schema_version"],
            source_fingerprint=row["source_fingerprint"],
            sections_json=row["sections_json"],
        )

    def is_healthy(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False
