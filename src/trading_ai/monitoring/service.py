"""Application service joining verified sources, view models, and local storage."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from trading_ai.backtesting.reproducibility import to_primitive
from trading_ai.monitoring.base import MonitoringSource, MonitoringStore
from trading_ai.monitoring.exceptions import MonitoringNotFoundError
from trading_ai.monitoring.models import MonitoringSnapshot
from trading_ai.monitoring.source import BacktestMonitoringData
from trading_ai.monitoring.views import BacktestViewBuilder


@dataclass(slots=True)
class _SnapshotCache:
    fingerprint: str
    cache_token: str
    snapshot: MonitoringSnapshot


class MonitoringService:
    """Read-only facade consumed by CLI and HTTP without engine mutation."""

    def __init__(
        self,
        source: MonitoringSource,
        store: MonitoringStore,
        builder: BacktestViewBuilder | None = None,
    ) -> None:
        self.source = source
        self.store = store
        self.builder = builder or BacktestViewBuilder()
        self._cache: dict[str, _SnapshotCache] = {}
        self._lock = RLock()

    def list_runs(self) -> tuple[dict[str, Any], ...]:
        return self.source.list_runs()

    def snapshot(self, run_id: str) -> MonitoringSnapshot:
        with self._lock:
            cached = self._cache.get(run_id)
            token_reader = getattr(self.source, "cache_token", None)
            if cached is not None and callable(token_reader):
                if token_reader(run_id) == cached.cache_token:
                    return cached.snapshot
            loaded = self.source.load_run(run_id)
            if not isinstance(loaded, BacktestMonitoringData):
                raise TypeError("MonitoringService requires a normalized monitoring source")
            if cached is not None and cached.fingerprint == loaded.source_fingerprint:
                return cached.snapshot
            snapshot = self.builder.build(
                loaded,
                monitoring_store_healthy=self.store.is_healthy(),
            )
            events_for_data = getattr(self.source, "events_for_data", None)
            if callable(events_for_data):
                self.store.append_events(events_for_data(loaded))
            self.store.save_snapshot(snapshot)
            self._cache[run_id] = _SnapshotCache(
                loaded.source_fingerprint, loaded.cache_token, snapshot
            )
            return snapshot

    def inspect(self, run_id: str) -> dict[str, Any]:
        snapshot = self.snapshot(run_id)
        return {
            "snapshot_id": snapshot.snapshot_id,
            "run_id": snapshot.run_id,
            "timestamp": snapshot.timestamp.isoformat(),
            "mode": snapshot.mode,
            "status": snapshot.status.value,
            "source_schema_version": snapshot.source_schema_version,
            "source_fingerprint": snapshot.source_fingerprint,
            **snapshot.sections,
        }

    def section(self, run_id: str, name: str) -> dict[str, Any] | list[Any]:
        sections = self.snapshot(run_id).sections
        if name not in sections:
            raise MonitoringNotFoundError("monitoring section not found")
        return sections[name]

    def decisions(
        self,
        run_id: str,
        *,
        component: str | None = None,
        symbol: str | None = None,
        strategy: str | None = None,
        status: str | None = None,
        reason: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise ValueError("decision limit must be in [1, 5000]")
        records = list(self.section(run_id, "decisions"))
        filters = {
            "component": component,
            "symbol": symbol,
            "strategy": strategy,
            "status": status,
        }
        for field, expected in filters.items():
            if expected is not None:
                records = [
                    item
                    for item in records
                    if str(item.get(field, "")).casefold() == expected.casefold()
                ]
        if reason is not None:
            needle = reason.casefold()
            records = [
                item
                for item in records
                if needle in " ".join(str(value) for value in item.get("reasons", [])).casefold()
            ]
        return records[:limit]

    def decision_trace(
        self, run_id: str, trace_id: str | None = None
    ) -> dict[str, Any]:
        traces = list(self.section(run_id, "decision_traces"))
        if not traces:
            return {
                "status": "UNAVAILABLE",
                "reason": "no order lineage is available in this run",
                "stages": [],
            }
        if trace_id is None:
            return traces[0]
        for trace in traces:
            if trace.get("trace_id") == trace_id:
                return trace
        raise MonitoringNotFoundError("decision trace not found")

    def health_without_run(self) -> dict[str, Any]:
        runs = self.list_runs()
        return {
            "status": "HEALTHY" if self.store.is_healthy() else "ERROR",
            "monitoring_store": "HEALTHY" if self.store.is_healthy() else "ERROR",
            "available_runs": len(runs),
            "trusted_runs": sum(item.get("integrity") == "VERIFIED" for item in runs),
            "source": type(self.source).__name__,
            "store": type(self.store).__name__,
        }

    def events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        self.snapshot(run_id)
        return [
            to_primitive(item)
            for item in self.store.list_events(
                run_id,
                event_type=event_type,
                symbol=symbol,
                strategy_name=strategy_name,
                status=status,
                limit=limit,
            )
        ]

    @staticmethod
    def primitive(value: Any) -> Any:
        return to_primitive(value)
