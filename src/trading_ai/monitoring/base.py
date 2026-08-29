"""Provider-neutral observability source and persistence contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trading_ai.monitoring.models import MonitoringEvent, MonitoringSnapshot


class MonitoringSource(ABC):
    """Read-only source of already-produced engine observations."""

    @abstractmethod
    def list_runs(self) -> tuple[dict[str, Any], ...]:
        raise NotImplementedError

    @abstractmethod
    def load_run(self, run_id: str) -> Any:
        raise NotImplementedError


class EventMonitoringSource(ABC):
    """Future Paper/Live event boundary; no broker or network implementation exists."""

    @abstractmethod
    def events_after(self, cursor: str | None = None) -> tuple[MonitoringEvent, ...]:
        raise NotImplementedError


class MonitoringStore(ABC):
    """Local event/snapshot persistence independent from dashboard transport."""

    @abstractmethod
    def append_event(self, event: MonitoringEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_events(self, events: tuple[MonitoringEvent, ...]) -> None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def save_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_snapshot(self, snapshot_id: str) -> MonitoringSnapshot | None:
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> bool:
        raise NotImplementedError
