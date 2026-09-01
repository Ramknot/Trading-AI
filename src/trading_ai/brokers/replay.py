"""Deterministic local event replay and read-only Paper shadow audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.core.hashing import stable_hash


@dataclass(frozen=True, slots=True)
class PaperReplayReport:
    session_id: str
    event_count: int
    order_count: int
    execution_count: int
    replay_hash: str
    broker_fills_reproduced: bool = False


@dataclass(frozen=True, slots=True)
class PaperShadowAuditReport:
    session_id: str
    decision_envelopes: int
    outcome_envelopes: int
    divergences: tuple[str, ...]
    status: str
    audit_hash: str


class PaperEventReplay:
    def __init__(self, store: LocalPaperStore) -> None:
        self.store = store

    def replay(self, session_id: str) -> PaperReplayReport:
        payload = self.store.inspect(session_id)
        deterministic = {
            "session": payload["session"],
            "events": payload["events"],
            "orders": payload["orders"],
            "executions": payload["executions"],
        }
        return PaperReplayReport(
            session_id=session_id,
            event_count=len(payload["events"]),
            order_count=len(payload["orders"]),
            execution_count=len(payload["executions"]),
            replay_hash=stable_hash(deterministic),
            broker_fills_reproduced=False,
        )


class PaperShadowAudit:
    """Compares persisted deterministic input/output hashes without broker access."""

    def __init__(self, store: LocalPaperStore) -> None:
        self.store = store

    def audit(
        self,
        session_id: str,
        *,
        recalculated_decision_hashes: dict[str, str] | None = None,
    ) -> PaperShadowAuditReport:
        payload = self.store.inspect(session_id)
        comparison_available = recalculated_decision_hashes is not None
        expected = recalculated_decision_hashes or {}
        divergences: list[str] = []
        observed_ids: set[str] = set()
        for envelope in payload["decisions"]:
            envelope_id = str(envelope.get("envelope_id"))
            observed_ids.add(envelope_id)
            observed = stable_hash(envelope)
            if comparison_available and envelope_id not in expected:
                divergences.append(f"MISSING_RECALCULATION:{envelope_id}")
            elif envelope_id in expected and expected[envelope_id] != observed:
                divergences.append(f"HASH_MISMATCH:{envelope_id}")
        if comparison_available:
            for envelope_id in sorted(set(expected) - observed_ids):
                divergences.append(f"UNKNOWN_RECALCULATION:{envelope_id}")
        status = (
            "UNAVAILABLE"
            if not comparison_available
            else "IN_SYNC"
            if not divergences
            else "DRIFT"
        )
        identity: dict[str, Any] = {
            "session_id": session_id,
            "decisions": payload["decisions"],
            "outcomes": payload["outcomes"],
            "comparison_available": comparison_available,
            "recalculated_decision_hashes": dict(sorted(expected.items())),
            "divergences": sorted(divergences),
            "status": status,
        }
        return PaperShadowAuditReport(
            session_id=session_id,
            decision_envelopes=len(payload["decisions"]),
            outcome_envelopes=len(payload["outcomes"]),
            divergences=tuple(sorted(divergences)),
            status=status,
            audit_hash=stable_hash(identity),
        )
