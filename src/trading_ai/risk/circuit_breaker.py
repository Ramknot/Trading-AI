"""Explicit STOP_NEW_RISK circuit breaker without automatic liquidation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_ai.risk.models import CircuitBreakerReason, RiskStateSnapshot
from trading_ai.risk.state import RiskStateTracker


class RiskCircuitBreaker:
    """Small policy boundary around explicit halt/reset operations."""

    def __init__(self, tracker: RiskStateTracker) -> None:
        self._tracker = tracker

    def halt(
        self,
        reason: CircuitBreakerReason,
        timestamp: datetime,
        equity: Decimal,
    ) -> RiskStateSnapshot:
        return self._tracker.halt(reason, timestamp, equity)

    def reset(
        self,
        timestamp: datetime,
        equity: Decimal,
        *,
        authorization_reason: str,
    ) -> RiskStateSnapshot:
        return self._tracker.reset_halt(
            timestamp,
            equity,
            authorization_reason=authorization_reason,
        )
