"""Deterministic daily-loss and drawdown state tracking."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from trading_ai.risk.config import BalancedRiskConfig
from trading_ai.risk.models import (
    CircuitBreakerReason,
    RiskState,
    RiskStateSnapshot,
    RiskStateTransition,
)


ZERO = Decimal("0")


class RiskStateTracker:
    """Track NORMAL/REDUCED/HALTED without a broker or future observations."""

    def __init__(self, config: BalancedRiskConfig) -> None:
        self._config = config
        self._snapshot: RiskStateSnapshot | None = None
        self._transitions: list[RiskStateTransition] = []
        self._latched_reason: CircuitBreakerReason | None = None
        self._max_daily_loss = 0.0
        self._max_drawdown = 0.0

    @property
    def snapshot(self) -> RiskStateSnapshot:
        if self._snapshot is None:
            raise RuntimeError("risk state must be reset before use")
        return self._snapshot

    @property
    def transitions(self) -> tuple[RiskStateTransition, ...]:
        return tuple(self._transitions)

    @property
    def max_daily_loss(self) -> float:
        return self._max_daily_loss

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    def reset(self, timestamp: datetime, equity: Decimal) -> RiskStateSnapshot:
        """Start a fresh run; this is never an implicit intra-run reactivation."""

        self._validate_observation(timestamp, equity)
        risk_day = timestamp.astimezone(ZoneInfo(self._config.risk_day_timezone)).date()
        self._transitions.clear()
        self._latched_reason = None
        self._max_daily_loss = 0.0
        self._max_drawdown = 0.0
        self._snapshot = RiskStateSnapshot(
            timestamp=timestamp,
            state=RiskState.NORMAL,
            peak_equity=equity,
            day_start_equity=equity,
            current_equity=equity,
            risk_day=risk_day,
            daily_loss_pct=0.0,
            drawdown_pct=0.0,
        )
        return self._snapshot

    def observe(self, timestamp: datetime, equity: Decimal) -> RiskStateSnapshot:
        self._validate_observation(timestamp, equity)
        if self._snapshot is None:
            return self.reset(timestamp, equity)
        previous = self._snapshot
        if timestamp < previous.timestamp:
            raise ValueError("risk observations must be chronological")
        risk_day = timestamp.astimezone(ZoneInfo(self._config.risk_day_timezone)).date()
        is_new_day = risk_day != previous.risk_day
        day_start_equity = equity if is_new_day else previous.day_start_equity
        peak = max(previous.peak_equity, equity)
        daily_loss = (
            max(ZERO, (day_start_equity - equity) / day_start_equity)
            if day_start_equity > ZERO
            else Decimal("1")
        )
        drawdown = (
            max(ZERO, (peak - equity) / peak) if peak > ZERO else Decimal("1")
        )

        if (
            self._latched_reason is CircuitBreakerReason.DAILY_LOSS_LIMIT
            and is_new_day
        ):
            self._latched_reason = None
        if drawdown >= self._config.hard_drawdown_limit:
            self._latched_reason = CircuitBreakerReason.HARD_DRAWDOWN
        elif (
            self._latched_reason is None
            and daily_loss >= self._config.daily_loss_limit
        ):
            self._latched_reason = CircuitBreakerReason.DAILY_LOSS_LIMIT

        if self._latched_reason is not None:
            state = RiskState.HALTED
            halt_reason = self._latched_reason.value
        elif drawdown >= self._config.soft_drawdown_limit:
            state = RiskState.REDUCED
            halt_reason = None
        else:
            state = RiskState.NORMAL
            halt_reason = None

        current = RiskStateSnapshot(
            timestamp=timestamp,
            state=state,
            peak_equity=peak,
            day_start_equity=day_start_equity,
            current_equity=equity,
            risk_day=risk_day,
            daily_loss_pct=float(daily_loss),
            drawdown_pct=float(drawdown),
            halt_reason=halt_reason,
        )
        self._max_daily_loss = max(self._max_daily_loss, current.daily_loss_pct)
        self._max_drawdown = max(self._max_drawdown, current.drawdown_pct)
        self._record_transition(previous, current, halt_reason or self._state_reason(current))
        self._snapshot = current
        return current

    def halt(
        self,
        reason: CircuitBreakerReason,
        timestamp: datetime,
        equity: Decimal,
    ) -> RiskStateSnapshot:
        """Latch a manual or safety halt; no portfolio liquidation is performed."""

        current = self.observe(timestamp, equity)
        previous = current
        self._latched_reason = reason
        halted = RiskStateSnapshot(
            timestamp=timestamp,
            state=RiskState.HALTED,
            peak_equity=current.peak_equity,
            day_start_equity=current.day_start_equity,
            current_equity=equity,
            risk_day=current.risk_day,
            daily_loss_pct=current.daily_loss_pct,
            drawdown_pct=current.drawdown_pct,
            halt_reason=reason.value,
        )
        self._record_transition(previous, halted, reason.value)
        self._snapshot = halted
        return halted

    def reset_halt(
        self,
        timestamp: datetime,
        equity: Decimal,
        *,
        authorization_reason: str,
    ) -> RiskStateSnapshot:
        """Explicitly clear a latched halt; never called automatically."""

        if not authorization_reason.strip():
            raise ValueError("authorization_reason is required for risk reset")
        previous = self.snapshot
        if timestamp < previous.timestamp:
            raise ValueError("risk reset must be chronological")
        self._latched_reason = None
        current = self.observe(timestamp, equity)
        if previous.state is RiskState.HALTED and current.state is RiskState.HALTED:
            # Current equity can immediately retrigger a hard/daily threshold.
            return current
        return current

    def state_durations(self, completed_at: datetime) -> tuple[float, float]:
        """Return seconds spent in REDUCED and HALTED during this run."""

        snapshot = self.snapshot
        if completed_at < snapshot.timestamp:
            raise ValueError("completed_at precedes the latest risk observation")
        reduced = 0.0
        halted = 0.0
        if not self._transitions:
            duration = (completed_at - snapshot.timestamp).total_seconds()
            if snapshot.state is RiskState.REDUCED:
                reduced += duration
            elif snapshot.state is RiskState.HALTED:
                halted += duration
            return reduced, halted

        initial_time = min(item.timestamp for item in self._transitions)
        state = self._transitions[0].previous_state
        cursor = initial_time
        for transition in self._transitions:
            duration = max(0.0, (transition.timestamp - cursor).total_seconds())
            if state is RiskState.REDUCED:
                reduced += duration
            elif state is RiskState.HALTED:
                halted += duration
            state = transition.new_state
            cursor = transition.timestamp
        duration = max(0.0, (completed_at - cursor).total_seconds())
        if state is RiskState.REDUCED:
            reduced += duration
        elif state is RiskState.HALTED:
            halted += duration
        return reduced, halted

    @staticmethod
    def _validate_observation(timestamp: datetime, equity: Decimal) -> None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("risk observation timestamp must be timezone-aware")
        if not equity.is_finite() or equity < ZERO:
            raise ValueError("risk observation equity must be finite and non-negative")

    @staticmethod
    def _state_reason(snapshot: RiskStateSnapshot) -> str:
        if snapshot.state is RiskState.REDUCED:
            return "SOFT_DRAWDOWN"
        return snapshot.state.value

    def _record_transition(
        self,
        previous: RiskStateSnapshot,
        current: RiskStateSnapshot,
        reason: str,
    ) -> None:
        if previous.state is current.state:
            return
        self._transitions.append(
            RiskStateTransition(
                transition_id=f"risk-state-{len(self._transitions) + 1:06d}",
                timestamp=current.timestamp,
                previous_state=previous.state,
                new_state=current.state,
                reason=reason,
                equity=current.current_equity,
                daily_loss_pct=current.daily_loss_pct,
                drawdown_pct=current.drawdown_pct,
            )
        )
