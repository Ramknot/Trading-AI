"""Deterministic aggregate metrics for exported backtest risk provenance."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from trading_ai.core.models import RiskDecision, RiskDecisionStatus
from trading_ai.risk.models import RiskSummary
from trading_ai.risk.state import RiskStateTracker


def summarize_risk(
    *,
    engine_name: str,
    engine_version: str,
    config_hash: str,
    decisions: tuple[RiskDecision, ...],
    tracker: RiskStateTracker | None,
    completed_at: datetime,
) -> RiskSummary:
    rejection_reasons = Counter(
        code
        for decision in decisions
        if decision.status is RiskDecisionStatus.REJECT
        for code in decision.reason_codes
    )
    reduced_seconds = halted_seconds = 0.0
    if tracker is not None:
        reduced_seconds, halted_seconds = tracker.state_durations(completed_at)
    return RiskSummary(
        risk_engine_name=engine_name,
        risk_engine_version=engine_version,
        risk_config_hash=config_hash,
        approved_orders=sum(
            item.status is RiskDecisionStatus.APPROVE for item in decisions
        ),
        reduced_orders=sum(
            item.status is RiskDecisionStatus.REDUCE for item in decisions
        ),
        rejected_orders=sum(
            item.status is RiskDecisionStatus.REJECT for item in decisions
        ),
        rejection_reasons=tuple(sorted(rejection_reasons.items())),
        max_portfolio_exposure=max(
            (item.gross_exposure_after or 0.0 for item in decisions), default=0.0
        ),
        max_single_position_exposure=max(
            (item.position_exposure_after or 0.0 for item in decisions), default=0.0
        ),
        max_observed_drawdown=max(
            max((item.drawdown_pct or 0.0 for item in decisions), default=0.0),
            tracker.max_drawdown if tracker is not None else 0.0,
        ),
        max_daily_loss=max(
            max((item.daily_loss_pct or 0.0 for item in decisions), default=0.0),
            tracker.max_daily_loss if tracker is not None else 0.0,
        ),
        time_in_reduced_state_seconds=reduced_seconds,
        time_in_halted_state_seconds=halted_seconds,
    )
