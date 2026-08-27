"""Offline regime and activation reporting without strategy selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from trading_ai.regimes.models import (
    ActivationDecision,
    ActivationStatus,
    RegimeReport,
    RegimeSnapshot,
    RegimeTransition,
)


def build_regime_report(
    snapshots: Sequence[RegimeSnapshot],
    transitions: Sequence[RegimeTransition],
    decisions: Sequence[ActivationDecision],
) -> RegimeReport:
    structure = Counter(snapshot.structure_regime.value for snapshot in snapshots)
    volatility = Counter(snapshot.volatility_regime.value for snapshot in snapshots)
    signals = Counter(
        f"{decision.strategy_name}|{decision.structure_regime.value}|"
        f"{decision.volatility_regime.value}"
        for decision in decisions
    )
    statuses = Counter(decision.status for decision in decisions)
    return RegimeReport(
        bars_by_structure_regime=tuple(sorted(structure.items())),
        bars_by_volatility_regime=tuple(sorted(volatility.items())),
        transition_count=len(tuple(transitions)),
        signals_by_regime=tuple(sorted(signals.items())),
        activation_allow=statuses[ActivationStatus.ALLOW],
        activation_reduce=statuses[ActivationStatus.REDUCE],
        activation_block=statuses[ActivationStatus.BLOCK],
    )
