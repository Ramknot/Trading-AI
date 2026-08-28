"""Soft, point-in-time diversification preferences for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

from trading_ai.features.models import ReturnSeries
from trading_ai.risk.correlation import aligned_correlation


@dataclass(frozen=True, slots=True)
class DiversificationAssessment:
    symbol: str
    group_count: int
    group_unknown: bool
    highest_correlation: float | None
    high_correlation: bool
    correlation_unknown: bool


class PortfolioDiversification:
    """Ranks soft preferences only; hard enforcement remains in RiskEngine."""

    def __init__(self, threshold: float, minimum_observations: int) -> None:
        self.threshold = threshold
        self.minimum_observations = minimum_observations

    def assess(
        self,
        symbol: str,
        selected_symbols: tuple[str, ...],
        series: tuple[ReturnSeries, ...],
        groups: tuple[tuple[str, str | None], ...],
    ) -> DiversificationAssessment:
        by_symbol = {item.symbol: item for item in series}
        by_group = dict(groups)
        group = by_group.get(symbol)
        group_count = sum(
            by_group.get(existing) == group
            for existing in selected_symbols
            if group is not None
        )
        target = by_symbol.get(symbol)
        correlations: list[float] = []
        unknown = False
        for existing in sorted(set(selected_symbols) - {symbol}):
            other = by_symbol.get(existing)
            if target is None or other is None:
                unknown = True
                continue
            coefficient, _ = aligned_correlation(
                target,
                other,
                minimum_observations=self.minimum_observations,
            )
            if coefficient is None:
                unknown = True
            else:
                correlations.append(coefficient)
        highest = max(correlations) if correlations else None
        return DiversificationAssessment(
            symbol=symbol,
            group_count=group_count,
            group_unknown=group is None,
            highest_correlation=highest,
            high_correlation=(highest is not None and highest >= self.threshold),
            correlation_unknown=unknown or (bool(selected_symbols) and highest is None),
        )

    @staticmethod
    def sort_key(
        assessment: DiversificationAssessment,
        *,
        deprioritize_unknown: bool,
    ) -> tuple[int, int, int, int, float]:
        return (
            int(assessment.group_unknown),
            assessment.group_count,
            int(assessment.high_correlation),
            int(deprioritize_unknown and assessment.correlation_unknown),
            assessment.highest_correlation if assessment.highest_correlation is not None else 2.0,
        )
