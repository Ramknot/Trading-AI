"""Point-in-time correlation guard using exact common return timestamps."""

from __future__ import annotations

from math import sqrt

from trading_ai.features.models import ReturnSeries
from trading_ai.risk.models import CorrelationAssessment


def aligned_correlation(
    left: ReturnSeries,
    right: ReturnSeries,
    *,
    minimum_observations: int,
) -> tuple[float | None, int]:
    """Pearson correlation with no forward/back/future fill."""

    left_values = {item.timestamp: item.value for item in left.observations}
    right_values = {item.timestamp: item.value for item in right.observations}
    common = sorted(set(left_values).intersection(right_values))
    count = len(common)
    if count < minimum_observations:
        return None, count
    x = [left_values[timestamp] for timestamp in common]
    y = [right_values[timestamp] for timestamp in common]
    mean_x = sum(x) / count
    mean_y = sum(y) / count
    covariance = sum(
        (left_value - mean_x) * (right_value - mean_y)
        for left_value, right_value in zip(x, y)
    )
    variance_x = sum((value - mean_x) ** 2 for value in x)
    variance_y = sum((value - mean_y) ** 2 for value in y)
    denominator = sqrt(variance_x * variance_y)
    if denominator == 0:
        return None, count
    return covariance / denominator, count


class CorrelationGuard:
    def __init__(self, *, threshold: float, minimum_observations: int) -> None:
        self.threshold = threshold
        self.minimum_observations = minimum_observations

    def assess(
        self,
        symbol: str,
        open_symbols: tuple[str, ...],
        series: tuple[ReturnSeries, ...],
    ) -> tuple[CorrelationAssessment, ...]:
        by_symbol = {item.symbol: item for item in series}
        target = by_symbol.get(symbol)
        assessments: list[CorrelationAssessment] = []
        for existing in sorted(set(open_symbols) - {symbol}):
            other = by_symbol.get(existing)
            if target is None or other is None:
                coefficient, observations = None, 0
            else:
                coefficient, observations = aligned_correlation(
                    target,
                    other,
                    minimum_observations=self.minimum_observations,
                )
            assessments.append(
                CorrelationAssessment(
                    symbol=existing,
                    coefficient=coefficient,
                    observations=observations,
                    highly_correlated=(
                        coefficient is not None and coefficient >= self.threshold
                    ),
                )
            )
        return tuple(assessments)
