"""Pure true-range and volatility calculations."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import stdev


def true_range(high: float, low: float, previous_close: float | None) -> float:
    if previous_close is None:
        return high - low
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def average_true_range_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: int,
) -> tuple[float | None, ...]:
    """Wilder ATR seeded by the first full-window mean of true ranges."""

    if window < 1:
        raise ValueError("window must be positive")
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("OHLC series must have equal lengths")
    ranges = [
        true_range(high, low, closes[index - 1] if index else None)
        for index, (high, low) in enumerate(zip(highs, lows))
    ]
    result: list[float | None] = [None] * len(ranges)
    if len(ranges) < window:
        return tuple(result)
    current = sum(ranges[:window]) / window
    result[window - 1] = current
    for index in range(window, len(ranges)):
        current = ((current * (window - 1)) + ranges[index]) / window
        result[index] = current
    return tuple(result)


def rolling_volatility(values: Sequence[float], window: int) -> float | None:
    """Sample standard deviation of the latest ``window`` simple returns."""

    if window < 2:
        raise ValueError("window must be at least 2")
    if len(values) <= window:
        return None
    returns: list[float] = []
    start = len(values) - window
    for index in range(start, len(values)):
        previous = values[index - 1]
        if previous == 0.0:
            return None
        returns.append(values[index] / previous - 1.0)
    return stdev(returns)
