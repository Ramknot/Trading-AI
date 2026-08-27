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


def rolling_volatility_series(
    values: Sequence[float], window: int
) -> tuple[float | None, ...]:
    """Sample-volatility series using only returns realized at each point."""

    if window < 2:
        raise ValueError("window must be at least 2")
    result: list[float | None] = [None] * len(values)
    for end in range(window, len(values)):
        returns: list[float] = []
        for index in range(end - window + 1, end + 1):
            previous = values[index - 1]
            if previous == 0.0:
                returns = []
                break
            returns.append(values[index] / previous - 1.0)
        if len(returns) == window:
            result[end] = stdev(returns)
    return tuple(result)


def rolling_volatility(values: Sequence[float], window: int) -> float | None:
    """Sample standard deviation of the latest ``window`` simple returns."""

    series = rolling_volatility_series(values, window)
    return series[-1] if series else None


def rolling_percentile(
    values: Sequence[float | None], window: int
) -> float | None:
    """Mid-rank empirical percentile for the latest available observation.

    A complete trailing window of already-observed volatility values is
    required. Ties receive their midpoint rank, so a constant series is 0.5.
    """

    if window < 2:
        raise ValueError("window must be at least 2")
    available = [value for value in values if value is not None]
    if len(available) < window:
        return None
    sample = available[-window:]
    current = sample[-1]
    lower = sum(value < current for value in sample)
    equal = sum(value == current for value in sample)
    return (lower + equal / 2.0) / window
