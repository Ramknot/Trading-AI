"""Pure market-structure features whose reference window excludes the present."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import pstdev


def efficiency_ratio(values: Sequence[float], window: int) -> float | None:
    """Return direction divided by total path length over ``window`` changes.

    The current close is included and no observation after it is required. A
    completely flat window has no measurable ratio and returns ``None``.
    """

    if window < 1:
        raise ValueError("window must be positive")
    if len(values) <= window:
        return None
    sample = values[-1 - window :]
    movement = abs(sample[-1] - sample[0])
    noise = sum(
        abs(current - previous)
        for previous, current in zip(sample, sample[1:])
    )
    if noise == 0.0:
        return None
    return movement / noise


def price_zscore(values: Sequence[float], window: int) -> float | None:
    """Population z-score of the current close in the latest full window."""

    if window < 2:
        raise ValueError("window must be at least 2")
    if len(values) < window:
        return None
    sample = values[-window:]
    mean = sum(sample) / window
    deviation = pstdev(sample)
    if deviation == 0.0:
        return None
    return (sample[-1] - mean) / deviation


def previous_rolling_high(values: Sequence[float], window: int) -> float | None:
    if window < 1:
        raise ValueError("window must be positive")
    if len(values) <= window:
        return None
    return max(values[-1 - window : -1])


def previous_rolling_low(values: Sequence[float], window: int) -> float | None:
    if window < 1:
        raise ValueError("window must be positive")
    if len(values) <= window:
        return None
    return min(values[-1 - window : -1])


def distance_to_level(price: float, level: float | None) -> float | None:
    if level is None or level == 0.0:
        return None
    return price / level - 1.0
