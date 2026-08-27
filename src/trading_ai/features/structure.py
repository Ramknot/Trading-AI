"""Pure market-structure features whose reference window excludes the present."""

from __future__ import annotations

from collections.abc import Sequence


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
