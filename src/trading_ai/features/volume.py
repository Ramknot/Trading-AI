"""Pure volume features with explicit zero-volume behavior."""

from __future__ import annotations

from collections.abc import Sequence


def rolling_average_volume(values: Sequence[float], window: int) -> float | None:
    if window < 1:
        raise ValueError("window must be positive")
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def relative_volume(values: Sequence[float], window: int) -> float | None:
    average = rolling_average_volume(values, window)
    if average is None or average == 0.0:
        return None
    return values[-1] / average
