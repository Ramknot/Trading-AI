"""Pure trend calculations with explicit warm-up semantics."""

from __future__ import annotations

from collections.abc import Sequence


def simple_moving_average_series(
    values: Sequence[float], window: int
) -> tuple[float | None, ...]:
    if window < 1:
        raise ValueError("window must be positive")
    result: list[float | None] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / window if index + 1 >= window else None)
    return tuple(result)


def exponential_moving_average_series(
    values: Sequence[float], window: int
) -> tuple[float | None, ...]:
    """EMA seeded by the first full-window SMA; no partial warm-up values."""

    if window < 1:
        raise ValueError("window must be positive")
    if not values:
        return ()
    result: list[float | None] = [None] * len(values)
    if len(values) < window:
        return tuple(result)
    seed = sum(values[:window]) / window
    result[window - 1] = seed
    alpha = 2.0 / (window + 1.0)
    previous = seed
    for index in range(window, len(values)):
        previous = alpha * values[index] + (1.0 - alpha) * previous
        result[index] = previous
    return tuple(result)


def moving_average_slope(
    moving_average: Sequence[float | None], lookback: int
) -> float | None:
    """Average per-bar change over ``lookback`` bars at the series end."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    if len(moving_average) <= lookback:
        return None
    current = moving_average[-1]
    previous = moving_average[-1 - lookback]
    if current is None or previous is None:
        return None
    return (current - previous) / lookback


def price_to_average_distance(price: float, average: float | None) -> float | None:
    if average is None or average == 0.0:
        return None
    return price / average - 1.0
