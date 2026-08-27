"""Pure time-series and cross-sectional momentum calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from trading_ai.features.models import RelativeStrengthValue


def simple_return(values: Sequence[float]) -> float | None:
    return rolling_return(values, 1)


def rolling_return(values: Sequence[float], window: int) -> float | None:
    """Decimal return from exactly ``window`` bars ago to the current bar."""

    if window < 1:
        raise ValueError("window must be positive")
    if len(values) <= window:
        return None
    previous = values[-1 - window]
    if previous == 0.0:
        return None
    return values[-1] / previous - 1.0


def rate_of_change(values: Sequence[float], window: int) -> float | None:
    """Percentage rate of change; ``return_N`` remains the decimal equivalent."""

    value = rolling_return(values, window)
    return None if value is None else value * 100.0


def relative_strength_values(
    returns_by_symbol: Mapping[str, float],
) -> tuple[RelativeStrengthValue, ...]:
    """Rank higher returns first; equal returns receive equal average ranks."""

    ordered = sorted(returns_by_symbol.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return ()
    observations: list[RelativeStrengthValue] = []
    index = 0
    count = len(ordered)
    while index < count:
        end = index + 1
        while end < count and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        percentile = 1.0 if count == 1 else (count - rank) / (count - 1)
        for symbol, value in ordered[index:end]:
            observations.append(
                RelativeStrengthValue(
                    symbol=symbol,
                    rolling_return=value,
                    rank=rank,
                    percentile=percentile,
                )
            )
        index = end
    return tuple(observations)
