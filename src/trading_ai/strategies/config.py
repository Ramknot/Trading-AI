"""Immutable, non-optimized configuration for Lot 3 research baselines."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


def _positive_integer(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _fraction(value: Decimal, field_name: str) -> None:
    if not value.is_finite() or not Decimal("0") < value <= Decimal("1"):
        raise ValueError(f"{field_name} must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class TrendConfig:
    """Trend windows are bars, not days; defaults are not optimized."""

    fast_window: int = 20
    slow_window: int = 50
    slope_lookback: int = 5
    allocation_fraction: Decimal = Decimal("0.25")

    def __post_init__(self) -> None:
        for field_name in ("fast_window", "slow_window", "slope_lookback"):
            _positive_integer(getattr(self, field_name), field_name)
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be lower than slow_window")
        _fraction(self.allocation_fraction, "allocation_fraction")

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    ("allocation_fraction", str(self.allocation_fraction)),
                    ("fast_window", str(self.fast_window)),
                    ("slope_lookback", str(self.slope_lookback)),
                    ("slow_window", str(self.slow_window)),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    """Cross-sectional selection baseline with bar-count rebalancing."""

    lookback: int = 20
    top_k: int = 3
    rebalance_every: int = 5
    allocation_fraction: Decimal = Decimal("0.60")
    minimum_return: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for field_name in ("lookback", "top_k", "rebalance_every"):
            _positive_integer(getattr(self, field_name), field_name)
        _fraction(self.allocation_fraction, "allocation_fraction")
        if not self.minimum_return.is_finite() or self.minimum_return <= Decimal("-1"):
            raise ValueError("minimum_return must be finite and greater than -1")

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    ("allocation_fraction", str(self.allocation_fraction)),
                    ("lookback", str(self.lookback)),
                    ("minimum_return", str(self.minimum_return)),
                    ("rebalance_every", str(self.rebalance_every)),
                    ("top_k", str(self.top_k)),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class BreakoutConfig:
    """Previous-range breakout baseline; both windows are bars."""

    entry_window: int = 20
    exit_window: int = 10
    allocation_fraction: Decimal = Decimal("0.25")

    def __post_init__(self) -> None:
        _positive_integer(self.entry_window, "entry_window")
        _positive_integer(self.exit_window, "exit_window")
        _fraction(self.allocation_fraction, "allocation_fraction")

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    ("allocation_fraction", str(self.allocation_fraction)),
                    ("entry_window", str(self.entry_window)),
                    ("exit_window", str(self.exit_window)),
                )
            )
        )
