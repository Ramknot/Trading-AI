"""Explicit future-only label construction isolated from all model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from trading_ai.core.models import MarketBar


class EntryReference(str, Enum):
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"


class ExitReference(str, Enum):
    HORIZON_CLOSE = "HORIZON_CLOSE"


@dataclass(frozen=True, slots=True)
class LabelConfig:
    horizon_bars: int = 5
    minimum_forward_return_bps: Decimal = Decimal("0")
    entry_reference: EntryReference = EntryReference.NEXT_BAR_OPEN
    exit_reference: ExitReference = ExitReference.HORIZON_CLOSE

    def __post_init__(self) -> None:
        if type(self.horizon_bars) is not int or self.horizon_bars < 1:
            raise ValueError("horizon_bars must be a positive integer")
        if not self.minimum_forward_return_bps.is_finite():
            raise ValueError("minimum_forward_return_bps must be finite")

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    ("entry_reference", self.entry_reference.value),
                    ("exit_reference", self.exit_reference.value),
                    ("horizon_bars", str(self.horizon_bars)),
                    (
                        "minimum_forward_return_bps",
                        str(self.minimum_forward_return_bps),
                    ),
                )
            )
        )

@dataclass(frozen=True, slots=True)
class SignalLabel:
    signal_timestamp: datetime
    label_end_timestamp: datetime
    entry_price: Decimal
    exit_price: Decimal
    forward_return: float
    target: int

    def __post_init__(self) -> None:
        if self.signal_timestamp.tzinfo is None or self.label_end_timestamp.tzinfo is None:
            raise ValueError("label timestamps must be timezone-aware")
        if self.label_end_timestamp <= self.signal_timestamp:
            raise ValueError("label end must be after its signal")
        if self.entry_price <= 0 or self.exit_price <= 0:
            raise ValueError("label prices must be positive")
        if self.target not in {0, 1}:
            raise ValueError("binary label target must be 0 or 1")


class LabelBuilder:
    """Build y from future bars; this is the only production future-data boundary."""

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    def build(
        self, bars: tuple[MarketBar, ...], signal_index: int
    ) -> SignalLabel | None:
        if not bars:
            return None
        if signal_index < 0 or signal_index >= len(bars):
            raise IndexError("signal_index is outside the bar series")
        if any(
            bar.symbol != bars[0].symbol or bar.timeframe != bars[0].timeframe
            for bar in bars
        ):
            raise ValueError("label series must contain one symbol and timeframe")
        timestamps = [bar.timestamp for bar in bars]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("label bars must be unique and chronological")
        entry_index = signal_index + 1
        exit_index = signal_index + self.config.horizon_bars
        if entry_index >= len(bars) or exit_index >= len(bars):
            return None
        entry_price = bars[entry_index].open
        exit_price = bars[exit_index].close
        forward_return = float(exit_price / entry_price - Decimal("1"))
        threshold = float(self.config.minimum_forward_return_bps / Decimal("10000"))
        return SignalLabel(
            signal_timestamp=bars[signal_index].timestamp,
            label_end_timestamp=bars[exit_index].timestamp,
            entry_price=entry_price,
            exit_price=exit_price,
            forward_return=forward_return,
            target=int(forward_return > threshold),
        )
