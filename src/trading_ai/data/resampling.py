"""Session-anchored deterministic 1h to 4h bar derivation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable

from trading_ai.core.models import MarketBar
from trading_ai.data.calendar import MarketCalendarService
from trading_ai.data.exceptions import DataValidationError
from trading_ai.data.models import InstrumentMetadata


def resample_1h_to_4h(
    bars: Iterable[MarketBar],
    metadata: InstrumentMetadata,
    *,
    calendar_service: MarketCalendarService | None = None,
) -> tuple[MarketBar, ...]:
    """Group within each exchange session, anchored at that session's open."""

    calendar = calendar_service or MarketCalendarService()
    source_bars = sorted(bars, key=lambda bar: bar.timestamp)
    if not source_bars:
        raise DataValidationError("cannot derive 4h bars from an empty dataset")
    if any(bar.timeframe != "1h" for bar in source_bars):
        raise DataValidationError("4h derivation requires only normalized 1h bars")
    if any(bar.symbol != metadata.symbol for bar in source_bars):
        raise DataValidationError("4h derivation requires one metadata-matched symbol")
    keys = [(bar.symbol, bar.timeframe, bar.timestamp) for bar in source_bars]
    if len(keys) != len(set(keys)):
        raise DataValidationError("4h derivation refuses duplicate 1h bars")
    groups: defaultdict[tuple[datetime, int], list[MarketBar]] = defaultdict(list)
    for bar in source_bars:
        session = calendar.session_for_timestamp(bar.timestamp, metadata)
        if session is None:
            raise DataValidationError(
                f"1h bar {bar.timestamp.isoformat()} is outside an exchange session"
            )
        elapsed = bar.timestamp - session.market_open
        bucket_index = int(elapsed.total_seconds() // timedelta(hours=4).total_seconds())
        groups[(session.market_open, bucket_index)].append(bar)
    derived: list[MarketBar] = []
    for (session_open, bucket_index), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda bar: bar.timestamp)
        adjusted_values = [
            bar.adjusted_close for bar in ordered if bar.adjusted_close is not None
        ]
        derived.append(
            MarketBar(
                symbol=ordered[0].symbol,
                timeframe="4h",
                timestamp=(
                    session_open + timedelta(hours=4 * bucket_index)
                ).astimezone(timezone.utc),
                open=ordered[0].open,
                high=max(bar.high for bar in ordered),
                low=min(bar.low for bar in ordered),
                close=ordered[-1].close,
                volume=sum((bar.volume for bar in ordered), Decimal("0")),
                adjusted_close=(adjusted_values[-1] if adjusted_values else None),
                source=f"derived:{ordered[0].source}",
            )
        )
    return tuple(derived)
