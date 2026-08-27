"""Small deterministic builders shared by offline Lot 2 tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable

from trading_ai.backtesting.input import memory_dataset
from trading_ai.backtesting.models import BacktestDataset
from trading_ai.core.models import MarketBar
from trading_ai.data.models import CorporateAction, DataQualityReport, QualityStatus


START = datetime(2024, 1, 2, 21, tzinfo=timezone.utc)


def bar(
    index: int,
    *,
    symbol: str = "AAPL",
    timeframe: str = "1d",
    opening: str | Decimal = "100",
    high: str | Decimal | None = None,
    low: str | Decimal | None = None,
    close: str | Decimal | None = None,
    adjusted_close: str | Decimal | None = None,
    timestamp: datetime | None = None,
) -> MarketBar:
    open_value = Decimal(opening)
    high_value = Decimal(high) if high is not None else open_value + Decimal("2")
    low_value = Decimal(low) if low is not None else open_value - Decimal("2")
    close_value = Decimal(close) if close is not None else open_value + Decimal("1")
    adjusted_value = (
        Decimal(adjusted_close) if adjusted_close is not None else None
    )
    return MarketBar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp or START + timedelta(days=index),
        open=open_value,
        high=high_value,
        low=low_value,
        close=close_value,
        volume=Decimal("1000"),
        adjusted_close=adjusted_value,
        source="synthetic",
    )


def dataset(
    bars: Iterable[MarketBar],
    *,
    actions: Iterable[CorporateAction] = (),
    status: QualityStatus = QualityStatus.PASS,
    missing: int = 0,
    unexpected_gaps: int = 0,
    warnings: tuple[str, ...] = (),
) -> BacktestDataset:
    normalized = tuple(bars)
    first = normalized[0]
    report = DataQualityReport(
        symbol=first.symbol,
        timeframe=first.timeframe,
        row_count=len(normalized),
        duplicate_count=0,
        invalid_bar_count=0,
        missing_expected_bar_count=missing,
        unexpected_gap_count=unexpected_gaps,
        first_timestamp=normalized[0].timestamp,
        last_timestamp=normalized[-1].timestamp,
        timezone_valid=True,
        sorted=True,
        quality_status=status,
        warnings=warnings,
    )
    return memory_dataset(
        bars=normalized,
        corporate_actions=actions,
        quality_report=report,
    )
