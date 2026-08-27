"""Shared deterministic, offline fixtures for the project test suite."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_ai.core.models import (
    ExecutionEnvironment,
    OrderRequest,
    OrderSide,
    PortfolioSnapshot,
    TradingContext,
    TradingProfileName,
)
from trading_ai.data.models import (
    Dividend,
    InstrumentMetadata,
    ProviderBar,
    StockSplit,
)
from trading_ai.data.providers import FakeDataProvider


@pytest.fixture
def paper_context() -> TradingContext:
    return TradingContext(
        environment=ExecutionEnvironment.PAPER,
        profile=TradingProfileName.BALANCED,
    )


@pytest.fixture
def portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        cash=Decimal("10000"),
        total_equity=Decimal("10000"),
    )


@pytest.fixture
def order() -> OrderRequest:
    return OrderRequest(
        order_id="order-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
    )


@pytest.fixture
def market_start() -> datetime:
    return datetime(2024, 7, 1, tzinfo=timezone.utc)


@pytest.fixture
def market_end() -> datetime:
    return datetime(2024, 7, 3, tzinfo=timezone.utc)


@pytest.fixture
def aapl_metadata() -> InstrumentMetadata:
    return InstrumentMetadata(
        symbol="AAPL",
        exchange="NMS",
        exchange_timezone="America/New_York",
        calendar="NYSE",
        source="fake",
        currency="USD",
    )


@pytest.fixture
def paris_metadata() -> InstrumentMetadata:
    return InstrumentMetadata(
        symbol="MC.PA",
        exchange="PAR",
        exchange_timezone="Europe/Paris",
        calendar="XPAR",
        source="fake",
        currency="EUR",
    )


@pytest.fixture
def aapl_hourly_rows() -> tuple[ProviderBar, ...]:
    rows: list[ProviderBar] = []
    for session_open in (
        datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
        datetime(2024, 7, 2, 13, 30, tzinfo=timezone.utc),
    ):
        for index in range(7):
            opening = Decimal("100") + index
            rows.append(
                ProviderBar(
                    symbol="AAPL",
                    timeframe="1h",
                    timestamp=session_open + timedelta(hours=index),
                    open=opening,
                    high=opening + Decimal("2"),
                    low=opening - Decimal("1"),
                    close=opening + Decimal("1"),
                    volume=Decimal(10 * (index + 1)),
                    adjusted_close=opening + Decimal("0.5"),
                    source="fake",
                )
            )
    return tuple(rows)


@pytest.fixture
def aapl_daily_rows() -> tuple[ProviderBar, ...]:
    return tuple(
        ProviderBar(
            symbol="AAPL",
            timeframe="1d",
            timestamp=datetime(2024, 7, day, 4, tzinfo=timezone.utc),
            open=Decimal("100") + day,
            high=Decimal("103") + day,
            low=Decimal("99") + day,
            close=Decimal("102") + day,
            volume=Decimal("1000") * day,
            adjusted_close=Decimal("101.5") + day,
            source="fake",
        )
        for day in (1, 2)
    )


@pytest.fixture
def aapl_actions() -> tuple[Dividend | StockSplit, ...]:
    return (
        Dividend(
            symbol="AAPL",
            timestamp=datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
            value=Decimal("0.25"),
            source="fake",
        ),
        StockSplit(
            symbol="AAPL",
            timestamp=datetime(2024, 7, 2, 13, 30, tzinfo=timezone.utc),
            value=Decimal("4"),
            source="fake",
        ),
    )


@pytest.fixture
def fake_data_provider(
    aapl_metadata: InstrumentMetadata,
    aapl_hourly_rows: tuple[ProviderBar, ...],
    aapl_daily_rows: tuple[ProviderBar, ...],
    aapl_actions: tuple[Dividend | StockSplit, ...],
) -> FakeDataProvider:
    return FakeDataProvider(
        datasets={
            ("AAPL", "1h"): aapl_hourly_rows,
            ("AAPL", "1d"): aapl_daily_rows,
        },
        metadata_by_symbol={"AAPL": aapl_metadata},
        actions_by_symbol={"AAPL": aapl_actions},
    )
