"""Shared deterministic Lot 0 fixtures."""

from datetime import datetime, timezone
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
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
    )
