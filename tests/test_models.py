from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trading_ai.core.models import (
    OrderRequest,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    Position,
    RiskApprovedOrder,
    RiskDecision,
    RiskDecisionStatus,
    Signal,
)


def test_signal_is_validated_and_immutable() -> None:
    signal = Signal(
        signal_id="signal-1",
        symbol="BTC-USD",
        strength=0.8,
        generated_at=datetime.now(timezone.utc),
    )

    with pytest.raises(FrozenInstanceError):
        signal.strength = 0.2  # type: ignore[misc]
    with pytest.raises(ValueError, match="between -1 and 1"):
        Signal(
            signal_id="signal-2",
            symbol="BTC-USD",
            strength=1.1,
            generated_at=datetime.now(timezone.utc),
        )


def test_limit_order_requires_positive_price() -> None:
    with pytest.raises(ValueError, match="limit_price is required"):
        OrderRequest(
            order_id="limit-1",
            symbol="ETH-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            order_type=OrderType.LIMIT,
        )


def test_rejected_decision_cannot_form_broker_envelope(order) -> None:
    rejected = RiskDecision(
        decision_id="risk-1",
        order_id=order.order_id,
        status=RiskDecisionStatus.REJECT,
        reason="rejected",
        risk_engine="test",
    )

    with pytest.raises(ValueError, match="only approved"):
        RiskApprovedOrder(order=order, risk_decision=rejected)


def test_portfolio_rejects_duplicate_symbols() -> None:
    positions = (
        Position("BTC-USD", Decimal("1"), Decimal("50000")),
        Position("BTC-USD", Decimal("2"), Decimal("51000")),
    )

    with pytest.raises(ValueError, match="unique symbols"):
        PortfolioSnapshot(
            as_of=datetime.now(timezone.utc),
            cash=Decimal("1000"),
            total_equity=Decimal("100000"),
            positions=positions,
        )
