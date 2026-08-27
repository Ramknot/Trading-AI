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
        symbol="AAPL",
        strength=0.8,
        generated_at=datetime.now(timezone.utc),
    )

    with pytest.raises(FrozenInstanceError):
        signal.strength = 0.2  # type: ignore[misc]
    with pytest.raises(ValueError, match="between -1 and 1"):
        Signal(
            signal_id="signal-2",
            symbol="AAPL",
            strength=1.1,
            generated_at=datetime.now(timezone.utc),
        )


def test_limit_order_requires_positive_price() -> None:
    with pytest.raises(ValueError, match="limit_price is required"):
        OrderRequest(
            order_id="limit-1",
            symbol="MSFT",
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
        Position("AAPL", Decimal("1"), Decimal("200")),
        Position("AAPL", Decimal("2"), Decimal("210")),
    )

    with pytest.raises(ValueError, match="unique symbols"):
        PortfolioSnapshot(
            as_of=datetime.now(timezone.utc),
            cash=Decimal("1000"),
            total_equity=Decimal("100000"),
            positions=positions,
        )


def test_risk_decision_cannot_increase_requested_quantity() -> None:
    with pytest.raises(ValueError, match="never exceed"):
        RiskDecision(
            decision_id="risk-too-large",
            order_id="order-1",
            status=RiskDecisionStatus.REDUCE,
            reason="invalid",
            risk_engine="test",
            requested_quantity=Decimal("1"),
            approved_quantity=Decimal("2"),
        )


def test_reduce_decision_requires_strictly_smaller_positive_quantity() -> None:
    with pytest.raises(ValueError, match="smaller positive"):
        RiskDecision(
            decision_id="risk-not-reduced",
            order_id="order-1",
            status=RiskDecisionStatus.REDUCE,
            reason="invalid",
            risk_engine="test",
            requested_quantity=Decimal("1"),
            approved_quantity=Decimal("1"),
        )
