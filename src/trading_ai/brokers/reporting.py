"""Expected-versus-observed Paper execution diagnostics."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_ai.brokers.models import (
    BrokerCommissionReport,
    BrokerExecution,
    BrokerOrderRecord,
    CommissionKnowledge,
    ExpectedObservedMetrics,
)


def _milliseconds(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def build_expected_observed_metrics(
    *,
    order: BrokerOrderRecord,
    executions: tuple[BrokerExecution, ...],
    commissions: tuple[BrokerCommissionReport, ...],
    decision_at: datetime | None,
    submitted_at: datetime | None,
    acknowledged_at: datetime | None,
    expected_fill_price: Decimal | None,
    estimated_slippage: Decimal | None,
    estimated_commission: Decimal | None,
    rejects: int = 0,
    cancels: int = 0,
    reconnects: int = 0,
    drifts: int = 0,
) -> ExpectedObservedMetrics:
    relevant = tuple(item for item in executions if item.client_order_key == order.client_order_key)
    total_quantity = sum((item.quantity for item in relevant), Decimal("0"))
    average = (
        sum((item.quantity * item.price for item in relevant), Decimal("0")) / total_quantity
        if total_quantity > Decimal("0") else None
    )
    observed_slippage = (
        abs(average - expected_fill_price)
        if average is not None and expected_fill_price is not None else None
    )
    known_commission = [
        item.amount
        for item in commissions
        if item.status is CommissionKnowledge.KNOWN and item.amount is not None
    ]
    final_fill_at = max((item.received_at for item in relevant), default=None)
    return ExpectedObservedMetrics(
        decision_to_submit_ms=_milliseconds(decision_at, submitted_at),
        submit_to_ack_ms=_milliseconds(submitted_at, acknowledged_at),
        submit_to_fill_ms=_milliseconds(submitted_at, final_fill_at),
        expected_fill_price=expected_fill_price,
        observed_average_fill_price=average,
        estimated_slippage=estimated_slippage,
        observed_slippage=observed_slippage,
        estimated_commission=estimated_commission,
        broker_commission=(sum(known_commission, Decimal("0")) if known_commission else None),
        rejects=rejects,
        cancels=cancels,
        partial_fills=max(0, len(relevant) - (1 if relevant else 0)),
        reconnects=reconnects,
        drifts=drifts,
    )
