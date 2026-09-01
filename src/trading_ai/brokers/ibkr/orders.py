"""Exact Risk-approved order conversion to the narrow Lot 9 IBKR subset."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_ai.brokers.ibkr.contracts import IBKRContractCandidate
from trading_ai.core.models import OrderSide, OrderType, RiskApprovedOrder


@dataclass(frozen=True, slots=True)
class IBKROrderSpec:
    action: str
    order_type: str
    total_quantity: Decimal
    tif: str
    transmit: bool
    limit_price: Decimal | None


def to_ibkr_order(
    approved: RiskApprovedOrder, *, tif: str, transmit: bool
) -> IBKROrderSpec:
    order = approved.order
    if order.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
        raise ValueError("Lot 9 IBKR adapter permits MARKET and LIMIT only")
    if approved.risk_decision.approved_quantity is not None and (
        order.quantity != approved.risk_decision.approved_quantity
    ):
        raise ValueError("broker quantity must equal Risk-approved quantity")
    return IBKROrderSpec(
        action="BUY" if order.side is OrderSide.BUY else "SELL",
        order_type="MKT" if order.order_type is OrderType.MARKET else "LMT",
        total_quantity=order.quantity,
        tif=tif,
        transmit=transmit,
        limit_price=order.limit_price,
    )


def contract_payload(contract: IBKRContractCandidate) -> dict[str, object]:
    return {
        "conId": contract.con_id,
        "symbol": contract.symbol,
        "secType": contract.sec_type,
        "exchange": contract.exchange,
        "primaryExchange": contract.primary_exchange,
        "currency": contract.currency,
        "localSymbol": contract.local_symbol,
    }
