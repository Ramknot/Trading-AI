"""Pure quantity caps; Risk may reduce but can never increase a proposal."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN


ZERO = Decimal("0")


def floor_quantity(quantity: Decimal, step: Decimal) -> Decimal:
    if quantity <= ZERO:
        return ZERO
    return (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step


def capped_quantity(
    requested: Decimal,
    caps: tuple[Decimal, ...],
    *,
    step: Decimal,
) -> Decimal:
    """Apply deterministic non-negative caps and a configured quantity step."""

    if requested <= ZERO:
        raise ValueError("requested quantity must be positive")
    applicable = (requested, *(max(ZERO, cap) for cap in caps))
    approved = floor_quantity(min(applicable), step)
    return min(requested, approved)


def quantity_by_trade_risk(
    *,
    equity: Decimal,
    risk_fraction: Decimal,
    risk_per_share: Decimal,
) -> Decimal:
    if risk_per_share <= ZERO:
        raise ValueError("risk_per_share must be positive")
    return equity * risk_fraction / risk_per_share
