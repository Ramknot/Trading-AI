"""Broker-neutral fixed, proportional, capped, and tiered commissions."""

from __future__ import annotations

from decimal import Decimal

from trading_ai.costs.models import BPS, TariffProfile, ZERO


def commission_amount(
    tariff: TariffProfile,
    *,
    quantity: Decimal,
    notional: Decimal,
    monthly_volume_before: Decimal = ZERO,
) -> Decimal:
    """Calculate one order's commission entirely from a dated tariff profile."""

    if quantity <= ZERO or notional <= ZERO:
        raise ValueError("commission requires positive quantity and notional")
    variable_per_unit = quantity * tariff.per_unit
    if tariff.tiers:
        # IBKR publishes cumulative monthly tiers on a marginal basis. An
        # order crossing a boundary therefore pays each rate only on the
        # shares falling inside that tier; applying one rate to the whole
        # order would misstate both estimate and actual cost.
        remaining = quantity
        cumulative = monthly_volume_before
        variable_per_unit = ZERO
        for tier in tariff.tiers:
            if remaining <= ZERO:
                break
            if tier.up_to_monthly_quantity is None:
                in_tier = remaining
            else:
                capacity = max(ZERO, tier.up_to_monthly_quantity - cumulative)
                if capacity <= ZERO:
                    continue
                in_tier = min(remaining, capacity)
            variable_per_unit += in_tier * tier.per_unit
            remaining -= in_tier
            cumulative += in_tier
        if remaining > ZERO:
            raise ValueError("commission tiers do not cover the requested volume")
    raw = (
        tariff.fixed_per_order
        + variable_per_unit
        + notional * tariff.proportional_bps / BPS
    )
    amount = max(raw, tariff.minimum_per_order)
    caps = []
    if tariff.maximum_per_order is not None:
        caps.append(tariff.maximum_per_order)
    if tariff.maximum_notional_fraction is not None:
        caps.append(notional * tariff.maximum_notional_fraction)
    if caps:
        amount = min(amount, *caps)
    return amount
