"""Explicit exchange/pass-through fee calculations kept separate from commission."""

from decimal import Decimal

from trading_ai.costs.models import BPS, TariffProfile, ZERO


def exchange_fee_amount(
    tariff: TariffProfile, *, quantity: Decimal, notional: Decimal
) -> Decimal:
    if quantity <= ZERO or notional <= ZERO:
        raise ValueError("exchange fees require positive quantity and notional")
    if tariff.exchange_fees_included_in_commission:
        return ZERO
    return (
        quantity * tariff.exchange_fee_per_unit
        + notional * tariff.exchange_fee_bps / BPS
    )
