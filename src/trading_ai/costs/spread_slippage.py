"""Shared economic representation of existing Backtester spread/slippage rules."""

from decimal import Decimal

from trading_ai.costs.models import BPS, ZERO


def estimated_spread_slippage(
    notional: Decimal, spread_bps: Decimal, slippage_bps: Decimal
) -> tuple[Decimal, Decimal]:
    """Match BarExecutionModel exactly; the ledger must not debit these twice."""

    if notional <= ZERO:
        raise ValueError("spread/slippage estimate requires positive notional")
    if spread_bps < ZERO or slippage_bps < ZERO:
        raise ValueError("spread/slippage bps must not be negative")
    return notional * spread_bps / BPS, notional * slippage_bps / BPS
