"""Temporary technical sizing for Lot 3 baselines, not a PortfolioEngine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from trading_ai.core.models import PortfolioSnapshot


@dataclass(frozen=True, slots=True)
class BaselineSizer:
    """Allocate a fixed equity fraction while never requesting more cash.

    Fractional quantities are supported at a deterministic six-decimal step.
    The Backtesting Portfolio Ledger remains the authority for cash and short
    constraints. This helper is intentionally temporary until Lot 7.
    """

    allocation_fraction: Decimal
    quantity_step: Decimal = Decimal("0.000001")

    def __post_init__(self) -> None:
        if (
            not self.allocation_fraction.is_finite()
            or not Decimal("0") < self.allocation_fraction <= Decimal("1")
        ):
            raise ValueError("allocation_fraction must be in (0, 1]")
        if not self.quantity_step.is_finite() or self.quantity_step <= Decimal("0"):
            raise ValueError("quantity_step must be positive and finite")

    def entry_quantity(
        self,
        portfolio: PortfolioSnapshot,
        price: Decimal,
        *,
        slots: int = 1,
        available_cash: Decimal | None = None,
    ) -> Decimal | None:
        if not price.is_finite() or price <= Decimal("0"):
            raise ValueError("price must be positive and finite")
        if slots < 1:
            raise ValueError("slots must be positive")
        cash = portfolio.cash if available_cash is None else available_cash
        if cash <= Decimal("0") or portfolio.total_equity <= Decimal("0"):
            return None
        target = portfolio.total_equity * self.allocation_fraction / Decimal(slots)
        budget = min(cash, target)
        quantity = (budget / price).quantize(self.quantity_step, rounding=ROUND_DOWN)
        return quantity if quantity >= self.quantity_step else None
