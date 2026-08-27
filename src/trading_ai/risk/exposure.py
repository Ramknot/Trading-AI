"""Broker-neutral current and projected exposure calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_ai.core.models import OrderSide, PortfolioSnapshot


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class ExposureProjection:
    gross_value_before: Decimal
    gross_value_after: Decimal
    position_value_before: Decimal
    position_value_after: Decimal

    def gross_before(self, equity: Decimal) -> float:
        return float(self.gross_value_before / equity) if equity > ZERO else 1.0

    def gross_after(self, equity: Decimal) -> float:
        return float(self.gross_value_after / equity) if equity > ZERO else 1.0

    def position_before(self, equity: Decimal) -> float:
        return float(self.position_value_before / equity) if equity > ZERO else 1.0

    def position_after(self, equity: Decimal) -> float:
        return float(self.position_value_after / equity) if equity > ZERO else 1.0


def position_values(
    portfolio: PortfolioSnapshot,
    market_prices: tuple[tuple[str, Decimal], ...],
) -> dict[str, Decimal]:
    prices = dict(market_prices)
    values: dict[str, Decimal] = {}
    for position in portfolio.positions:
        price = prices.get(position.symbol)
        if price is None:
            raise ValueError(f"missing current market price for {position.symbol}")
        values[position.symbol] = abs(position.quantity * price)
    return values


def project_exposure(
    *,
    portfolio: PortfolioSnapshot,
    market_prices: tuple[tuple[str, Decimal], ...],
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    expected_price: Decimal,
) -> ExposureProjection:
    values = position_values(portfolio, market_prices)
    position = next(
        (item for item in portfolio.positions if item.symbol == symbol), None
    )
    held = position.quantity if position is not None else ZERO
    projected_quantity = held + quantity if side is OrderSide.BUY else held - quantity
    before = values.get(symbol, ZERO)
    after = abs(projected_quantity * expected_price)
    gross_before = sum(values.values(), ZERO)
    gross_after = gross_before - before + after
    return ExposureProjection(gross_before, gross_after, before, after)


def max_increment_by_portfolio_exposure(
    *,
    equity: Decimal,
    gross_value_before: Decimal,
    limit: Decimal,
    price: Decimal,
) -> Decimal:
    return max(ZERO, equity * limit - gross_value_before) / price


def max_increment_by_position_exposure(
    *,
    equity: Decimal,
    position_value_before: Decimal,
    limit: Decimal,
    price: Decimal,
) -> Decimal:
    return max(ZERO, equity * limit - position_value_before) / price
