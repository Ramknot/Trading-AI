"""Cash-only, long-only deterministic portfolio ledger for Balanced V1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from trading_ai.backtesting.exceptions import BacktestExecutionError
from trading_ai.backtesting.models import (
    EquityPoint,
    Fill,
    LedgerEntry,
    LedgerEntryType,
    PositionView,
)
from trading_ai.core.models import OrderSide, PortfolioSnapshot, Position
from trading_ai.data.models import Dividend, StockSplit


ZERO = Decimal("0")


@dataclass(slots=True)
class _PositionState:
    quantity: Decimal
    average_entry_price: Decimal
    entry_commission_remaining: Decimal
    entry_other_fees_remaining: Decimal = ZERO


class PortfolioLedger:
    """Every cash or position mutation is represented by a LedgerEntry."""

    def __init__(self, starting_cash: Decimal, *, allow_short: bool = False) -> None:
        if starting_cash <= ZERO:
            raise ValueError("starting_cash must be positive")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.allow_short = allow_short
        self._positions: dict[str, _PositionState] = {}
        self._entries: list[LedgerEntry] = []
        self.realized_pnl = ZERO
        self.dividend_income = ZERO
        self.commission_paid = ZERO
        self.spread_cost = ZERO
        self.slippage_cost = ZERO

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def quantity(self, symbol: str) -> Decimal:
        state = self._positions.get(symbol)
        return state.quantity if state is not None else ZERO

    def validate_fill(self, fill: Fill) -> str | None:
        if fill.side is OrderSide.BUY:
            required = fill.price * fill.quantity + fill.cash_fees_excluding_price_impact
            if required > self.cash:
                return "insufficient cash; leverage is disabled"
            return None
        held = self.quantity(fill.symbol)
        if fill.quantity > held:
            return "sell quantity exceeds the long position; short selling is disabled"
        return None

    def apply_fill(self, fill: Fill) -> LedgerEntry:
        rejection = self.validate_fill(fill)
        if rejection is not None:
            raise BacktestExecutionError(rejection)
        state = self._positions.get(fill.symbol)
        if fill.side is OrderSide.BUY:
            separate_fees = fill.cash_fees_excluding_price_impact
            cost = fill.price * fill.quantity + separate_fees
            self.cash -= cost
            if state is None:
                self._positions[fill.symbol] = _PositionState(
                    fill.quantity,
                    fill.price,
                    fill.commission,
                    separate_fees - fill.commission,
                )
            else:
                new_quantity = state.quantity + fill.quantity
                state.average_entry_price = (
                    state.average_entry_price * state.quantity
                    + fill.price * fill.quantity
                ) / new_quantity
                state.quantity = new_quantity
                state.entry_commission_remaining += fill.commission
                state.entry_other_fees_remaining += separate_fees - fill.commission
            cash_change = -cost
            quantity_change = fill.quantity
        else:
            if state is None:
                raise BacktestExecutionError("sell fill has no open position")
            separate_fees = fill.cash_fees_excluding_price_impact
            proceeds = fill.price * fill.quantity - separate_fees
            self.cash += proceeds
            entry_commission = (
                state.entry_commission_remaining
                * fill.quantity
                / state.quantity
            )
            entry_other_fees = (
                state.entry_other_fees_remaining
                * fill.quantity
                / state.quantity
            )
            self.realized_pnl += (
                fill.price - state.average_entry_price
            ) * fill.quantity - entry_commission - entry_other_fees - separate_fees
            state.entry_commission_remaining -= entry_commission
            state.entry_other_fees_remaining -= entry_other_fees
            state.quantity -= fill.quantity
            if state.quantity == ZERO:
                del self._positions[fill.symbol]
            cash_change = proceeds
            quantity_change = -fill.quantity
        if self.cash < ZERO:
            raise BacktestExecutionError("ledger cash became negative")
        self.commission_paid += fill.commission
        self.spread_cost += fill.spread_cost
        self.slippage_cost += fill.slippage_cost
        entry = LedgerEntry(
            entry_id=f"ledger-{len(self._entries) + 1:06d}",
            timestamp=fill.timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=fill.symbol,
            cash_change=cash_change,
            quantity_change=quantity_change,
            amount=fill.price * fill.quantity,
            reference_id=fill.fill_id,
            message=f"simulated {fill.side.value} fill",
        )
        self._entries.append(entry)
        return entry

    def apply_dividend(self, action: Dividend) -> LedgerEntry | None:
        quantity = self.quantity(action.symbol)
        if quantity <= ZERO:
            return None
        credit = quantity * action.value
        self.cash += credit
        self.dividend_income += credit
        entry = LedgerEntry(
            entry_id=f"ledger-{len(self._entries) + 1:06d}",
            timestamp=action.timestamp,
            entry_type=LedgerEntryType.DIVIDEND,
            symbol=action.symbol,
            cash_change=credit,
            quantity_change=ZERO,
            amount=credit,
            reference_id=f"dividend:{action.source}:{action.timestamp.isoformat()}",
            message="explicit raw-price dividend credit",
        )
        self._entries.append(entry)
        return entry

    def apply_split(
        self, action: StockSplit, last_prices: dict[str, Decimal]
    ) -> LedgerEntry | None:
        state = self._positions.get(action.symbol)
        if state is None or state.quantity <= ZERO:
            return None
        old_quantity = state.quantity
        state.quantity *= action.value
        state.average_entry_price /= action.value
        if action.symbol in last_prices:
            last_prices[action.symbol] /= action.value
        entry = LedgerEntry(
            entry_id=f"ledger-{len(self._entries) + 1:06d}",
            timestamp=action.timestamp,
            entry_type=LedgerEntryType.SPLIT,
            symbol=action.symbol,
            cash_change=ZERO,
            quantity_change=state.quantity - old_quantity,
            amount=action.value,
            reference_id=f"split:{action.source}:{action.timestamp.isoformat()}",
            message="split adjusted quantity and average entry without PnL",
        )
        self._entries.append(entry)
        return entry

    def position_views(self, prices: Mapping[str, Decimal]) -> tuple[PositionView, ...]:
        views: list[PositionView] = []
        for symbol, state in sorted(self._positions.items()):
            market_price = prices.get(symbol, state.average_entry_price)
            market_value = state.quantity * market_price
            views.append(
                PositionView(
                    symbol=symbol,
                    quantity=state.quantity,
                    average_entry_price=state.average_entry_price,
                    market_price=market_price,
                    market_value=market_value,
                    unrealized_pnl=(
                        market_price - state.average_entry_price
                    )
                    * state.quantity
                    - state.entry_commission_remaining
                    - state.entry_other_fees_remaining,
                )
            )
        return tuple(views)

    def snapshot(
        self, as_of: datetime, prices: Mapping[str, Decimal]
    ) -> PortfolioSnapshot:
        views = self.position_views(prices)
        return PortfolioSnapshot(
            as_of=as_of,
            cash=self.cash,
            total_equity=self.cash + sum(
                (view.market_value for view in views), ZERO
            ),
            positions=tuple(
                Position(
                    symbol=view.symbol,
                    quantity=view.quantity,
                    average_price=view.average_entry_price,
                )
                for view in views
            ),
        )

    def equity_point(
        self, timestamp: datetime, prices: Mapping[str, Decimal]
    ) -> EquityPoint:
        views = self.position_views(prices)
        positions_value = sum((view.market_value for view in views), ZERO)
        unrealized = sum((view.unrealized_pnl for view in views), ZERO)
        return EquityPoint(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            equity=self.cash + positions_value,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
        )
