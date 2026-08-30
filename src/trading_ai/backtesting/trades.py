"""FIFO closed-trade reconstruction, including splits and partial exits."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_ai.backtesting.exceptions import BacktestExecutionError
from trading_ai.backtesting.models import Fill, Trade
from trading_ai.core.models import OrderSide
from trading_ai.data.models import CorporateAction, StockSplit


ZERO = Decimal("0")


@dataclass(slots=True)
class _OpenLot:
    fill: Fill
    quantity: Decimal
    entry_price: Decimal
    commission_remaining: Decimal
    spread_remaining: Decimal
    slippage_remaining: Decimal
    exchange_fees_remaining: Decimal
    transaction_tax_remaining: Decimal
    fx_cost_remaining: Decimal
    financing_cost_remaining: Decimal
    other_variable_cost_remaining: Decimal


def reconstruct_trades(
    fills: tuple[Fill, ...], corporate_actions: tuple[CorporateAction, ...]
) -> tuple[Trade, ...]:
    """Pair long fills FIFO; V1 fills are complete but exits may be partial."""

    events: list[tuple[object, int, object]] = []
    events.extend((action.timestamp, 0, action) for action in corporate_actions)
    events.extend((fill.timestamp, 1, fill) for fill in fills)
    events.sort(
        key=lambda item: (
            item[0],
            item[1],
            getattr(item[2], "symbol", ""),
            getattr(item[2], "fill_id", ""),
        )
    )
    lots: dict[str, list[_OpenLot]] = {}
    trades: list[Trade] = []
    for _, event_priority, event in events:
        if event_priority == 0:
            if isinstance(event, StockSplit):
                for lot in lots.get(event.symbol, []):
                    lot.quantity *= event.value
                    lot.entry_price /= event.value
            continue
        if not isinstance(event, Fill):
            continue
        fill = event
        if fill.side is OrderSide.BUY:
            lots.setdefault(fill.symbol, []).append(
                _OpenLot(
                    fill=fill,
                    quantity=fill.quantity,
                    entry_price=fill.price,
                    commission_remaining=fill.commission,
                    spread_remaining=fill.spread_cost,
                    slippage_remaining=fill.slippage_cost,
                    exchange_fees_remaining=fill.exchange_fees,
                    transaction_tax_remaining=fill.transaction_tax,
                    fx_cost_remaining=fill.fx_cost,
                    financing_cost_remaining=fill.financing_cost,
                    other_variable_cost_remaining=fill.other_variable_cost,
                )
            )
            continue
        remaining = fill.quantity
        symbol_lots = lots.get(fill.symbol, [])
        exit_commission_remaining = fill.commission
        exit_spread_remaining = fill.spread_cost
        exit_slippage_remaining = fill.slippage_cost
        exit_exchange_remaining = fill.exchange_fees
        exit_tax_remaining = fill.transaction_tax
        exit_fx_remaining = fill.fx_cost
        exit_financing_remaining = fill.financing_cost
        exit_other_remaining = fill.other_variable_cost
        while remaining > ZERO:
            if not symbol_lots:
                raise BacktestExecutionError(
                    "sell fill cannot be reconstructed without an entry lot"
                )
            lot = symbol_lots[0]
            closed = min(remaining, lot.quantity)
            entry_fraction = closed / lot.quantity
            exit_fraction = closed / remaining
            entry_commission = lot.commission_remaining * entry_fraction
            entry_spread = lot.spread_remaining * entry_fraction
            entry_slippage = lot.slippage_remaining * entry_fraction
            exit_commission = exit_commission_remaining * exit_fraction
            exit_spread = exit_spread_remaining * exit_fraction
            exit_slippage = exit_slippage_remaining * exit_fraction
            entry_exchange = lot.exchange_fees_remaining * entry_fraction
            entry_tax = lot.transaction_tax_remaining * entry_fraction
            entry_fx = lot.fx_cost_remaining * entry_fraction
            entry_financing = lot.financing_cost_remaining * entry_fraction
            entry_other = lot.other_variable_cost_remaining * entry_fraction
            exit_exchange = exit_exchange_remaining * exit_fraction
            exit_tax = exit_tax_remaining * exit_fraction
            exit_fx = exit_fx_remaining * exit_fraction
            exit_financing = exit_financing_remaining * exit_fraction
            exit_other = exit_other_remaining * exit_fraction
            gross = (fill.price - lot.entry_price) * closed
            exchange = entry_exchange + exit_exchange
            tax = entry_tax + exit_tax
            fx = entry_fx + exit_fx
            financing = entry_financing + exit_financing
            other = entry_other + exit_other
            fees = (
                entry_commission + exit_commission + exchange + tax + fx
                + financing + other
            )
            net = gross - fees
            capital = lot.entry_price * closed
            trades.append(
                Trade(
                    trade_id=f"trade-{len(trades) + 1:06d}",
                    symbol=fill.symbol,
                    entry_time=lot.fill.timestamp,
                    exit_time=fill.timestamp,
                    entry_price=lot.entry_price,
                    exit_price=fill.price,
                    quantity=closed,
                    gross_pnl=gross,
                    fees=fees,
                    spread_cost=entry_spread + exit_spread,
                    slippage_cost=entry_slippage + exit_slippage,
                    net_pnl=net,
                    return_pct=(net / capital if capital > ZERO else ZERO),
                    holding_period_seconds=(
                        fill.timestamp - lot.fill.timestamp
                    ).total_seconds(),
                    exchange_fees=exchange,
                    transaction_tax=tax,
                    fx_cost=fx,
                    financing_cost=financing,
                    other_variable_cost=other,
                )
            )
            lot.quantity -= closed
            lot.commission_remaining -= entry_commission
            lot.spread_remaining -= entry_spread
            lot.slippage_remaining -= entry_slippage
            lot.exchange_fees_remaining -= entry_exchange
            lot.transaction_tax_remaining -= entry_tax
            lot.fx_cost_remaining -= entry_fx
            lot.financing_cost_remaining -= entry_financing
            lot.other_variable_cost_remaining -= entry_other
            exit_commission_remaining -= exit_commission
            exit_spread_remaining -= exit_spread
            exit_slippage_remaining -= exit_slippage
            exit_exchange_remaining -= exit_exchange
            exit_tax_remaining -= exit_tax
            exit_fx_remaining -= exit_fx
            exit_financing_remaining -= exit_financing
            exit_other_remaining -= exit_other
            remaining -= closed
            if lot.quantity == ZERO:
                symbol_lots.pop(0)
    return tuple(trades)
