from datetime import timedelta
from decimal import Decimal

import pytest

from backtest_support import START
from trading_ai.backtesting.exceptions import BacktestExecutionError
from trading_ai.backtesting.models import Fill, LedgerEntryType
from trading_ai.backtesting.portfolio import PortfolioLedger
from trading_ai.backtesting.trades import reconstruct_trades
from trading_ai.core.models import OrderSide
from trading_ai.data.models import Dividend, StockSplit


def _fill(
    number: int,
    *,
    side: OrderSide,
    quantity: str,
    price: str,
    commission: str = "0",
) -> Fill:
    return Fill(
        fill_id=f"fill-{number}",
        order_id=f"order-{number}",
        symbol="AAPL",
        side=side,
        quantity=Decimal(quantity),
        reference_price=Decimal(price),
        price=Decimal(price),
        timestamp=START + timedelta(days=number),
        commission=Decimal(commission),
        spread_cost=Decimal("0"),
        slippage_cost=Decimal("0"),
    )


def test_ledger_buy_average_partial_sell_cash_and_pnl() -> None:
    ledger = PortfolioLedger(Decimal("10000"))
    ledger.apply_fill(_fill(1, side=OrderSide.BUY, quantity="10", price="100"))
    ledger.apply_fill(_fill(2, side=OrderSide.BUY, quantity="10", price="120"))

    before_sale = ledger.position_views({"AAPL": Decimal("130")})[0]
    ledger.apply_fill(
        _fill(
            3,
            side=OrderSide.SELL,
            quantity="5",
            price="130",
            commission="1",
        )
    )
    after_sale = ledger.position_views({"AAPL": Decimal("130")})[0]

    assert before_sale.average_entry_price == Decimal("110")
    assert before_sale.unrealized_pnl == Decimal("400")
    assert after_sale.quantity == Decimal("15")
    assert after_sale.average_entry_price == Decimal("110")
    assert ledger.cash == Decimal("8449")
    assert ledger.realized_pnl == Decimal("99")
    assert len(ledger.entries) == 3
    assert all(entry.entry_type is LedgerEntryType.FILL for entry in ledger.entries)


def test_ledger_rejects_insufficient_cash_without_mutation() -> None:
    ledger = PortfolioLedger(Decimal("100"))
    fill = _fill(1, side=OrderSide.BUY, quantity="2", price="60")

    assert ledger.validate_fill(fill) == "insufficient cash; leverage is disabled"
    with pytest.raises(BacktestExecutionError, match="insufficient cash"):
        ledger.apply_fill(fill)
    assert ledger.cash == Decimal("100")
    assert ledger.entries == ()


def test_commissions_are_reflected_in_realized_and_unrealized_pnl() -> None:
    ledger = PortfolioLedger(Decimal("2000"))
    ledger.apply_fill(
        _fill(
            1,
            side=OrderSide.BUY,
            quantity="10",
            price="100",
            commission="10",
        )
    )

    open_point = ledger.equity_point(
        START + timedelta(days=1), {"AAPL": Decimal("100")}
    )
    ledger.apply_fill(
        _fill(
            2,
            side=OrderSide.SELL,
            quantity="10",
            price="100",
            commission="5",
        )
    )
    closed_point = ledger.equity_point(
        START + timedelta(days=2), {"AAPL": Decimal("100")}
    )

    assert open_point.unrealized_pnl == Decimal("-10")
    assert open_point.equity == Decimal("1990")
    assert ledger.realized_pnl == Decimal("-15")
    assert closed_point.equity == Decimal("1985")


def test_ledger_rejects_short_and_oversell() -> None:
    ledger = PortfolioLedger(Decimal("1000"))

    with pytest.raises(BacktestExecutionError, match="short selling is disabled"):
        ledger.apply_fill(
            _fill(1, side=OrderSide.SELL, quantity="1", price="100")
        )
    ledger.apply_fill(_fill(2, side=OrderSide.BUY, quantity="1", price="100"))
    with pytest.raises(BacktestExecutionError, match="short selling is disabled"):
        ledger.apply_fill(
            _fill(3, side=OrderSide.SELL, quantity="2", price="100")
        )
    assert ledger.quantity("AAPL") == Decimal("1")


def test_dividend_credit_is_explicit() -> None:
    ledger = PortfolioLedger(Decimal("1000"))
    ledger.apply_fill(_fill(1, side=OrderSide.BUY, quantity="10", price="50"))
    action = Dividend(
        symbol="AAPL",
        timestamp=START + timedelta(days=2),
        value=Decimal("0.50"),
        source="synthetic",
    )

    entry = ledger.apply_dividend(action)

    assert entry is not None
    assert entry.entry_type is LedgerEntryType.DIVIDEND
    assert entry.cash_change == Decimal("5.00")
    assert ledger.cash == Decimal("505.00")
    assert ledger.dividend_income == Decimal("5.00")


def test_split_two_for_one_preserves_economic_value() -> None:
    ledger = PortfolioLedger(Decimal("2000"))
    ledger.apply_fill(_fill(1, side=OrderSide.BUY, quantity="10", price="100"))
    prices = {"AAPL": Decimal("100")}
    before = ledger.equity_point(START + timedelta(days=1), prices)
    split = StockSplit(
        symbol="AAPL",
        timestamp=START + timedelta(days=2),
        value=Decimal("2"),
        source="synthetic",
    )

    entry = ledger.apply_split(split, prices)
    after = ledger.equity_point(START + timedelta(days=2), prices)
    position = ledger.position_views(prices)[0]

    assert entry is not None
    assert entry.entry_type is LedgerEntryType.SPLIT
    assert position.quantity == Decimal("20")
    assert position.average_entry_price == Decimal("50")
    assert prices["AAPL"] == Decimal("50")
    assert after.equity == before.equity


def test_fifo_trade_reconstruction_preserves_partial_exits_and_fees() -> None:
    fills = (
        _fill(
            1,
            side=OrderSide.BUY,
            quantity="10",
            price="100",
            commission="10",
        ),
        _fill(
            2,
            side=OrderSide.SELL,
            quantity="4",
            price="110",
            commission="4",
        ),
        _fill(
            3,
            side=OrderSide.SELL,
            quantity="6",
            price="120",
            commission="6",
        ),
    )

    trades = reconstruct_trades(fills, ())

    assert [trade.quantity for trade in trades] == [Decimal("4"), Decimal("6")]
    assert [trade.gross_pnl for trade in trades] == [
        Decimal("40"),
        Decimal("120"),
    ]
    assert [trade.fees for trade in trades] == [Decimal("8.0"), Decimal("12.0")]
    assert [trade.net_pnl for trade in trades] == [Decimal("32.0"), Decimal("108.0")]


def test_trade_reconstruction_adjusts_open_lot_for_split() -> None:
    buy = _fill(1, side=OrderSide.BUY, quantity="10", price="100")
    sell = _fill(3, side=OrderSide.SELL, quantity="20", price="60")
    split = StockSplit(
        symbol="AAPL",
        timestamp=START + timedelta(days=2),
        value=Decimal("2"),
        source="synthetic",
    )

    trades = reconstruct_trades((buy, sell), (split,))

    assert trades[0].quantity == Decimal("20")
    assert trades[0].entry_price == Decimal("50")
    assert trades[0].gross_pnl == Decimal("200")
