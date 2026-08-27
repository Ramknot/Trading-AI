from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from backtest_support import START, bar, dataset
from trading_ai.backtesting.execution import (
    BarExecutionModel,
    ConfigurableCommissionModel,
)
from trading_ai.backtesting.models import (
    BacktestConfig,
    BacktestOrder,
    CommissionConfig,
    OrderIntent,
    OrderStatus,
    StrategyContext,
)
from trading_ai.core.config import load_runtime_settings
from trading_ai.core.models import OrderSide, OrderType, PortfolioSnapshot


def _order(
    *,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
) -> BacktestOrder:
    return BacktestOrder(
        order_id="order-1",
        symbol="AAPL",
        timeframe="1d",
        side=side,
        quantity=Decimal("10"),
        order_type=order_type,
        created_at=START,
        limit_price=limit_price,
    )


def test_backtest_config_is_immutable_and_validated() -> None:
    config = BacktestConfig(starting_cash=Decimal("1000"))

    with pytest.raises(FrozenInstanceError):
        config.starting_cash = Decimal("1")  # type: ignore[misc]
    with pytest.raises(ValueError, match="starting_cash"):
        BacktestConfig(starting_cash=Decimal("0"))
    with pytest.raises(ValueError, match="combined"):
        BacktestConfig(spread_bps=Decimal("6000"), slippage_bps=Decimal("4000"))


def test_order_intent_validates_quantity_limit_and_reserved_types() -> None:
    with pytest.raises(ValueError, match="quantity"):
        OrderIntent("AAPL", OrderSide.BUY, Decimal("0"))
    with pytest.raises(ValueError, match="limit_price"):
        OrderIntent(
            "AAPL", OrderSide.BUY, Decimal("1"), order_type=OrderType.LIMIT
        )
    with pytest.raises(ValueError, match="reserved"):
        OrderIntent(
            "AAPL", OrderSide.BUY, Decimal("1"), order_type=OrderType.STOP
        )


def test_strategy_context_rejects_future_history_and_has_no_future_api(
    paper_context,
) -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    current = bar(0)
    future = bar(1)
    portfolio = PortfolioSnapshot(
        as_of=current.timestamp,
        cash=Decimal("1000"),
        total_equity=Decimal("1000"),
    )

    with pytest.raises(ValueError, match="end with current_bar"):
        StrategyContext(
            current_time=current.timestamp,
            current_bar=current,
            history=(current, future),
            portfolio=portfolio,
            trading_context=paper_context,
            profile=settings.profile,
        )
    with pytest.raises(ValueError, match="future bars"):
        StrategyContext(
            current_time=current.timestamp,
            current_bar=current,
            history=(future, current),
            portfolio=portfolio,
            trading_context=paper_context,
            profile=settings.profile,
        )
    context = StrategyContext(
        current_time=current.timestamp,
        current_bar=current,
        history=(current,),
        portfolio=portfolio,
        trading_context=paper_context,
        profile=settings.profile,
    )
    with pytest.raises(AttributeError):
        getattr(context, "future_bar")


def test_backtest_dataset_requires_utc_normalized_bars() -> None:
    non_utc = timezone(timedelta(hours=2))
    local_bar = bar(0, timestamp=datetime(2024, 1, 2, 23, tzinfo=non_utc))

    with pytest.raises(ValueError, match="must use UTC"):
        dataset((local_bar,))


def test_market_order_waits_for_strictly_later_bar_and_uses_next_open() -> None:
    model = BarExecutionModel(BacktestConfig(starting_cash=Decimal("10000")))
    order = _order()

    assert model.try_fill(order, bar(0), "fill-1") is None
    fill = model.try_fill(order, bar(1, opening="105"), "fill-1")

    assert fill is not None
    assert fill.timestamp == bar(1).timestamp
    assert fill.reference_price == Decimal("105.00000000")
    assert fill.price == Decimal("105.00000000")


@pytest.mark.parametrize(
    ("side", "expected_price"),
    [
        (OrderSide.BUY, Decimal("100.15000000")),
        (OrderSide.SELL, Decimal("99.85000000")),
    ],
)
def test_market_spread_and_slippage_are_adverse(
    side: OrderSide, expected_price: Decimal
) -> None:
    model = BarExecutionModel(
        BacktestConfig(
            starting_cash=Decimal("10000"),
            spread_bps=Decimal("5"),
            slippage_bps=Decimal("10"),
        )
    )

    fill = model.try_fill(_order(side=side), bar(1, opening="100"), "fill-1")

    assert fill is not None
    assert fill.price == expected_price
    assert fill.spread_cost == Decimal("0.50000000")
    assert fill.slippage_cost == Decimal("1.00000000")


@pytest.mark.parametrize(
    ("side", "opening", "low", "high", "close", "reached"),
    [
        (OrderSide.BUY, "100", "98", "105", "100", True),
        (OrderSide.BUY, "102", "101", "105", "103", False),
        (OrderSide.SELL, "100", "95", "102", "100", True),
        (OrderSide.SELL, "98", "95", "99", "97", False),
    ],
)
def test_limit_order_touch_policy(
    side: OrderSide,
    opening: str,
    low: str,
    high: str,
    close: str,
    reached: bool,
) -> None:
    model = BarExecutionModel(BacktestConfig(starting_cash=Decimal("10000")))
    order = _order(
        side=side,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )
    current = bar(1, opening=opening, low=low, high=high, close=close)

    fill = model.try_fill(order, current, "fill-1")

    assert (fill is not None) is reached
    if fill is not None:
        assert fill.price == Decimal("100.00000000")


def test_limit_all_in_price_remains_bounded_when_costs_are_enabled() -> None:
    model = BarExecutionModel(
        BacktestConfig(
            starting_cash=Decimal("10000"),
            spread_bps=Decimal("5"),
            slippage_bps=Decimal("5"),
        )
    )
    order = _order(
        order_type=OrderType.LIMIT, limit_price=Decimal("100")
    )

    assert (
        model.try_fill(
            order,
            bar(1, opening="100", low="99.95", high="102", close="101"),
            "fill-1",
        )
        is None
    )
    fill = model.try_fill(
        order,
        bar(1, opening="100", low="99.8", high="102", close="101"),
        "fill-1",
    )
    assert fill is not None
    assert fill.price == Decimal("100.00000000")


def test_commission_model_supports_fixed_percentage_and_minimum() -> None:
    percentage = ConfigurableCommissionModel(
        CommissionConfig(
            fixed=Decimal("1"),
            percentage_bps=Decimal("10"),
            minimum=Decimal("2"),
        )
    )

    assert percentage.calculate(Decimal("100")) == Decimal("2.00000000")
    assert percentage.calculate(Decimal("10000")) == Decimal("11.00000000")


def test_execution_model_rejects_non_pending_order() -> None:
    model = BarExecutionModel(BacktestConfig())
    order = BacktestOrder(
        order_id="order-1",
        symbol="AAPL",
        timeframe="1d",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        order_type=OrderType.MARKET,
        created_at=START,
        status=OrderStatus.REJECTED,
        completed_at=START,
    )

    with pytest.raises(Exception, match="only pending"):
        model.try_fill(order, bar(1), "fill-1")
