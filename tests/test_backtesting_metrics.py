import math
import statistics
from datetime import timedelta
from decimal import Decimal

import pytest

from backtest_support import START
from trading_ai.backtesting.metrics import (
    MetricsEngine,
    annualization_factor,
    drawdown_statistics,
)
from trading_ai.backtesting.models import EquityPoint, Fill, Trade
from trading_ai.core.models import OrderSide


def _point(index: int, equity: str, positions: str = "0") -> EquityPoint:
    equity_value = Decimal(equity)
    positions_value = Decimal(positions)
    return EquityPoint(
        timestamp=START + timedelta(days=index),
        cash=equity_value - positions_value,
        positions_value=positions_value,
        equity=equity_value,
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
    )


def _trade(index: int, net: str) -> Trade:
    net_value = Decimal(net)
    return Trade(
        trade_id=f"trade-{index}",
        symbol="AAPL",
        entry_time=START,
        exit_time=START + timedelta(days=index),
        entry_price=Decimal("100"),
        exit_price=Decimal("100") + net_value,
        quantity=Decimal("1"),
        gross_pnl=net_value,
        fees=Decimal("0"),
        spread_cost=Decimal("0"),
        slippage_cost=Decimal("0"),
        net_pnl=net_value,
        return_pct=net_value / Decimal("100"),
        holding_period_seconds=float(index * 86400),
    )


def _fill(index: int, price: str, commission: str = "0") -> Fill:
    return Fill(
        fill_id=f"fill-{index}",
        order_id=f"order-{index}",
        symbol="AAPL",
        side=OrderSide.BUY if index == 1 else OrderSide.SELL,
        quantity=Decimal("1"),
        reference_price=Decimal(price),
        price=Decimal(price),
        timestamp=START + timedelta(days=index),
        commission=Decimal(commission),
        spread_cost=Decimal("0.10"),
        slippage_cost=Decimal("0.20"),
    )


def test_annualization_factors_are_centralized_by_timeframe() -> None:
    assert annualization_factor("1d") == 252
    assert annualization_factor("4h") == 504
    assert annualization_factor("1h") == 1638
    with pytest.raises(ValueError, match="unsupported"):
        annualization_factor("5m")


def test_drawdown_peak_trough_recovery_and_duration() -> None:
    curve = (
        _point(0, "100"),
        _point(1, "110"),
        _point(2, "88"),
        _point(3, "121"),
    )

    drawdown, start, bottom, recovery, duration = drawdown_statistics(curve)

    assert drawdown == pytest.approx(-0.2)
    assert start == START + timedelta(days=1)
    assert bottom == START + timedelta(days=2)
    assert recovery == START + timedelta(days=3)
    assert duration == 2 * 86400


def test_metrics_have_known_numerical_results() -> None:
    curve = (
        _point(0, "100"),
        _point(1, "110", "50"),
        _point(2, "88", "44"),
        _point(3, "121"),
    )
    fills = (_fill(1, "100", "1"), _fill(2, "120", "1"))
    trades = (_trade(1, "10"), _trade(2, "-5"))

    metrics = MetricsEngine().calculate(
        initial_capital=Decimal("100"),
        curve=curve,
        fills=fills,
        trades=trades,
        timeframe="1d",
        risk_free_rate=Decimal("0"),
        dividend_income=Decimal("3"),
    )

    returns = [0.1, -0.2, 0.375]
    expected_volatility = statistics.stdev(returns) * math.sqrt(252)
    expected_sharpe = (
        statistics.mean(returns)
        / statistics.stdev(returns)
        * math.sqrt(252)
    )
    expected_sortino = statistics.mean(returns) / 0.2 * math.sqrt(252)
    assert metrics.initial_capital == Decimal("100")
    assert metrics.final_equity == Decimal("121")
    assert metrics.total_return == pytest.approx(0.21)
    assert metrics.annualized_return == pytest.approx(1.21 ** (252 / 3) - 1)
    assert metrics.volatility == pytest.approx(expected_volatility)
    assert metrics.sharpe_ratio == pytest.approx(expected_sharpe)
    assert metrics.sortino_ratio == pytest.approx(expected_sortino)
    assert metrics.max_drawdown_pct == pytest.approx(-0.2)
    assert metrics.calmar_ratio == pytest.approx(
        metrics.annualized_return / 0.2  # type: ignore[operator]
    )
    assert metrics.profit_factor == pytest.approx(2.0)
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.average_win == Decimal("10")
    assert metrics.average_loss == Decimal("-5")
    assert metrics.expectancy == Decimal("2.5")
    assert metrics.number_of_trades == 2
    assert metrics.turnover == pytest.approx(220 / ((100 + 110 + 88 + 121) / 4))
    assert metrics.exposure == pytest.approx((0 + 50 / 110 + 44 / 88 + 0) / 4)
    assert metrics.best_trade == Decimal("10")
    assert metrics.worst_trade == Decimal("-5")
    assert metrics.average_holding_period_seconds == pytest.approx(1.5 * 86400)
    assert metrics.total_commission == Decimal("2")
    assert metrics.total_spread_cost == Decimal("0.20")
    assert metrics.total_slippage_cost == Decimal("0.40")
    assert metrics.dividend_income == Decimal("3")


def test_metrics_handle_no_trades_no_downside_and_zero_variance_explicitly() -> None:
    curve = (_point(0, "100"), _point(1, "110"), _point(2, "121"))

    metrics = MetricsEngine().calculate(
        initial_capital=Decimal("100"),
        curve=curve,
        fills=(),
        trades=(),
        timeframe="1d",
        risk_free_rate=Decimal("0"),
        dividend_income=Decimal("0"),
    )

    assert metrics.profit_factor is None
    assert metrics.win_rate is None
    assert metrics.expectancy is None
    assert metrics.sortino_ratio is None
    assert metrics.sharpe_ratio is None
    assert metrics.drawdown_start is None
    assert metrics.drawdown_bottom is None


def test_profit_factor_without_a_losing_trade_is_explicitly_none() -> None:
    metrics = MetricsEngine().calculate(
        initial_capital=Decimal("100"),
        curve=(_point(0, "100"), _point(1, "110")),
        fills=(),
        trades=(_trade(1, "10"),),
        timeframe="1d",
        risk_free_rate=Decimal("0"),
        dividend_income=Decimal("0"),
    )

    assert metrics.profit_factor is None
