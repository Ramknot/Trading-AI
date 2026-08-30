"""Independent deterministic performance metrics for equity and closed trades."""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from decimal import Decimal

from trading_ai.backtesting.models import BacktestMetrics, EquityPoint, Fill, Trade


ZERO = Decimal("0")


def annualization_factor(timeframe: str) -> float:
    """Equity-market convention centralized for daily and intraday bars."""

    factors = {
        "1d": 252.0,
        "4h": 504.0,
        "1h": 1638.0,
    }
    try:
        return factors[timeframe]
    except KeyError as exc:
        raise ValueError(f"unsupported annualization timeframe {timeframe!r}") from exc


def drawdown_statistics(
    curve: tuple[EquityPoint, ...],
) -> tuple[float, datetime | None, datetime | None, datetime | None, float]:
    if not curve:
        return 0.0, None, None, None, 0.0
    peak_value = curve[0].equity
    peak_time = curve[0].timestamp
    worst = 0.0
    worst_start = peak_time
    worst_bottom = peak_time
    worst_peak_value = peak_value
    for point in curve:
        if point.equity > peak_value:
            peak_value = point.equity
            peak_time = point.timestamp
        if peak_value <= ZERO:
            continue
        drawdown = float(point.equity / peak_value - Decimal("1"))
        if drawdown < worst:
            worst = drawdown
            worst_start = peak_time
            worst_bottom = point.timestamp
            worst_peak_value = peak_value
    recovery: datetime | None = None
    if worst == 0:
        return 0.0, None, None, None, 0.0
    if worst < 0:
        recovery = next(
            (
                point.timestamp
                for point in curve
                if point.timestamp > worst_bottom
                and point.equity >= worst_peak_value
            ),
            None,
        )
    end = recovery or curve[-1].timestamp
    duration = max(0.0, (end - worst_start).total_seconds()) if worst < 0 else 0.0
    return worst, worst_start, worst_bottom, recovery, duration


class MetricsEngine:
    def calculate(
        self,
        *,
        initial_capital: Decimal,
        curve: tuple[EquityPoint, ...],
        fills: tuple[Fill, ...],
        trades: tuple[Trade, ...],
        timeframe: str,
        risk_free_rate: Decimal,
        dividend_income: Decimal,
    ) -> BacktestMetrics:
        if not curve:
            raise ValueError("metrics require a non-empty equity curve")
        final_equity = curve[-1].equity
        total_return = float(final_equity / initial_capital - Decimal("1"))
        returns = [
            float(current.equity / previous.equity - Decimal("1"))
            for previous, current in zip(curve, curve[1:])
            if previous.equity > ZERO
        ]
        factor = annualization_factor(timeframe)
        annualized_return: float | None = None
        if returns and final_equity > ZERO:
            annualized_return = float(
                (final_equity / initial_capital)
                ** Decimal(str(factor / len(returns)))
                - Decimal("1")
            )
        volatility: float | None = None
        sharpe: float | None = None
        sortino: float | None = None
        if len(returns) >= 2:
            standard_deviation = statistics.stdev(returns)
            volatility = standard_deviation * math.sqrt(factor)
            risk_free_per_period = (
                (1.0 + float(risk_free_rate)) ** (1.0 / factor) - 1.0
            )
            excess = [value - risk_free_per_period for value in returns]
            excess_deviation = statistics.stdev(excess)
            if excess_deviation > 0:
                sharpe = statistics.mean(excess) / excess_deviation * math.sqrt(
                    factor
                )
            downside = [value for value in excess if value < 0]
            if downside:
                downside_deviation = math.sqrt(
                    sum(value * value for value in downside) / len(downside)
                )
                if downside_deviation > 0:
                    sortino = statistics.mean(excess) / downside_deviation * math.sqrt(
                        factor
                    )
        (
            max_drawdown,
            drawdown_start,
            drawdown_bottom,
            drawdown_recovery,
            drawdown_duration,
        ) = drawdown_statistics(curve)
        calmar = None
        if annualized_return is not None and max_drawdown < 0:
            calmar = annualized_return / abs(max_drawdown)
        winning = tuple(trade for trade in trades if trade.net_pnl > ZERO)
        losing = tuple(trade for trade in trades if trade.net_pnl < ZERO)
        gross_profit = sum((trade.net_pnl for trade in winning), ZERO)
        gross_loss = sum((trade.net_pnl for trade in losing), ZERO)
        profit_factor = (
            float(gross_profit / abs(gross_loss)) if gross_loss < ZERO else None
        )
        trade_count = len(trades)
        average_win = (
            sum((trade.net_pnl for trade in winning), ZERO) / len(winning)
            if winning
            else None
        )
        average_loss = (
            sum((trade.net_pnl for trade in losing), ZERO) / len(losing)
            if losing
            else None
        )
        expectancy = (
            sum((trade.net_pnl for trade in trades), ZERO) / trade_count
            if trades
            else None
        )
        average_equity = sum(
            (point.equity for point in curve), ZERO
        ) / len(curve)
        traded_notional = sum(
            (fill.price * fill.quantity for fill in fills), ZERO
        )
        turnover = (
            float(traded_notional / average_equity) if average_equity > ZERO else 0.0
        )
        exposure_values = [
            float(point.positions_value / point.equity)
            for point in curve
            if point.equity > ZERO
        ]
        exposure = statistics.mean(exposure_values) if exposure_values else 0.0
        net_values = [trade.net_pnl for trade in trades]
        total_exchange_fees = sum((fill.exchange_fees for fill in fills), ZERO)
        total_transaction_tax = sum((fill.transaction_tax for fill in fills), ZERO)
        total_fx_cost = sum((fill.fx_cost for fill in fills), ZERO)
        total_financing_cost = sum((fill.financing_cost for fill in fills), ZERO)
        total_other_variable_cost = sum(
            (fill.other_variable_cost for fill in fills), ZERO
        )
        total_commission = sum((fill.commission for fill in fills), ZERO)
        total_spread = sum((fill.spread_cost for fill in fills), ZERO)
        total_slippage = sum((fill.slippage_cost for fill in fills), ZERO)
        return BacktestMetrics(
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown,
            calmar_ratio=calmar,
            drawdown_start=drawdown_start,
            drawdown_bottom=drawdown_bottom,
            drawdown_recovery=drawdown_recovery,
            drawdown_duration_seconds=drawdown_duration,
            profit_factor=profit_factor,
            win_rate=(len(winning) / trade_count if trade_count else None),
            average_win=average_win,
            average_loss=average_loss,
            expectancy=expectancy,
            number_of_trades=trade_count,
            turnover=turnover,
            exposure=exposure,
            best_trade=max(net_values) if net_values else None,
            worst_trade=min(net_values) if net_values else None,
            average_holding_period_seconds=(
                statistics.mean(
                    trade.holding_period_seconds for trade in trades
                )
                if trades
                else None
            ),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_commission=total_commission,
            total_spread_cost=total_spread,
            total_slippage_cost=total_slippage,
            dividend_income=dividend_income,
            total_exchange_fees=total_exchange_fees,
            total_transaction_tax=total_transaction_tax,
            total_fx_cost=total_fx_cost,
            total_financing_cost=total_financing_cost,
            total_other_variable_cost=total_other_variable_cost,
            total_variable_cost=(
                total_commission + total_spread + total_slippage
                + total_exchange_fees + total_transaction_tax + total_fx_cost
                + total_financing_cost + total_other_variable_cost
            ),
        )
