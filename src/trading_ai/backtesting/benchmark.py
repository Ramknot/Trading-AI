"""Frictionless raw-price Buy & Hold benchmark with explicit actions."""

from __future__ import annotations

from decimal import Decimal

from trading_ai.backtesting.models import BenchmarkResult, EquityPoint
from trading_ai.backtesting.metrics import drawdown_statistics
from trading_ai.core.models import MarketBar
from trading_ai.data.models import CorporateAction, Dividend, StockSplit


ZERO = Decimal("0")


class BuyAndHoldBenchmark:
    def run(
        self,
        *,
        symbol: str,
        bars: tuple[MarketBar, ...],
        corporate_actions: tuple[CorporateAction, ...],
        initial_capital: Decimal,
        strategy_total_return: float,
    ) -> BenchmarkResult:
        selected = tuple(sorted(
            (bar for bar in bars if bar.symbol == symbol),
            key=lambda bar: bar.timestamp,
        ))
        if not selected:
            raise ValueError(f"benchmark dataset is missing for {symbol}")
        entry_time = selected[0].timestamp
        quantity = initial_capital / selected[0].close
        cash = ZERO
        action_index = 0
        actions = tuple(sorted(
            (action for action in corporate_actions if action.symbol == symbol),
            key=lambda action: action.timestamp,
        ))
        while action_index < len(actions) and actions[action_index].timestamp <= entry_time:
            action_index += 1
        curve: list[EquityPoint] = []
        for bar in selected:
            while (
                action_index < len(actions)
                and actions[action_index].timestamp <= bar.timestamp
            ):
                action = actions[action_index]
                if isinstance(action, Dividend):
                    cash += quantity * action.value
                elif isinstance(action, StockSplit):
                    quantity *= action.value
                action_index += 1
            positions_value = quantity * bar.close
            equity = cash + positions_value
            curve.append(
                EquityPoint(
                    timestamp=bar.timestamp,
                    cash=cash,
                    positions_value=positions_value,
                    equity=equity,
                    realized_pnl=ZERO,
                    unrealized_pnl=equity - initial_capital,
                )
            )
        final_equity = curve[-1].equity
        total_return = float(final_equity / initial_capital - Decimal("1"))
        max_drawdown, _, _, _, _ = drawdown_statistics(tuple(curve))
        return BenchmarkResult(
            symbol=symbol,
            initial_equity=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown_pct=max_drawdown,
            excess_return=strategy_total_return - total_return,
            equity_curve=tuple(curve),
        )
