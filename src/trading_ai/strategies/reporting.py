"""Comparable strategy summaries that deliberately do not select a winner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trading_ai.core.models import BacktestResult


@dataclass(frozen=True, slots=True)
class StrategyReport:
    strategy: str
    version: str
    parameters: tuple[tuple[str, str], ...]
    number_of_trades: int
    total_return: float
    max_drawdown_pct: float
    sharpe_ratio: float | None
    turnover: float
    benchmark_return: float | None
    excess_return: float | None


def strategy_report(result: BacktestResult) -> StrategyReport:
    benchmark_return = (
        result.benchmark.total_return if result.benchmark is not None else None
    )
    return StrategyReport(
        strategy=result.strategy_name,
        version=result.strategy_version,
        parameters=result.strategy_parameters,
        number_of_trades=result.metrics.number_of_trades,
        total_return=result.metrics.total_return,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        sharpe_ratio=result.metrics.sharpe_ratio,
        turnover=result.metrics.turnover,
        benchmark_return=benchmark_return,
        excess_return=(
            result.metrics.total_return - benchmark_return
            if benchmark_return is not None
            else None
        ),
    )


def compare_reports(results: Sequence[BacktestResult]) -> tuple[StrategyReport, ...]:
    """Preserve caller order and expose metrics without declaring a best strategy."""

    return tuple(strategy_report(result) for result in results)
