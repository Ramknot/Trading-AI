"""Transparent portfolio-construction metrics without fabricated PnL attribution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Sequence

from trading_ai.backtesting.models import EquityPoint, Fill
from trading_ai.portfolio.config import BalancedPortfolioConfig
from trading_ai.portfolio.models import (
    PortfolioDecision,
    PortfolioDecisionStatus,
    PortfolioMetrics,
    RebalancePlan,
    StrategySleeveState,
)

if TYPE_CHECKING:
    from trading_ai.core.models import BacktestResult


@dataclass(frozen=True, slots=True)
class PortfolioResearchRun:
    strategy_name: str
    total_return: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    number_of_trades: int
    turnover: float
    fees: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioResearchComparison:
    dataset_ids: tuple[str, ...]
    started_at: str
    completed_at: str
    single_strategy_runs: tuple[PortfolioResearchRun, ...]
    multi_strategy_run: PortfolioResearchRun
    assumptions_verified: bool = True
    automatic_winner_selection: bool = False


def _research_run(result: "BacktestResult") -> PortfolioResearchRun:
    return PortfolioResearchRun(
        strategy_name=result.strategy_name,
        total_return=result.metrics.total_return,
        sharpe_ratio=result.metrics.sharpe_ratio,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        number_of_trades=result.metrics.number_of_trades,
        turnover=result.metrics.turnover,
        fees=(
            result.metrics.total_commission
            + result.metrics.total_spread_cost
            + result.metrics.total_slippage_cost
        ),
    )


def compare_single_to_multi(
    single_results: Sequence["BacktestResult"],
    multi_result: "BacktestResult",
) -> PortfolioResearchComparison:
    """Compare mechanics under identical assumptions without selecting a winner."""

    singles = tuple(single_results)
    if not singles:
        raise ValueError("at least one single-strategy result is required")
    if multi_result.portfolio_engine_name == "unavailable":
        raise ValueError("multi_result must use an explicit PortfolioEngine")
    if multi_result.ml_mode != "DISABLED" or any(
        item.ml_mode != "DISABLED" for item in singles
    ):
        raise ValueError("Lot 7 portfolio comparison currently requires ML DISABLED")
    reference_fields = (
        "dataset_references",
        "config",
        "started_at",
        "completed_at",
        "risk_config_hash",
        "regime_config_hash",
        "strategy_policy_config_hash",
        "source_hash_sha256",
        "code_version",
    )
    for result in singles:
        if any(
            getattr(result, field_name) != getattr(multi_result, field_name)
            for field_name in reference_fields
        ):
            raise ValueError(
                "single and multi-strategy comparisons require identical datasets, "
                "period, costs, code, regime/policy, and Risk assumptions"
            )
        prefix = f"strategy.{result.strategy_name}."
        multi_parameters = tuple(
            sorted(
                (name.removeprefix(prefix), value)
                for name, value in multi_result.strategy_parameters
                if name.startswith(prefix)
            )
        )
        if multi_parameters != result.strategy_parameters:
            raise ValueError(
                f"multi-strategy config does not match {result.strategy_name} run"
            )
    return PortfolioResearchComparison(
        dataset_ids=tuple(
            item.dataset_id for item in multi_result.dataset_references
        ),
        started_at=multi_result.started_at.isoformat(),
        completed_at=multi_result.completed_at.isoformat(),
        single_strategy_runs=tuple(
            _research_run(item)
            for item in sorted(singles, key=lambda result: result.strategy_name)
        ),
        multi_strategy_run=_research_run(multi_result),
    )


def build_portfolio_metrics(
    plans: Sequence[RebalancePlan],
    decisions: Sequence[PortfolioDecision],
    sleeves: Sequence[StrategySleeveState],
    fills: Sequence[Fill],
    equity_curve: Sequence[EquityPoint],
    config: BalancedPortfolioConfig,
) -> PortfolioMetrics:
    exposures = [
        float(abs(point.positions_value) / point.equity)
        for point in equity_curve
        if point.equity > Decimal("0")
    ]
    cash_fractions = [
        float(point.cash / point.equity)
        for point in equity_curve
        if point.equity > Decimal("0")
    ]
    max_positions = max(
        (
            sum(target.target_weight > Decimal("0") for target in plan.targets)
            for plan in plans
        ),
        default=0,
    )
    equity_by_time = {point.timestamp: point.equity for point in equity_curve}
    executed_turnover = 0.0
    for fill in fills:
        equity = equity_by_time.get(fill.timestamp)
        if equity is not None and equity > Decimal("0"):
            executed_turnover += float(fill.quantity * fill.price / equity)
    latest: dict[tuple[str, str], StrategySleeveState] = {}
    for item in sleeves:
        key = (item.strategy_name, item.symbol)
        if key not in latest or item.last_updated_at >= latest[key].last_updated_at:
            latest[key] = item
    sleeve_totals = tuple(
        sorted(
            (
                sleeve.strategy_name,
                float(
                    sum(
                        (
                            item.target_weight_contribution
                            for item in latest.values()
                            if item.strategy_name == sleeve.strategy_name
                        ),
                        Decimal("0"),
                    )
                ),
            )
            for sleeve in config.strategy_sleeves
        )
    )
    unused_count = 0
    for plan in plans:
        by_strategy: dict[str, Decimal] = {}
        for target in plan.targets:
            for item in target.contributors:
                by_strategy[item.strategy_name] = (
                    by_strategy.get(item.strategy_name, Decimal("0")) + item.weight
                )
        unused_count += sum(
            by_strategy.get(sleeve.strategy_name, Decimal("0")) < sleeve.budget_weight
            for sleeve in config.strategy_sleeves
        )
    group_history: list[tuple[str, str, float]] = []
    for plan in plans:
        group_cycle: dict[str, Decimal] = {}
        for target in plan.targets:
            group = target.group or "UNKNOWN"
            group_cycle[group] = group_cycle.get(group, Decimal("0")) + target.target_weight
        group_history.extend(
            (plan.timestamp.isoformat(), group, float(weight))
            for group, weight in sorted(group_cycle.items())
        )
    codes = [code for decision in decisions for code in decision.reason_codes]
    return PortfolioMetrics(
        average_gross_exposure=(sum(exposures) / len(exposures) if exposures else 0.0),
        max_gross_exposure=max(exposures, default=0.0),
        average_cash_fraction=(
            sum(cash_fractions) / len(cash_fractions) if cash_fractions else 1.0
        ),
        max_unique_positions=max_positions,
        planned_turnover=sum(float(plan.planned_turnover) for plan in plans),
        executed_turnover=executed_turnover,
        opportunities_selected=sum(
            item.status is PortfolioDecisionStatus.SELECT for item in decisions
        ),
        opportunities_deferred=sum(
            item.status is PortfolioDecisionStatus.DEFER for item in decisions
        ),
        opportunities_rejected=sum(
            item.status is PortfolioDecisionStatus.REJECT for item in decisions
        ),
        targets_by_strategy_sleeve=sleeve_totals,
        time_with_unused_strategy_budget=unused_count,
        group_exposure_over_time=tuple(group_history),
        high_correlation_selections=sum("HIGH_CORRELATION_SOFT" in code for code in codes),
        unknown_correlation_cases=sum("CORRELATION_UNKNOWN" in code for code in codes),
    )
