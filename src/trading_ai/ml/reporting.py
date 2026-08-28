"""Neutral Quant versus Quant+ML comparison reporting."""

from __future__ import annotations

from dataclasses import dataclass

from trading_ai.core.models import BacktestResult
from trading_ai.ml.models import TimeRange


@dataclass(frozen=True, slots=True)
class BacktestComparisonMetrics:
    total_return: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    trades: int
    turnover: float
    fees: str


@dataclass(frozen=True, slots=True)
class ResearchComparisonReport:
    model_id: str
    strategy_name: str
    sample_scope: str
    quant: BacktestComparisonMetrics
    quant_plus_ml: BacktestComparisonMetrics
    return_delta: float
    sharpe_delta: float | None
    drawdown_delta: float
    trade_delta: int
    turnover_delta: float


def _metrics(result: BacktestResult) -> BacktestComparisonMetrics:
    return BacktestComparisonMetrics(
        total_return=result.metrics.total_return,
        sharpe_ratio=result.metrics.sharpe_ratio,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        trades=result.metrics.number_of_trades,
        turnover=result.metrics.turnover,
        fees=str(
            result.metrics.total_commission
            + result.metrics.total_spread_cost
            + result.metrics.total_slippage_cost
        ),
    )


def compare_quant_to_ml(
    quant: BacktestResult,
    quant_plus_ml: BacktestResult,
    *,
    model_id: str,
    training_period: TimeRange,
    validation_period: TimeRange,
    test_period: TimeRange,
) -> ResearchComparisonReport:
    """Compare identical assumptions without selecting or praising a winner."""

    invariants = (
        quant.strategy_name == quant_plus_ml.strategy_name,
        quant.strategy_version == quant_plus_ml.strategy_version,
        quant.strategy_parameters == quant_plus_ml.strategy_parameters,
        quant.dataset_references == quant_plus_ml.dataset_references,
        quant.config == quant_plus_ml.config,
        quant.regime_config_hash == quant_plus_ml.regime_config_hash,
        quant.strategy_policy_config_hash == quant_plus_ml.strategy_policy_config_hash,
        quant.risk_config_hash == quant_plus_ml.risk_config_hash,
        (quant.benchmark.symbol if quant.benchmark else None)
        == (quant_plus_ml.benchmark.symbol if quant_plus_ml.benchmark else None),
        quant.started_at == quant_plus_ml.started_at,
        quant.completed_at == quant_plus_ml.completed_at,
        quant_plus_ml.ml_model_id == model_id,
    )
    if not all(invariants):
        raise ValueError("Quant and Quant+ML comparisons require identical assumptions")
    if quant.started_at >= test_period.start and quant.completed_at < test_period.end:
        scope = "OUT_OF_SAMPLE"
    elif (
        quant.started_at >= validation_period.start
        and quant.completed_at < validation_period.end
    ):
        scope = "VALIDATION"
    elif quant.started_at >= training_period.start and quant.completed_at < training_period.end:
        scope = "IN_SAMPLE"
    else:
        scope = "OUTSIDE_DECLARED_SPLIT"
    quant_metrics = _metrics(quant)
    ml_metrics = _metrics(quant_plus_ml)
    sharpe_delta = (
        None
        if quant_metrics.sharpe_ratio is None or ml_metrics.sharpe_ratio is None
        else ml_metrics.sharpe_ratio - quant_metrics.sharpe_ratio
    )
    return ResearchComparisonReport(
        model_id=model_id,
        strategy_name=quant.strategy_name,
        sample_scope=scope,
        quant=quant_metrics,
        quant_plus_ml=ml_metrics,
        return_delta=ml_metrics.total_return - quant_metrics.total_return,
        sharpe_delta=sharpe_delta,
        drawdown_delta=ml_metrics.max_drawdown_pct - quant_metrics.max_drawdown_pct,
        trade_delta=ml_metrics.trades - quant_metrics.trades,
        turnover_delta=ml_metrics.turnover - quant_metrics.turnover,
    )
