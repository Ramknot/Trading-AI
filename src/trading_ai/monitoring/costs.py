"""Observability-only cost views built from costs already produced by engines."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from trading_ai.monitoring.models import (
    CostComponent,
    CostCoverageStatus,
    CostKnowledge,
    CostSnapshot,
    OperatingCostBreakdown,
    TradingCostBreakdown,
)


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _metric_cost(metrics: dict[str, Any], name: str) -> CostComponent:
    amount = _decimal(metrics.get(name))
    if amount is None or amount < Decimal("0"):
        return CostComponent.unavailable("backtest metric not available")
    return CostComponent.known(amount, f"BacktestMetrics.{name}")


def _explicit_cost(
    payload: dict[str, Any], name: str, fallback: CostComponent
) -> CostComponent:
    raw = payload.get(name)
    if not isinstance(raw, dict):
        return fallback
    try:
        status = CostKnowledge(str(raw.get("status")))
    except ValueError:
        return fallback
    amount = _decimal(raw.get("amount"))
    source = str(raw.get("source") or "explicit trusted monitoring input")
    if status is CostKnowledge.UNAVAILABLE:
        return CostComponent.unavailable(source)
    if amount is None:
        return fallback
    return CostComponent(status, amount, source)


def build_cost_snapshot(summary: dict[str, Any], timestamp: datetime) -> CostSnapshot:
    """Expose modeled costs without assigning zero to unimplemented categories."""

    metrics = summary.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    commission = _metric_cost(metrics, "total_commission")
    spread = _metric_cost(metrics, "total_spread_cost")
    slippage = _metric_cost(metrics, "total_slippage_cost")
    explicit_trading = summary.get("trading_costs")
    explicit_trading = explicit_trading if isinstance(explicit_trading, dict) else {}
    commission = _explicit_cost(explicit_trading, "commission", commission)
    spread = _explicit_cost(explicit_trading, "spread", spread)
    slippage = _explicit_cost(explicit_trading, "slippage", slippage)
    unavailable = CostComponent.unavailable("Lot 8.1 transaction-cost engine required")
    exchange_fees = _explicit_cost(explicit_trading, "exchange_fees", unavailable)
    transaction_tax = _explicit_cost(explicit_trading, "transaction_tax", unavailable)
    fx_cost = _explicit_cost(explicit_trading, "fx_cost", unavailable)
    financing_cost = _explicit_cost(explicit_trading, "financing_cost", unavailable)
    other_variable_cost = _explicit_cost(
        explicit_trading, "other_variable_cost", unavailable
    )
    components = (
        commission,
        spread,
        slippage,
        exchange_fees,
        transaction_tax,
        fx_cost,
        financing_cost,
        other_variable_cost,
    )
    if all(item.status is not CostKnowledge.UNAVAILABLE for item in components):
        total_amount = sum((item.amount or Decimal("0") for item in components), Decimal("0"))
        total_status = (
            CostKnowledge.ESTIMATED
            if any(item.status is CostKnowledge.ESTIMATED for item in components)
            else CostKnowledge.KNOWN
        )
        total = CostComponent(total_status, total_amount, "sum of complete cost components")
    else:
        total = CostComponent.unavailable("one or more variable-cost components unavailable")
    trading = TradingCostBreakdown(
        commission=commission,
        spread=spread,
        slippage=slippage,
        exchange_fees=exchange_fees,
        transaction_tax=transaction_tax,
        fx_cost=fx_cost,
        financing_cost=financing_cost,
        other_variable_cost=other_variable_cost,
        total_variable_cost=total,
    )
    operating_unavailable = CostComponent.unavailable(
        "operating costs were not supplied for this run"
    )
    explicit_operating = summary.get("operating_costs")
    explicit_operating = (
        explicit_operating if isinstance(explicit_operating, dict) else {}
    )
    operating = OperatingCostBreakdown(
        market_data_subscription=_explicit_cost(
            explicit_operating, "market_data_subscription", operating_unavailable
        ),
        server_vps=_explicit_cost(explicit_operating, "server_vps", operating_unavailable),
        software_subscriptions=_explicit_cost(
            explicit_operating, "software_subscriptions", operating_unavailable
        ),
        other_fixed_cost=_explicit_cost(
            explicit_operating, "other_fixed_cost", operating_unavailable
        ),
    )
    all_components = trading.components + operating.components
    known = sum(
        (item.amount or Decimal("0") for _, item in all_components if item.status is CostKnowledge.KNOWN),
        Decimal("0"),
    )
    estimated = sum(
        (
            item.amount or Decimal("0")
            for _, item in all_components
            if item.status is CostKnowledge.ESTIMATED
        ),
        Decimal("0"),
    )
    known_trading = sum(
        (
            item.amount or Decimal("0")
            for _, item in trading.components
            if item.status is CostKnowledge.KNOWN
        ),
        Decimal("0"),
    )
    estimated_trading = sum(
        (
            item.amount or Decimal("0")
            for _, item in trading.components
            if item.status is CostKnowledge.ESTIMATED
        ),
        Decimal("0"),
    )
    initial = _decimal(summary.get("initial_cash"))
    final = _decimal(summary.get("final_equity"))
    modeled_net = final - initial if initial is not None and final is not None else None
    modeled_components_known = all(
        item.status is CostKnowledge.KNOWN for item in (commission, spread, slippage)
    )
    gross_pnl = (
        modeled_net + known_trading
        if modeled_net is not None and modeled_components_known
        else None
    )
    has_unavailable = any(
        item.status is CostKnowledge.UNAVAILABLE for _, item in all_components
    )
    coverage = (
        CostCoverageStatus.UNAVAILABLE
        if not metrics
        else CostCoverageStatus.INCOMPLETE
        if has_unavailable
        else CostCoverageStatus.COMPLETE
    )
    return CostSnapshot(
        run_id=str(summary.get("run_id", "unknown")),
        timestamp=timestamp,
        trading=trading,
        operating=operating,
        gross_pnl=gross_pnl,
        known_trading_costs=known_trading,
        estimated_trading_costs=estimated_trading,
        known_operating_costs=known - known_trading,
        estimated_operating_costs=estimated - estimated_trading,
        net_pnl_known=modeled_net,
        net_pnl_estimated=None if has_unavailable else (gross_pnl - known - estimated if gross_pnl is not None else None),
        coverage_status=coverage,
        warnings=(
            "Cost coverage incomplete: exchange fees, transaction tax, FX, financing, other variable, and operating costs remain UNAVAILABLE until explicitly supplied.",
        ) if has_unavailable else (),
    )
