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
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    costs_root = summary.get("costs") if isinstance(summary.get("costs"), dict) else {}
    domain = costs_root.get("summary") if isinstance(costs_root.get("summary"), dict) else {}
    explicit_trading = (
        summary.get("trading_costs")
        if isinstance(summary.get("trading_costs"), dict)
        else {}
    )
    statuses = dict(domain.get("component_statuses") or [])
    amount_fields = {
        "commission": "total_commission",
        "spread": "total_spread",
        "slippage": "total_slippage",
        "exchange_fees": "total_exchange_fees",
        "transaction_tax": "total_transaction_tax",
        "fx_cost": "total_fx_cost",
        "financing_cost": "total_financing_cost",
        "other_variable_cost": "total_other_variable_cost",
    }

    legacy = {
        "commission": _metric_cost(metrics, "total_commission"),
        "spread": _metric_cost(metrics, "total_spread_cost"),
        "slippage": _metric_cost(metrics, "total_slippage_cost"),
    }
    unavailable = CostComponent.unavailable(
        "transaction-cost component unavailable in this export"
    )

    def domain_component(name: str) -> CostComponent:
        fallback = legacy.get(name, unavailable)
        if not domain:
            return _explicit_cost(explicit_trading, name, fallback)
        try:
            status = CostKnowledge(str(statuses.get(name, "UNAVAILABLE")))
        except ValueError:
            return unavailable
        amount = _decimal(domain.get(amount_fields[name]))
        source = f"CostSummary.{amount_fields[name]}"
        if status is CostKnowledge.UNAVAILABLE:
            return CostComponent.unavailable(source)
        if status is CostKnowledge.NOT_APPLICABLE:
            return CostComponent.not_applicable(source)
        if amount is None:
            return CostComponent.unavailable(source)
        return CostComponent(status, amount, source)

    commission = domain_component("commission")
    spread = domain_component("spread")
    slippage = domain_component("slippage")
    exchange_fees = domain_component("exchange_fees")
    transaction_tax = domain_component("transaction_tax")
    fx_cost = domain_component("fx_cost")
    financing_cost = domain_component("financing_cost")
    other_variable_cost = domain_component("other_variable_cost")
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
    explicit_operating = costs_root.get("operating")
    explicit_operating = (
        explicit_operating
        if isinstance(explicit_operating, dict)
        else summary.get("operating_costs")
        if isinstance(summary.get("operating_costs"), dict)
        else {}
    )
    operating = OperatingCostBreakdown(
        market_data_subscription=_explicit_cost(
            explicit_operating, "market_data_subscription", operating_unavailable
        ),
        server_vps=_explicit_cost(explicit_operating, "server_vps", operating_unavailable),
        software_subscriptions=_explicit_cost(
            explicit_operating,
            (
                "software_subscription"
                if "software_subscription" in explicit_operating
                else "software_subscriptions"
            ),
            operating_unavailable,
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
    gross_pnl = _decimal(domain.get("gross_trading_pnl")) if domain else None
    if gross_pnl is None and not domain:
        modeled_components_known = all(
            item.status is CostKnowledge.KNOWN
            for item in (commission, spread, slippage)
        )
        gross_pnl = modeled_net + known_trading if modeled_net is not None and modeled_components_known else None
    has_unavailable = any(
        item.status is CostKnowledge.UNAVAILABLE for _, item in all_components
    )
    coverage = (
        CostCoverageStatus.UNAVAILABLE
        if not metrics and not domain
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
        net_pnl_known=(
            _decimal(domain.get("net_trading_pnl_before_operating"))
            if domain else modeled_net
        ),
        net_pnl_estimated=(
            _decimal(domain.get("net_economic_pnl"))
            if domain and not has_unavailable
            else None if has_unavailable
            else (gross_pnl - known - estimated if gross_pnl is not None else None)
        ),
        coverage_status=coverage,
        warnings=(
            "Cost coverage incomplete: exchange fees, transaction tax, FX, financing, other variable, and operating costs remain UNAVAILABLE until explicitly supplied.",
        ) if has_unavailable else (),
        net_trading_pnl_before_operating=_decimal(
            domain.get("net_trading_pnl_before_operating")
        ),
        operating_costs_total=_decimal(domain.get("operating_costs")),
        net_economic_pnl=_decimal(domain.get("net_economic_pnl")),
        tariff_profile_id=(
            str(domain.get("tariff_profile_id"))
            if domain.get("tariff_profile_id") else costs_root.get("tariff_profile_id")
        ),
        tariff_status=(
            str(domain.get("tariff_status"))
            if domain.get("tariff_status") else costs_root.get("tariff_status")
        ),
    )
