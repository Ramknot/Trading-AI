"""Cost-aware PnL summaries; complete net economics require complete coverage."""

from __future__ import annotations

from decimal import Decimal

from trading_ai.costs.models import (
    ActualTradingCost,
    CostCoverage,
    CostStatus,
    CostSummary,
    EconomicDecision,
    EconomicDecisionStatus,
    OperatingCostBreakdown,
    PreTradeCostEstimate,
    TariffStatus,
    ZERO,
)


def _component_total(actuals: tuple[ActualTradingCost, ...], name: str) -> Decimal | None:
    components = tuple(getattr(item.breakdown, name) for item in actuals)
    if any(item.status is CostStatus.UNAVAILABLE for item in components):
        return None
    if any(item.amount is None for item in components):
        raise AssertionError("available cost component has no numeric amount")
    return sum((item.amount for item in components if item.amount is not None), ZERO)


def build_cost_summary(
    *,
    engine_name: str,
    engine_version: str,
    config_hash: str,
    tariff_profile_id: str,
    tariff_status: TariffStatus,
    estimates: tuple[PreTradeCostEstimate, ...],
    actuals: tuple[ActualTradingCost, ...],
    decisions: tuple[EconomicDecision, ...],
    initial_cash: Decimal,
    final_equity: Decimal,
    operating: OperatingCostBreakdown,
) -> CostSummary:
    names = (
        "commission", "spread", "slippage", "exchange_fees",
        "transaction_tax", "fx_cost", "financing_cost", "other_variable_cost",
    )
    totals = {name: _component_total(actuals, name) for name in names}
    def aggregate_status(name: str) -> CostStatus:
        values = tuple(getattr(item.breakdown, name).status for item in actuals)
        if not values:
            return CostStatus.UNAVAILABLE
        if CostStatus.UNAVAILABLE in values:
            return CostStatus.UNAVAILABLE
        if CostStatus.ESTIMATED in values:
            return CostStatus.ESTIMATED
        if all(value is CostStatus.NOT_APPLICABLE for value in values):
            return CostStatus.NOT_APPLICABLE
        return CostStatus.KNOWN
    complete = all(value is not None for value in totals.values())
    variable = sum((value for value in totals.values() if value is not None), ZERO) if complete else None
    modeled_net = final_equity - initial_cash
    gross = modeled_net + variable if variable is not None else None
    operating_total = operating.total_operating_cost.amount
    net_economic = modeled_net - operating_total if operating_total is not None else None
    return CostSummary(
        engine_name=engine_name,
        engine_version=engine_version,
        config_hash=config_hash,
        tariff_profile_id=tariff_profile_id,
        tariff_status=tariff_status,
        estimate_count=len(estimates),
        actual_count=len(actuals),
        economic_pass=sum(item.status is EconomicDecisionStatus.PASS for item in decisions),
        economic_block=sum(item.status is EconomicDecisionStatus.BLOCK for item in decisions),
        economic_incomplete=sum(item.status is EconomicDecisionStatus.INCOMPLETE for item in decisions),
        total_commission=totals["commission"],
        total_spread=totals["spread"],
        total_slippage=totals["slippage"],
        total_exchange_fees=totals["exchange_fees"],
        total_transaction_tax=totals["transaction_tax"],
        total_fx_cost=totals["fx_cost"],
        total_financing_cost=totals["financing_cost"],
        total_other_variable_cost=totals["other_variable_cost"],
        total_variable_cost=variable,
        cost_coverage=CostCoverage.COMPLETE if complete else CostCoverage.INCOMPLETE,
        gross_trading_pnl=gross,
        net_trading_pnl_before_operating=modeled_net if complete else None,
        operating_costs=operating_total,
        net_economic_pnl=net_economic if complete else None,
        gross_return=float(gross / initial_cash) if gross is not None else None,
        net_return_before_operating=float(modeled_net / initial_cash) if complete else None,
        net_economic_return=float(net_economic / initial_cash) if net_economic is not None and complete else None,
        component_statuses=tuple(sorted(
            (name, aggregate_status(name)) for name in names
        )),
    )
