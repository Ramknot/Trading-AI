"""Period-level operating costs, never allocated arbitrarily to fills."""

from datetime import datetime
from decimal import Decimal

from trading_ai.costs.config import OperatingComponentConfig, OperatingCostConfig
from trading_ai.costs.models import (
    CostComponent,
    CostStatus,
    OperatingCostBreakdown,
)


def _component(name: str, config: OperatingComponentConfig, currency: str) -> CostComponent:
    if config.status is CostStatus.UNAVAILABLE:
        return CostComponent.unavailable(
            name, currency, config.source_reference, "operating cost not supplied"
        )
    if config.status is CostStatus.NOT_APPLICABLE:
        return CostComponent.not_applicable(
            name, currency, config.source_reference, "explicitly not applicable"
        )
    assert config.amount is not None
    return CostComponent(
        name, config.status, config.amount, currency, config.source_reference
    )


def build_operating_costs(
    config: OperatingCostConfig, period_start: datetime, period_end: datetime
) -> OperatingCostBreakdown:
    components = (
        _component("market_data_subscription", config.market_data_subscription, config.currency),
        _component("server_vps", config.server_vps, config.currency),
        _component("software_subscription", config.software_subscription, config.currency),
        _component("other_fixed_cost", config.other_fixed_cost, config.currency),
    )
    if any(item.status is CostStatus.UNAVAILABLE for item in components):
        total = CostComponent.unavailable(
            "total_operating_cost", config.currency, "operating cost configuration",
            "one or more operating costs are unavailable",
        )
    else:
        amount = sum(
            (item.amount for item in components if item.amount is not None),
            Decimal("0"),
        )
        status = CostStatus.ESTIMATED if any(item.status is CostStatus.ESTIMATED for item in components) else CostStatus.KNOWN
        total = CostComponent(
            "total_operating_cost", status, amount, config.currency,
            "sum of period-level operating costs",
        )
    return OperatingCostBreakdown(
        period_start,
        period_end,
        components[0],
        components[1],
        components[2],
        components[3],
        total,
    )
