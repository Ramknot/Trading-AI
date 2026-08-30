"""Point-in-time currency conversion and distinct FX transaction cost."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_ai.costs.models import BPS, CostComponent
from trading_ai.portfolio.currency import CurrencyConverter


def convert_component(
    component: CostComponent,
    *,
    target_currency: str,
    timestamp: datetime,
    converter: CurrencyConverter,
) -> CostComponent:
    if component.status.value == "UNAVAILABLE" or component.currency == target_currency:
        return component
    if not converter.has_rate(component.currency, target_currency, timestamp):
        return CostComponent.unavailable(
            component.name,
            target_currency,
            component.source,
            f"no point-in-time FX rate for {component.currency}/{target_currency}",
        )
    assert component.amount is not None
    converted = converter.convert(
        component.amount, component.currency, target_currency, timestamp
    )
    return CostComponent(
        component.name,
        component.status,
        converted,
        target_currency,
        component.source,
        component.reason,
    )


def fx_cost_component(
    *,
    notional: Decimal,
    from_currency: str,
    base_currency: str,
    timestamp: datetime,
    converter: CurrencyConverter,
    fx_cost_bps: Decimal,
) -> CostComponent:
    if from_currency == base_currency:
        return CostComponent.not_applicable(
            "fx_cost", base_currency, "cost configuration",
            "instrument and account use the same currency",
        )
    if not converter.has_rate(from_currency, base_currency, timestamp):
        return CostComponent.unavailable(
            "fx_cost", base_currency, "CurrencyConverter",
            f"no point-in-time FX rate for {from_currency}/{base_currency}",
        )
    converted_notional = converter.convert(
        notional, from_currency, base_currency, timestamp
    )
    return CostComponent.estimated(
        "fx_cost",
        converted_notional * fx_cost_bps / BPS,
        base_currency,
        "configuration-driven FX cost assumption",
    )
