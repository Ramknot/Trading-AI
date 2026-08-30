"""Financing-cost policy for Balanced cash-only simulation."""

from trading_ai.costs.models import CostComponent


def balanced_financing_component(currency: str) -> CostComponent:
    return CostComponent.not_applicable(
        "financing_cost",
        currency,
        "Balanced cash-only profile",
        "margin and leverage are disabled",
    )
