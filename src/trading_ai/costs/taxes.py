"""Dated transaction-tax calculation driven by explicit instrument metadata."""

from decimal import Decimal

from trading_ai.core.models import OrderSide
from trading_ai.costs.models import (
    BPS,
    CostComponent,
    CostStatus,
    InstrumentCostMetadata,
    TariffStatus,
    TransactionTaxRule,
)


def transaction_tax_component(
    *,
    metadata: InstrumentCostMetadata,
    rule: TransactionTaxRule | None,
    side: OrderSide,
    notional: Decimal,
    timestamp,
    allow_retrospective: bool,
) -> CostComponent:
    currency = metadata.currency
    if metadata.transaction_tax_applicable is None:
        return CostComponent.unavailable(
            "transaction_tax", currency, metadata.source_reference,
            "instrument tax applicability is unknown",
        )
    if metadata.transaction_tax_applicable is False:
        return CostComponent.not_applicable(
            "transaction_tax", currency, metadata.source_reference,
            "instrument is explicitly outside configured transaction tax",
        )
    if metadata.metadata_status is not TariffStatus.VERIFIED:
        return CostComponent.unavailable(
            "transaction_tax", currency, metadata.source_reference,
            "instrument tax applicability metadata is not VERIFIED",
        )
    if rule is None:
        return CostComponent.unavailable(
            "transaction_tax", currency, metadata.source_reference,
            "explicit transaction tax rule is missing",
        )
    if side is not rule.applicable_side:
        return CostComponent.not_applicable(
            "transaction_tax", currency, rule.source_reference,
            f"rule applies only to {rule.applicable_side.value}",
        )
    amount = notional * rule.rate_bps / BPS
    if rule.covers(timestamp) and rule.status is TariffStatus.VERIFIED:
        return CostComponent.known(
            "transaction_tax", amount, currency, rule.source_reference
        )
    if allow_retrospective:
        return CostComponent.estimated(
            "transaction_tax", amount, currency, rule.source_reference,
            "CURRENT_TARIFF_APPLIED_RETROSPECTIVELY",
        )
    return CostComponent.unavailable(
        "transaction_tax", currency, rule.source_reference,
        "tax rule does not cover decision timestamp",
    )
