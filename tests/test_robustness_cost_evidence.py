from __future__ import annotations

from datetime import datetime, timezone

from trading_ai.robustness.cost_evidence import (
    EvidenceKind,
    EvidenceStatus,
    load_historical_cost_evidence,
)


def test_historical_cost_registry_is_hashed_dated_and_offline() -> None:
    registry = load_historical_cost_evidence()
    assert len(registry.registry_hash) == 64
    assert registry.registry_hash == load_historical_cost_evidence().registry_hash
    assert registry.verified_at == datetime(2026, 8, 30, tzinfo=timezone.utc)
    assert registry.broker_tariffs[0].status is EvidenceStatus.HISTORICAL_TARIFF_UNVERIFIED
    assert registry.broker_tariffs[0].evidence_kind is EvidenceKind.CURRENT_OFFICIAL_SOURCE
    assert registry.broker_tariffs[0].warning == "CURRENT_TARIFF_APPLIED_RETROSPECTIVELY"


def test_french_ftt_rate_changes_on_first_april_2025() -> None:
    registry = load_historical_cost_evidence()
    before = registry.tax_rate_at(datetime(2025, 3, 31, tzinfo=timezone.utc))
    after = registry.tax_rate_at(datetime(2025, 4, 1, tzinfo=timezone.utc))
    assert before is not None and before.rate_bps == 30
    assert after is not None and after.rate_bps == 40
    assert before.evidence_kind is EvidenceKind.ARCHIVED_OFFICIAL_SOURCE


def test_tax_eligibility_is_annual_explicit_and_never_inferred_from_ticker_suffix() -> None:
    registry = load_historical_cost_evidence()
    timestamp = datetime(2023, 6, 1, tzinfo=timezone.utc)
    lvmh = registry.eligibility_at("MC.PA", timestamp)
    airbus = registry.eligibility_at("AIR.PA", timestamp)
    unknown = registry.eligibility_at("UNKNOWN.PA", timestamp)
    assert lvmh is not None and lvmh.issuer == "LVMH" and lvmh.eligible is True
    assert airbus is not None and airbus.issuer == "Airbus SE" and airbus.eligible is False
    assert unknown is None

    holdout_2026 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert registry.eligibility_at("MC.PA", holdout_2026).eligible is True
    assert registry.eligibility_at("AIR.PA", holdout_2026).eligible is False


def test_unknown_exchange_and_operating_costs_never_become_numeric_zero() -> None:
    registry = load_historical_cost_evidence()
    assert registry.exchange_fee_status is EvidenceStatus.UNAVAILABLE
    scenarios = dict(registry.operating_scenarios)
    local = {item.component: item for item in scenarios["LOCAL_RESEARCH"]}
    paper = {item.component: item for item in scenarios["PAPER_ESTIMATE"]}
    assert local["market_data_subscription"].amount is None
    assert local["server_vps"].status is EvidenceStatus.NOT_APPLICABLE
    assert local["server_vps"].amount == 0
    assert all(item.amount is None for item in paper.values())


def test_official_evidence_registry_has_no_blog_forum_or_tls_bypass() -> None:
    registry = load_historical_cost_evidence()
    references = [item.source_reference for item in registry.broker_tariffs]
    references += [item.source_reference for item in registry.tax_rates]
    references += [item.source_reference for item in registry.tax_eligibility]
    assert all("interactivebrokers.com" in item or "bofip.impots.gouv.fr" in item for item in references)
    assert not any("verify=false" in item.lower() for item in references)
