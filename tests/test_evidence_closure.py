from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

from trading_ai.core.hashing import stable_hash
from trading_ai.cli import main
from trading_ai.monitoring.source import BacktestMonitoringData
from trading_ai.robustness.evidence import (
    AppliedTariffAssumption,
    EconomicEvidenceStatus,
    EvidenceConflict,
    EvidenceReassessmentMode,
    EvidenceRegistryV2,
    EvidenceSourceType,
    EvidenceVerificationStatus,
    InvarianceStatus,
    OperatingCostRange,
    PaperOperatingScenario,
    TariffCompatibilityStatus,
    TariffEvidenceComparator,
    load_evidence_registry_v2,
    load_paper_operating_scenarios,
)
from trading_ai.robustness.exceptions import HoldoutGovernanceError, RobustnessError
from trading_ai.robustness.models import (
    HoldoutRecord,
    HoldoutStatus,
    PaperReadinessStatus,
    PeriodClassification,
    ResearchPeriod,
)
from trading_ai.robustness.reassessment import (
    EvidenceClosureService,
    EvidenceReassessmentEngine,
    PaperReadinessReviewerV2,
    compare_invariance,
)
from trading_ai.robustness.storage import LocalRobustnessStore


START = datetime(2025, 1, 2, tzinfo=timezone.utc)
END = datetime(2026, 8, 27, tzinfo=timezone.utc)
CORE_HASH = "c" * 64
RESULT_HASH = "a" * 64


def _run(*, fills=(), signal_suffix: str = "1", costs=()) -> BacktestMonitoringData:
    tables = {
        "signals": ({"signal_id": f"signal-{signal_suffix}", "timestamp": START.isoformat()},),
        "ml_decisions": (),
        "activation_decisions": (),
        "portfolio_decisions": (),
        "economic_decisions": (),
        "risk_decisions": (),
        "orders": (),
        "fills": tuple(fills),
        "cost_estimates": (),
        "cost_actuals": tuple(costs),
        "cost_reconciliation": (),
    }
    return BacktestMonitoringData(
        run_id="bt-consumed",
        schema_version="1.6",
        source_fingerprint="f" * 64,
        cache_token="token",
        summary={
            "run_id": "bt-consumed",
            "result_hash": RESULT_HASH,
            "started_at": START.isoformat(),
            "completed_at": END.isoformat(),
            "costs": {
                "summary": {
                    "gross_trading_pnl": "12000",
                    "net_trading_pnl_before_operating": "10800",
                    "total_variable_cost": "1200",
                    "total_commission": "20",
                    "total_exchange_fees": "0",
                    "total_transaction_tax": "0",
                    "total_fx_cost": "0",
                    "total_spread": "590",
                    "total_slippage": "590",
                }
            },
        },
        tables=tables,
    )


def _holdout(status: HoldoutStatus = HoldoutStatus.CONSUMED) -> HoldoutRecord:
    period = ResearchPeriod(
        name="final_holdout",
        classification=PeriodClassification.FINAL_HOLDOUT,
        start=START,
        end=END,
        label="FINAL_HOLDOUT_V2",
    )
    return HoldoutRecord(
        holdout_id="holdout-test",
        plan_hash="b" * 64,
        period=period,
        status=status,
        expected_core_hash=CORE_HASH,
        expected_config_hashes=(),
        record_hash="d" * 64,
        consumed_at=(datetime(2026, 8, 30, tzinfo=timezone.utc) if status is HoldoutStatus.CONSUMED else None),
        result_hash=(RESULT_HASH if status is HoldoutStatus.CONSUMED else None),
    )


def _validation() -> dict[str, object]:
    return {
        "validation_id": "validation-test",
        "status": "FAIL",
        "criteria": [
            {"name": "dataset_integrity", "status": "PASS"},
            {"name": "closed_trades", "status": "PASS"},
            {"name": "net_return", "status": "PASS"},
            {"name": "cash_non_negative", "status": "PASS"},
            {"name": "tariff_period", "status": "FAIL"},
            {"name": "operating_costs", "status": "WARNING"},
        ],
        "stress_results": [
            {"multiplier": "2.00", "stressed_net_pnl": "10000"}
        ],
    }


def _robustness() -> dict[str, object]:
    return {
        "report_id": "robustness-test",
        "survivorship_status": "SURVIVORSHIP_BIAS_UNRESOLVED",
        "concentration": {"top1_positive_pnl_share": "0.79"},
    }


def _exact_tariff(registry: EvidenceRegistryV2):
    reference = next(
        item for item in registry.records if item.evidence_id == "ibkr-us-fixed-2025-01-01"
    )
    assumption = AppliedTariffAssumption(
        provider="Interactive Brokers",
        market="US",
        fee_type="BROKER_COMMISSION",
        plan="IBKR_PRO_FIXED",
        normalized_rules=reference.normalized_rules,
    )
    return TariffEvidenceComparator.compare(
        assumption, registry, period_start=START, period_end=END
    )


def _assess(*, run=None, candidate=None, tariff=None, scenario=None, core=CORE_HASH):
    registry = load_evidence_registry_v2()
    run = run or _run()
    candidate = candidate or run
    tariff = tariff or _exact_tariff(registry)
    scenario = scenario or {
        item.scenario_id: item for item in load_paper_operating_scenarios()
    }["PAPER_ESTIMATE_V1"]
    report = EvidenceReassessmentEngine().assess(
        run=run,
        candidate=candidate,
        robustness_report=_robustness(),
        validation=_validation(),
        holdout=_holdout(),
        registry=registry,
        broker_tariff=tariff,
        operating_scenario=scenario,
        current_core_hash=core,
    )
    return report, PaperReadinessReviewerV2().review(report, _validation()), registry, scenario


def test_registry_v2_is_offline_hashed_dated_and_conflict_free() -> None:
    registry = load_evidence_registry_v2()
    assert registry.registry_version == "2.0"
    assert registry.registry_hash == load_evidence_registry_v2().registry_hash
    assert len(registry.records) == 13
    assert registry.conflicts == ()
    source_types = {item.source_type for item in registry.records}
    assert EvidenceSourceType.ARCHIVED_OFFICIAL_SOURCE in source_types
    assert EvidenceSourceType.REGULATORY_SOURCE in source_types
    assert EvidenceSourceType.CURRENT_OFFICIAL_SOURCE in source_types
    assert all("blog" not in item.source_reference for item in registry.records)


def test_tariff_comparator_exact_conservative_different_and_current_only() -> None:
    registry = load_evidence_registry_v2()
    exact = _exact_tariff(registry)
    assert exact.status is TariffCompatibilityStatus.EXACT_MATCH
    rules = dict(next(item for item in registry.records if item.evidence_id.startswith("ibkr-us-fixed-2025")).normalized_rules)

    conservative_rules = dict(rules)
    conservative_rules.update(per_unit="0.006", minimum_per_order="1.10", maximum_notional_fraction="0.02")
    conservative = TariffEvidenceComparator.compare(
        AppliedTariffAssumption(
            "Interactive Brokers", "US", "BROKER_COMMISSION", "IBKR_PRO_FIXED",
            tuple(sorted(conservative_rules.items())),
        ),
        registry,
        period_start=START,
        period_end=END,
    )
    assert conservative.status is TariffCompatibilityStatus.COMPATIBLE_CONSERVATIVE
    assert "cannot be below" in conservative.mathematical_demonstration

    cheaper_rules = dict(rules)
    cheaper_rules["per_unit"] = "0.004"
    different = TariffEvidenceComparator.compare(
        AppliedTariffAssumption(
            "Interactive Brokers", "US", "BROKER_COMMISSION", "IBKR_PRO_FIXED",
            tuple(sorted(cheaper_rules.items())),
        ),
        registry,
        period_start=START,
        period_end=END,
    )
    assert different.status is TariffCompatibilityStatus.NUMERICALLY_DIFFERENT

    current = next(
        item for item in registry.records if item.source_type is EvidenceSourceType.CURRENT_OFFICIAL_SOURCE
        and item.fee_type == "BROKER_COMMISSION"
    )
    current_only = EvidenceRegistryV2(
        registry_version="2.0",
        acquired_at=registry.acquired_at,
        records=(current,),
        conflicts=(),
        registry_hash="e" * 64,
    )
    insufficient = TariffEvidenceComparator.compare(
        AppliedTariffAssumption(
            "Interactive Brokers", "US", "BROKER_COMMISSION", "IBKR_PRO_FIXED",
            current.normalized_rules,
        ),
        current_only,
        period_start=START,
        period_end=END,
    )
    assert insufficient.status is TariffCompatibilityStatus.INSUFFICIENT_EVIDENCE


def test_conflicting_official_evidence_is_never_selected_silently() -> None:
    registry = load_evidence_registry_v2()
    conflict = EvidenceConflict(
        fee_type="BROKER_COMMISSION",
        market="US",
        plan="IBKR_PRO_FIXED",
        evidence_ids=("official-a", "official-b"),
        reason="scope conflict",
    )
    conflicted = replace(registry, conflicts=(conflict,), registry_hash="9" * 64)
    assessment = TariffEvidenceComparator.compare(
        EvidenceClosureService._tariff_assumption(),
        conflicted,
        period_start=START,
        period_end=END,
    )
    assert assessment.status is TariffCompatibilityStatus.INSUFFICIENT_EVIDENCE
    assert "CONFLICTING_OFFICIAL_EVIDENCE" in assessment.warnings


def test_archived_official_evidence_requires_an_official_original() -> None:
    record = next(
        item
        for item in load_evidence_registry_v2().records
        if item.source_type is EvidenceSourceType.ARCHIVED_OFFICIAL_SOURCE
    )
    with pytest.raises(RobustnessError, match="original official"):
        replace(record, original_source_reference="https://example.net/untrusted")


def test_exact_match_is_evidence_only_and_can_be_ready_for_manual_review() -> None:
    report, readiness, _, _ = _assess()
    assert report.mode is EvidenceReassessmentMode.EVIDENCE_ONLY_RECLASSIFICATION
    assert report.evidence_only is True
    assert report.decision_invariance.status is InvarianceStatus.IDENTICAL
    assert report.cost_invariance.status is InvarianceStatus.IDENTICAL
    assert report.economic_completeness.status is EconomicEvidenceStatus.COMPLETE_ESTIMATED
    assert report.strict_validation_evidence_status == "PASS"
    assert report.holdout_status == "CONSUMED"
    assert readiness.status is PaperReadinessStatus.READY_FOR_REVIEW
    assert readiness.unlocks_paper_or_live is False


def test_conservative_evidence_is_distinct_from_strict_validation() -> None:
    registry = load_evidence_registry_v2()
    exact = _exact_tariff(registry)
    conservative = replace(
        exact,
        status=TariffCompatibilityStatus.COMPATIBLE_CONSERVATIVE,
        mathematical_demonstration="Component-wise higher modeled cost for every positive order.",
    )
    report, readiness, _, _ = _assess(tariff=conservative)
    assert report.mode is EvidenceReassessmentMode.EVIDENCE_ONLY_RECLASSIFICATION
    assert report.strict_validation_evidence_status == "WARNING"
    assert readiness.status is PaperReadinessStatus.READY_FOR_REVIEW
    assert "STRICT_VALIDATION_REMAINS_DISTINCT_FROM_CONSERVATIVE_READINESS" in report.warnings


def test_positive_sec_fee_missing_requires_recomputation_and_not_ready() -> None:
    run = _run(
        fills=(
            {
                "fill_id": "fill-sell",
                "timestamp": "2026-05-01T00:00:00+00:00",
                "side": "SELL",
                "symbol": "AAPL",
                "price": "100",
                "quantity": "1000",
            },
        )
    )
    report, readiness, _, _ = _assess(run=run)
    assert report.broker_tariff.status is TariffCompatibilityStatus.EXACT_MATCH
    assert report.mode is EvidenceReassessmentMode.ECONOMIC_RECOMPUTATION_REQUIRED
    assert report.indicated_missing_regulatory_cost == Decimal("2.06")
    assert report.numeric_recomputation_required is True
    assert report.evidence_only is False
    assert report.holdout_remains_consumed is True
    assert readiness.status is PaperReadinessStatus.NOT_READY


def test_decision_or_cost_change_can_never_be_evidence_only() -> None:
    baseline = _run(costs=({"actual_cost_id": "a", "amount": "1"},))
    changed_decision = _run(signal_suffix="2", costs=baseline.tables["cost_actuals"])
    report, _, _, _ = _assess(run=baseline, candidate=changed_decision)
    assert report.mode is EvidenceReassessmentMode.DECISION_CORE_CHANGED
    assert report.decisions_or_costs_mutated is True

    changed_cost = _run(costs=({"actual_cost_id": "a", "amount": "2"},))
    report, _, _, _ = _assess(run=baseline, candidate=changed_cost)
    assert report.mode is EvidenceReassessmentMode.ECONOMIC_RECOMPUTATION_REQUIRED
    assert report.cost_invariance.status is InvarianceStatus.DIFFERENT

    report, _, _, _ = _assess(run=baseline, candidate=baseline, core="1" * 64)
    assert report.mode is EvidenceReassessmentMode.DECISION_CORE_CHANGED


def test_consumed_holdout_cannot_be_freshened_for_reassessment() -> None:
    registry = load_evidence_registry_v2()
    with pytest.raises(HoldoutGovernanceError, match="CONSUMED"):
        EvidenceReassessmentEngine().assess(
            run=_run(),
            candidate=_run(),
            robustness_report=_robustness(),
            validation=_validation(),
            holdout=_holdout(HoldoutStatus.UNTOUCHED),
            registry=registry,
            broker_tariff=_exact_tariff(registry),
            operating_scenario=load_paper_operating_scenarios()[0],
            current_core_hash=CORE_HASH,
        )


def test_missing_paper_operating_estimate_keeps_readiness_not_ready() -> None:
    base = load_paper_operating_scenarios()[0]
    unavailable = OperatingCostRange(
        component="market_data_subscription",
        status=EvidenceVerificationStatus.UNAVAILABLE,
        currency="USD",
        monthly_low=None,
        monthly_central=None,
        monthly_high=None,
        source_type=EvidenceSourceType.UNAVAILABLE,
        source_reference="UNAVAILABLE",
        notes="No estimate supplied.",
    )
    scenario = PaperOperatingScenario(
        scenario_id="missing-operating",
        scenario_version="1.0",
        deployment_mode="PAPER_REVIEW_ONLY",
        components=(unavailable, *base.components[1:]),
        scenario_hash=stable_hash("missing-operating"),
    )
    report, readiness, _, _ = _assess(scenario=scenario)
    assert report.economic_completeness.status is EconomicEvidenceStatus.INCOMPLETE
    assert readiness.status is PaperReadinessStatus.NOT_READY


def test_operating_ranges_and_break_even_never_turn_unknown_into_zero() -> None:
    scenarios = {item.scenario_id: item for item in load_paper_operating_scenarios()}
    paper = scenarios["PAPER_ESTIMATE_V1"]
    low, central, high = paper.monthly_totals
    assert low == Decimal("10")
    assert central == Decimal("30")
    assert high == Decimal("225")
    report, _, _, _ = _assess(scenario=paper)
    assert report.paper_economics.net_before_operating == Decimal("10800")
    assert report.paper_economics.break_even_monthly_fixed_cost > 0
    assert report.paper_economics.net_after_operating_high < Decimal("10800")


def test_schema_18_evidence_bundle_is_immutable_checksum_verified_and_path_safe(
    tmp_path: Path,
) -> None:
    report, readiness, registry, scenario = _assess()
    store = LocalRobustnessStore(tmp_path / "robustness")
    directory = store.save_evidence_bundle(
        reassessment=report,
        registry=registry,
        completeness=report.economic_completeness,
        operating_scenario=scenario,
        readiness=readiness,
    )
    inspected = store.inspect_evidence_bundle(report.reassessment_id)
    assert inspected["schema_version"] == "1.8"
    assert inspected["checksums_verified"] is True
    assert inspected["paper_readiness_v2"]["unlocks_paper_or_live"] is False
    assert store.latest_evidence_for_run(report.run_id)["reassessment_id"] == report.reassessment_id
    store.save_evidence_bundle(
        reassessment=report,
        registry=registry,
        completeness=report.economic_completeness,
        operating_scenario=scenario,
        readiness=readiness,
    )
    with pytest.raises(Exception, match="invalid"):
        store.inspect_evidence_bundle("../outside")
    path = directory / "economic_completeness.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(Exception, match="checksum mismatch"):
        store.inspect_evidence_bundle(report.reassessment_id)


def test_same_evidence_and_run_produce_same_semantic_hashes() -> None:
    first, first_readiness, _, _ = _assess()
    second, second_readiness, _, _ = _assess()
    assert first.reassessment_hash == second.reassessment_hash
    assert first.reassessment_id == second.reassessment_id
    assert first_readiness.review_hash == second_readiness.review_hash
    assert first.created_at != second.created_at


def test_evidence_cli_is_explicit_offline_and_read_only(tmp_path: Path, capsys) -> None:
    assert main(("evidence", "verify", "--data-root", str(tmp_path), "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["network_access"] is False
    assert payload["runtime_scraping"] is False


def test_invariance_hash_is_table_order_sensitive_but_repeatable() -> None:
    run = _run()
    first = compare_invariance(run, run, kind="DECISIONS", table_names=("signals",))
    second = compare_invariance(run, run, kind="DECISIONS", table_names=("signals",))
    assert first == second
    assert first.status is InvarianceStatus.IDENTICAL
