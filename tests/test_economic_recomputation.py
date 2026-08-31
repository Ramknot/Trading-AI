from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from trading_ai.core.hashing import stable_hash
from trading_ai.monitoring.source import BacktestMonitoringData
from trading_ai.robustness.economic_recomputation import (
    DecisionInvarianceStatus,
    EconomicEvidenceCompletenessV3,
    EconomicRecomputationEngine,
    HumanReviewStatus,
    PaperReadinessReviewerV3,
    Section31RuleBook,
    classify_decision_invariance,
    load_economic_recomputation_config,
    make_human_review_decision,
    make_initial_human_review,
)
from trading_ai.robustness.evidence import (
    EconomicEvidenceStatus,
    load_evidence_registry_v2,
    load_paper_operating_scenarios,
)
from trading_ai.robustness.exceptions import RobustnessError, RobustnessStorageError
from trading_ai.robustness.governance import decision_core_hash
from trading_ai.robustness.models import (
    HoldoutRecord,
    HoldoutStatus,
    PaperReadinessStatus,
    PeriodClassification,
    ResearchPeriod,
)
from trading_ai.robustness.storage import LocalRobustnessStore
from trading_ai.validation import load_validation_config


UTC = timezone.utc
SHA = "a" * 64
EXPECTED_CORE = "16a9e8ae3be7f5d4ca6b68d56d14a988bff64ec74d83d4cad0285ab5d233052e"


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _scenario():
    return next(
        item
        for item in load_paper_operating_scenarios()
        if item.scenario_id == "PAPER_ESTIMATE_V1"
    )


def _run() -> BacktestMonitoringData:
    buy_time = "2026-04-05T14:30:00+00:00"
    sell_time = "2026-04-10T14:30:00+00:00"
    return BacktestMonitoringData(
        run_id="bt-section31-fixture",
        schema_version="1.6",
        source_fingerprint=SHA,
        cache_token=SHA,
        summary={
            "run_id": "bt-section31-fixture",
            "result_hash": SHA,
            "initial_cash": "1000",
            "costs": {
                "engine_name": "balanced-transaction-cost",
                "engine_version": "1.0",
                "config_hash": "b" * 64,
                "summary": {
                    "total_variable_cost": "10",
                    "gross_trading_pnl": "100",
                    "net_trading_pnl_before_operating": "90",
                    "net_return_before_operating": "0.09",
                },
            },
            "metrics": {
                "number_of_trades": 2,
                "max_drawdown_pct": "0",
                "profit_factor": "10",
                "expectancy": "45",
            },
            "dataset_references": [],
            "regime": {},
        },
        tables={
            "equity": (
                {"timestamp": buy_time, "cash": "900", "equity": "1000"},
                {"timestamp": sell_time, "cash": "1090", "equity": "1090"},
            ),
            "fills": (
                {
                    "fill_id": "fill-buy",
                    "order_id": "order-buy",
                    "symbol": "SPY",
                    "side": "BUY",
                    "timestamp": buy_time,
                    "quantity": "2",
                    "price": "90",
                    "exchange_fees": "0",
                    "total_variable_cost": "1",
                },
                {
                    "fill_id": "fill-sell",
                    "order_id": "order-sell",
                    "symbol": "SPY",
                    "side": "SELL",
                    "timestamp": sell_time,
                    "quantity": "2",
                    "price": "100",
                    "exchange_fees": "0",
                    "total_variable_cost": "1",
                },
            ),
            "ledger": (
                {"reference_id": "fill-buy", "cash_change": "-181"},
                {"reference_id": "fill-sell", "cash_change": "199"},
            ),
            "trades": (
                {
                    "trade_id": "trade-win",
                    "symbol": "SPY",
                    "quantity": "2",
                    "exit_time": sell_time,
                    "net_pnl": "100",
                },
                {
                    "trade_id": "trade-loss",
                    "symbol": "AAPL",
                    "quantity": "1",
                    "exit_time": "2026-03-01T14:30:00+00:00",
                    "net_pnl": "-10",
                },
            ),
            "cost_estimates": (
                {
                    "estimate_id": "estimate-buy",
                    "symbol": "SPY",
                    "side": "BUY",
                    "timestamp": buy_time,
                    "quantity": "2",
                    "reference_price": "90",
                },
                {
                    "estimate_id": "estimate-sell",
                    "symbol": "SPY",
                    "side": "SELL",
                    "timestamp": sell_time,
                    "quantity": "2",
                    "reference_price": "100",
                },
            ),
            "economic_decisions": (
                {
                    "decision_id": "economic-buy",
                    "order_id": "order-buy",
                    "cost_estimate_id": "estimate-buy",
                    "status": "INCOMPLETE",
                    "allows_new_risk": True,
                    "estimated_round_trip_cost_bps": "20",
                    "expected_gross_edge_bps": None,
                },
                {
                    "decision_id": "economic-sell",
                    "order_id": "order-sell",
                    "cost_estimate_id": "estimate-sell",
                    "status": "NOT_APPLICABLE",
                    "allows_new_risk": True,
                    "estimated_round_trip_cost_bps": None,
                    "expected_gross_edge_bps": None,
                },
            ),
            "regime_snapshots": (),
            "signals": (),
            "ml_predictions": (),
            "ml_decisions": (),
            "activation_decisions": (),
            "portfolio_opportunities": (),
            "portfolio_decisions": (),
            "portfolio_targets": (),
            "risk_decisions": (),
            "orders": (),
        },
    )


def _holdout() -> HoldoutRecord:
    period = ResearchPeriod(
        name="final_holdout",
        classification=PeriodClassification.FINAL_HOLDOUT,
        start=_dt("2025-01-02T00:00:00"),
        end=_dt("2026-08-28T00:00:00"),
        label="FINAL_HOLDOUT_V2",
    )
    semantic = {
        "holdout_id": "holdout-fixture",
        "plan_hash": "c" * 64,
        "period": period,
        "status": HoldoutStatus.CONSUMED,
        "expected_core_hash": decision_core_hash(),
        "expected_config_hashes": (),
        "consumed_at": _dt("2026-08-30T00:00:00"),
        "result_hash": SHA,
    }
    return HoldoutRecord(record_hash=stable_hash(semantic), **semantic)


def _report(*, robustness_bundle=None):
    registry = load_evidence_registry_v2()
    _, validation_config_hash = load_validation_config()
    if robustness_bundle is None:
        robustness_bundle = {
            "run_id": "bt-section31-fixture",
            "report_id": "robustness-fixture",
            "baseline_manifest_hash": "d" * 64,
            "plan_hash": "c" * 64,
            "research_baseline": {"manifest_hash": "d" * 64},
            "robustness_plan": {
                "plan_hash": "c" * 64,
                "baseline_manifest_hash": "d" * 64,
            },
        }
    return EconomicRecomputationEngine().recompute(
        run=_run(),
        robustness_bundle=robustness_bundle,
        validation={
            "run_id": "bt-section31-fixture",
            "validation_id": "validation-fixture",
            "config_hash": validation_config_hash,
        },
        evidence_reassessment={
            "reassessment_id": "reassessment-fixture",
            "holdout_status": "CONSUMED",
            "mode": "ECONOMIC_RECOMPUTATION_REQUIRED",
            "evidence_registry_hash": registry.registry_hash,
        },
        holdout=_holdout(),
        registry=registry,
        operating_scenario=_scenario(),
        config=load_economic_recomputation_config(),
        created_at=_dt("2026-08-31T00:00:00"),
    )


def test_decision_core_stays_at_frozen_lot_83_hash() -> None:
    assert decision_core_hash() == EXPECTED_CORE


def test_recomputation_verifies_embedded_baseline_and_plan_hashes() -> None:
    bundle = {
        "run_id": "bt-section31-fixture",
        "report_id": "robustness-fixture",
        "baseline_manifest_hash": "d" * 64,
        "plan_hash": "c" * 64,
        "research_baseline": {"manifest_hash": "e" * 64},
        "robustness_plan": {
            "plan_hash": "c" * 64,
            "baseline_manifest_hash": "d" * 64,
        },
    }
    with pytest.raises(RobustnessError, match="ResearchBaselineManifest hash"):
        _report(robustness_bundle=bundle)


def test_section31_is_point_in_time_side_and_market_scoped() -> None:
    rules = Section31RuleBook(
        load_evidence_registry_v2(), load_economic_recomputation_config()
    )
    zero = rules.calculate(
        timestamp=_dt("2025-06-01T00:00:00"), side="SELL", market="US",
        notional=Decimal("100000"), covered_instrument=True,
    )
    after = rules.calculate(
        timestamp=_dt("2026-04-05T00:00:00"), side="SELL", market="US",
        notional=Decimal("100000"), covered_instrument=True,
    )
    buy = rules.calculate(
        timestamp=_dt("2026-04-05T00:00:00"), side="BUY", market="US",
        notional=Decimal("100000"), covered_instrument=True,
    )
    foreign = rules.calculate(
        timestamp=_dt("2026-04-05T00:00:00"), side="SELL", market="FR",
        notional=Decimal("100000"), covered_instrument=False,
    )
    assert zero.amount == Decimal("0E-8")
    assert after.amount == Decimal("2.06000000")
    assert after.rate_per_million == Decimal("20.60")
    assert buy.status == foreign.status == "NOT_APPLICABLE"
    assert buy.amount == foreign.amount == Decimal("0")


def test_section31_does_not_leak_a_future_regulatory_rate() -> None:
    rules = Section31RuleBook(
        load_evidence_registry_v2(), load_economic_recomputation_config()
    )
    before = rules.calculate(
        timestamp=_dt("2026-04-03T23:59:59"), side="SELL", market="US",
        notional=Decimal("1000000"), covered_instrument=True,
    )
    after = rules.calculate(
        timestamp=_dt("2026-04-04T00:00:00"), side="SELL", market="US",
        notional=Decimal("1000000"), covered_instrument=True,
    )
    assert before.amount == Decimal("0E-8")
    assert after.amount == Decimal("20.60000000")


def test_recomputation_is_additive_once_and_preserves_original() -> None:
    run = _run()
    original = tuple(dict(item) for item in run.tables["fills"])
    report = _report()
    affected = report.affected_fills
    assert len(affected) == 1
    expected = (Decimal("200") * Decimal("20.60") / Decimal("1000000")).quantize(
        Decimal("0.00000001")
    )
    assert affected[0].section31_cost == expected
    assert affected[0].recomputed_exchange_fees == expected
    assert affected[0].recomputed_total_variable_cost == Decimal("1") + expected
    assert report.metrics.pnl_delta == -expected
    assert report.metrics.recomputed_net_pnl_before_operating == Decimal("90") - expected
    assert report.metrics.recomputed_section31 == expected
    assert run.tables["fills"] == original
    assert report.original_exports_immutable is True


def test_recomputed_cash_equity_trade_and_metrics_reconcile() -> None:
    report = _report()
    fee = report.metrics.recomputed_section31
    assert report.recomputed_equity[-1].recomputed_cash == Decimal("1090") - fee
    assert report.recomputed_equity[-1].recomputed_equity == Decimal("1090") - fee
    assert report.affected_trades[0].recomputed_net_pnl == Decimal("100") - fee
    assert report.metrics.recomputed_expectancy == (Decimal("90") - fee) / Decimal("2")
    assert report.metrics.recomputed_minimum_cash >= 0
    assert report.decision_invariance.status is DecisionInvarianceStatus.STRICTLY_INVARIANT
    assert all(item.invariant for item in report.decision_invariance.layers)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        ({}, DecisionInvarianceStatus.STRICTLY_INVARIANT),
        ({"economic_changed": True}, DecisionInvarianceStatus.ECONOMIC_DECISION_CHANGED),
        ({"risk_changed": True}, DecisionInvarianceStatus.RISK_DECISION_CHANGED),
        ({"order_or_fill_changed": True}, DecisionInvarianceStatus.ORDER_OR_FILL_CHANGED),
        ({"core_changed": True}, DecisionInvarianceStatus.DECISION_CORE_CHANGED),
    ),
)
def test_decision_invariance_precedence(kwargs, expected) -> None:
    flags = dict(
        core_changed=False,
        economic_changed=False,
        risk_changed=False,
        order_or_fill_changed=False,
    )
    flags.update(kwargs)
    assert classify_decision_invariance(**flags) is expected


def test_readiness_v3_is_read_only_and_human_review_is_explicit() -> None:
    report = _report()
    ready_metrics = replace(
        report.metrics,
        closed_trades=38,
        recomputed_net_return=Decimal("0.10"),
        recomputed_expectancy=Decimal("1"),
        recomputed_profit_factor=Decimal("2"),
        recomputed_max_drawdown=Decimal("0.03"),
        recomputed_minimum_cash=Decimal("500"),
    )
    ready_report = replace(report, metrics=ready_metrics)
    readiness = PaperReadinessReviewerV3().review(
        ready_report, created_at=_dt("2026-08-31T00:00:00")
    )
    assert readiness.status is PaperReadinessStatus.READY_FOR_REVIEW
    assert readiness.human_review_status is HumanReviewStatus.AWAITING_HUMAN_REVIEW
    assert readiness.unlocks_paper_or_live is False
    initial = make_initial_human_review(
        readiness, recorded_at=_dt("2026-08-31T00:00:00")
    )
    assert initial.status is HumanReviewStatus.AWAITING_HUMAN_REVIEW
    with pytest.raises(RobustnessError, match="reason"):
        make_human_review_decision(
            readiness,
            status=HumanReviewStatus.HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT,
            reason="",
            recorded_at=_dt("2026-08-31T00:00:00"),
        )
    accepted = make_human_review_decision(
        readiness,
        status=HumanReviewStatus.HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT,
        reason="Reviewed evidence; authorize Lot 9 development only.",
        recorded_at=_dt("2026-08-31T00:00:00"),
    )
    assert accepted.authorizes_lot9_development_only is True
    assert accepted.unlocks_paper_or_live is False


def test_readiness_v3_distinguishes_not_ready_and_insufficient_evidence() -> None:
    report = _report()
    not_ready = PaperReadinessReviewerV3().review(
        report, created_at=_dt("2026-08-31T00:00:00")
    )
    assert not_ready.status is PaperReadinessStatus.NOT_READY
    incomplete = EconomicEvidenceCompletenessV3(
        status=EconomicEvidenceStatus.INCOMPLETE,
        component_statuses=(("section31", "UNAVAILABLE"),),
        critical_unresolved=("section31",),
        evidence_ids=(),
        completeness_hash="e" * 64,
    )
    insufficient = PaperReadinessReviewerV3().review(
        replace(report, completeness=incomplete),
        created_at=_dt("2026-08-31T00:00:00"),
    )
    assert insufficient.status is PaperReadinessStatus.INSUFFICIENT_EVIDENCE


def test_readiness_refuses_a_changed_frozen_validation_hash() -> None:
    with pytest.raises(RobustnessError, match="frozen Validation thresholds"):
        PaperReadinessReviewerV3().review(
            replace(_report(), validation_config_hash="f" * 64),
            created_at=_dt("2026-08-31T00:00:00"),
        )


def test_schema_19_bundle_is_checksum_verified_and_original_schema_is_unchanged(
    tmp_path: Path,
) -> None:
    report = _report()
    readiness = PaperReadinessReviewerV3().review(
        report, created_at=_dt("2026-08-31T00:00:00")
    )
    human = make_initial_human_review(
        readiness, recorded_at=_dt("2026-08-31T00:00:00")
    )
    store = LocalRobustnessStore(tmp_path)
    directory = store.save_recomputation_bundle(
        report=report,
        readiness=readiness,
        human_review=human,
        evidence_registry=load_evidence_registry_v2(),
    )
    inspected = store.inspect_recomputation_bundle(report.recomputation_id)
    assert inspected["schema_version"] == "1.9"
    assert inspected["holdout_status"] == "CONSUMED"
    assert inspected["paper_readiness_v3"]["human_review_status"] == "AWAITING_HUMAN_REVIEW"
    with pytest.raises(RobustnessStorageError):
        store.inspect_recomputation_bundle("../outside")
    path = directory / "economic_completeness.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RobustnessStorageError, match="checksum"):
        store.inspect_recomputation_bundle(report.recomputation_id)


def test_human_review_audit_is_separate_immutable_and_never_unlocks_execution(
    tmp_path: Path,
) -> None:
    report = _report()
    ready = PaperReadinessReviewerV3().review(
        replace(report, metrics=replace(report.metrics, closed_trades=38)),
        created_at=_dt("2026-08-31T00:00:00"),
    )
    initial = make_initial_human_review(
        ready, recorded_at=_dt("2026-08-31T00:00:00")
    )
    store = LocalRobustnessStore(tmp_path)
    store.save_recomputation_bundle(
        report=report,
        readiness=ready,
        human_review=initial,
        evidence_registry=load_evidence_registry_v2(),
    )
    accepted = make_human_review_decision(
        ready,
        status=HumanReviewStatus.HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT,
        reason="Authorize implementation review only, with execution still locked.",
        recorded_at=_dt("2026-09-01T00:00:00"),
    )
    store.save_human_review(accepted)
    latest = store.latest_human_review(ready.readiness_id)
    assert latest is not None
    assert latest["status"] == "HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT"
    assert latest["unlocks_paper_or_live"] is False
    assert store.inspect_recomputation_bundle(report.recomputation_id)["human_review"] == latest


def test_same_inputs_produce_same_semantic_recomputation_hash() -> None:
    assert _report().recomputation_hash == _report().recomputation_hash
