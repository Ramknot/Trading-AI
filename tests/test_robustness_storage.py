from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_ai.robustness.config import (
    load_research_baseline_manifest,
    load_research_plan,
    load_robustness_config,
)
from trading_ai.robustness.diagnostics import RobustnessAnalyzer
from trading_ai.robustness.exceptions import RobustnessStorageError
from trading_ai.robustness.models import (
    CampaignStatus,
    HoldoutStatus,
    PeriodClassification,
)
from trading_ai.robustness.readiness import PaperReadinessReviewer
from trading_ai.robustness.storage import LocalRobustnessStore


def _report():
    baseline = load_research_baseline_manifest()
    plan = load_research_plan(baseline=baseline)
    summary = {
        "run_id": "bt-synthetic-mechanics",
        "initial_cash": "100",
        "dataset_references": [
            {
                "symbol": "AAPL",
                "timeframe": "1d",
                "dataset_id": "dataset-synthetic",
                "checksum_sha256": "a" * 64,
                "actual_start": "2010-01-01T00:00:00+00:00",
                "actual_end": "2010-01-02T00:00:00+00:00",
                "provider": "fake",
            }
        ],
        "data_quality_reports": [
            {
                "symbol": "AAPL",
                "timeframe": "1d",
                "row_count": 2,
                "first_timestamp": "2010-01-01T00:00:00+00:00",
                "last_timestamp": "2010-01-02T00:00:00+00:00",
                "quality_status": "PASS",
            }
        ],
        "metrics": {"number_of_trades": 0, "max_drawdown_pct": 0},
        "validation": {"status": "BLOCKED_EXTERNAL_DATA"},
    }
    tables = {
        "equity": (
            {"timestamp": "2010-01-01T00:00:00+00:00", "equity": "100", "positions_value": "0"},
            {"timestamp": "2010-01-02T00:00:00+00:00", "equity": "100", "positions_value": "0"},
        ),
        "trades": (), "fills": (), "signals": (), "cost_actuals": (),
        "cost_estimates": (), "cost_reconciliation": (),
    }
    report = RobustnessAnalyzer(load_robustness_config()).analyze(
        summary=summary,
        tables=tables,
        integrity_verified=True,
        baseline=baseline,
        plan=plan,
        period_classification=PeriodClassification.DIAGNOSTIC,
    )
    readiness = PaperReadinessReviewer().review(
        report, validation_status="BLOCKED_EXTERNAL_DATA"
    )
    return baseline, plan, report, readiness


def test_schema_17_report_is_checksum_verified_and_path_safe(tmp_path: Path) -> None:
    baseline, plan, report, readiness = _report()
    store = LocalRobustnessStore(tmp_path / "robustness")
    directory = store.save_report(
        report, baseline=baseline, plan=plan, readiness=readiness
    )
    inspected = store.inspect_report(report.report_id)
    assert inspected["schema_version"] == "1.7"
    assert inspected["checksums_verified"] is True
    assert len(inspected["cost_robustness"]["evidence_registry_hash"]) == 64
    assert inspected["cost_robustness"]["annual_tax_eligibility_records"] == 14
    assert (directory / "single_strategy_runs.json").is_file()
    assert len(inspected["single_strategy_runs"]) == 4
    assert inspected["paper_readiness"]["unlocks_paper_or_live"] is False
    with pytest.raises(RobustnessStorageError, match="invalid"):
        store.inspect_report("../outside")

    report_path = directory / "robustness_report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(RobustnessStorageError, match="checksum mismatch"):
        store.inspect_report(report.report_id)


def test_final_holdout_campaign_preserves_validation_gate_failure() -> None:
    baseline = load_research_baseline_manifest()
    plan = load_research_plan(baseline=baseline)
    summary = {
        "run_id": "bt-final-holdout-mechanics",
        "initial_cash": "100",
        "dataset_references": (),
        "data_quality_reports": (),
        "metrics": {"number_of_trades": 30, "max_drawdown_pct": 0},
    }
    report = RobustnessAnalyzer(load_robustness_config()).analyze(
        summary=summary,
        tables={"equity": (), "trades": (), "fills": (), "signals": ()},
        integrity_verified=True,
        baseline=baseline,
        plan=plan,
        period_classification=PeriodClassification.FINAL_HOLDOUT,
        holdout_status=HoldoutStatus.CONSUMED,
        validation_status="FAIL",
    )
    assert report.campaign_status is CampaignStatus.FAIL
    assert "VALIDATION_GATE_FAIL" in report.warnings
    assert "SYMBOL_RESULT_CONCENTRATION" not in report.warnings
    assert "BASELINE_V1_SYMBOL_RESULT_CONCENTRATION" in report.warnings
