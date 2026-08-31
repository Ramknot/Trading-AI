from __future__ import annotations

import ast
from pathlib import Path

from trading_ai.monitoring.models import MonitoringEventType
from trading_ai.monitoring.security import redact_sensitive
from trading_ai.monitoring.source import BacktestMonitoringData, BacktestMonitoringSource


def test_monitoring_event_contract_reserves_all_engine_event_types() -> None:
    assert {item.value for item in MonitoringEventType} == {
        "DATA_QUALITY", "FEATURE", "REGIME", "SIGNAL", "ML_PREDICTION",
        "ML_DECISION", "ACTIVATION_DECISION", "PORTFOLIO_DECISION",
        "RISK_DECISION", "ORDER_INTENT", "FILL", "POSITION_UPDATE",
        "EQUITY_UPDATE", "SYSTEM_HEALTH", "COST_ESTIMATE", "COST_ACTUAL",
        "ECONOMIC_DECISION", "COST_RECONCILIATION", "VALIDATION_RESULT",
        "EVIDENCE_VERIFIED", "EVIDENCE_CONFLICT", "EVIDENCE_REASSESSMENT",
        "PAPER_READINESS_REVIEW",
        "ECONOMIC_RECOMPUTATION_STARTED", "ECONOMIC_RECOMPUTATION_COMPLETED",
        "DECISION_INVARIANCE_CHECK", "PAPER_READINESS_V3",
        "HUMAN_READINESS_REVIEW",
    }


def test_monitoring_modules_do_not_import_brokers_providers_or_network_clients() -> None:
    root = Path("src/trading_ai/monitoring")
    forbidden = (
        "trading_ai.brokers",
        "trading_ai.data.providers",
        "yfinance",
        "requests",
        "ibapi",
    )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(name.startswith(forbidden) for name in imported), (path, imported)


def test_dashboard_assets_have_no_remote_cdn_or_trading_mutation_controls() -> None:
    root = Path("src/trading_ai/monitoring")
    content = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in ("templates/*.html", "static/*.js", "static/*.css")
        for path in root.glob(pattern)
    ).lower()
    content = content.replace("http://www.w3.org/2000/svg", "")
    for forbidden in (
        "https://", "http://", "place_order", "submit_order", "force_live",
        "enable_live", "brokeradapter",
    ):
        assert forbidden not in content
    assert "innerhtml" not in content


def test_observability_redacts_secret_shaped_fields_before_rendering() -> None:
    payload = redact_sensitive(
        {
            "model_id": "model-1",
            "access_token": "do-not-display",
            "nested": {"broker-password": "do-not-display"},
        }
    )
    assert payload == {
        "model_id": "model-1",
        "access_token": "[REDACTED]",
        "nested": {"broker-password": "[REDACTED]"},
    }


def test_decision_invariance_event_identity_is_scoped_to_recomputation() -> None:
    def event_id(recomputation_id: str) -> str:
        data = BacktestMonitoringData(
            run_id="bt-monitoring-fixture",
            schema_version="1.6",
            source_fingerprint="a" * 64,
            cache_token="b" * 64,
            summary={
                "result_hash": "c" * 64,
                "economic_recomputation": {
                    "recomputation_id": recomputation_id,
                    "created_at": "2026-08-31T00:00:00+00:00",
                    "assessment_status": "PASS",
                    "decision_invariance": {
                        "report_hash": "d" * 64,
                        "status": "STRICTLY_INVARIANT",
                    },
                    "paper_readiness_v3": {
                        "readiness_id": f"readiness-{recomputation_id}",
                        "status": "READY_FOR_REVIEW",
                    },
                    "human_review": {
                        "review_event_id": f"human-{recomputation_id}",
                        "recorded_at": "2026-08-31T00:00:00+00:00",
                        "status": "AWAITING_HUMAN_REVIEW",
                    },
                },
            },
            tables={},
        )
        return next(
            item.event_id
            for item in BacktestMonitoringSource.events_for_data(data)
            if item.event_type is MonitoringEventType.DECISION_INVARIANCE_CHECK
        )

    assert event_id("recompute-a") != event_id("recompute-b")
