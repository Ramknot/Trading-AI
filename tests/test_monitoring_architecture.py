from __future__ import annotations

import ast
from pathlib import Path

from trading_ai.monitoring.models import MonitoringEventType
from trading_ai.monitoring.security import redact_sensitive


def test_monitoring_event_contract_reserves_all_engine_event_types() -> None:
    assert {item.value for item in MonitoringEventType} == {
        "DATA_QUALITY", "FEATURE", "REGIME", "SIGNAL", "ML_PREDICTION",
        "ML_DECISION", "ACTIVATION_DECISION", "PORTFOLIO_DECISION",
        "RISK_DECISION", "ORDER_INTENT", "FILL", "POSITION_UPDATE",
        "EQUITY_UPDATE", "SYSTEM_HEALTH", "COST_ESTIMATE", "COST_ACTUAL",
        "ECONOMIC_DECISION", "COST_RECONCILIATION", "VALIDATION_RESULT",
        "EVIDENCE_VERIFIED", "EVIDENCE_CONFLICT", "EVIDENCE_REASSESSMENT",
        "PAPER_READINESS_REVIEW",
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
