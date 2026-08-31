from __future__ import annotations

import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backtest_support import bar, dataset
from ml_support import ConstantAdapter, model_artifact
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.models import BacktestConfig
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.cli import build_parser, main
from trading_ai.core.config import load_runtime_settings
from trading_ai.ml.features import MLFeatureBuilder
from trading_ai.ml.inference import InferenceEngine, SignalMLScorer
from trading_ai.ml.models import MLMode
from trading_ai.monitoring.dashboard import DashboardSettings, create_dashboard_app
from trading_ai.monitoring.exceptions import (
    MonitoringConfigurationError,
    MonitoringNotFoundError,
)
from trading_ai.monitoring.service import MonitoringService
from trading_ai.monitoring.source import BacktestMonitoringSource
from trading_ai.monitoring.store import SQLiteMonitoringStore
from trading_ai.portfolio import BalancedPortfolioEngine
from trading_ai.regimes.detector import BalancedRegimeDetector
from trading_ai.regimes.policy import BalancedStrategyActivationPolicy
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.strategies.baselines import MomentumStrategy, TrendFollowingStrategy
from trading_ai.strategies.config import MomentumConfig


PORTFOLIO_FILES = (
    "portfolio_opportunities.parquet",
    "portfolio_decisions.parquet",
    "portfolio_targets.parquet",
    "portfolio_sleeves.parquet",
)
COST_FILES = (
    "cost_estimates.parquet",
    "cost_actuals.parquet",
    "economic_decisions.parquet",
    "cost_reconciliation.parquet",
    "validation_report.json",
)


def _rising(symbol: str, slope: int, count: int = 90):
    values = []
    for index in range(count):
        opening = Decimal("100") + Decimal(index * slope)
        values.append(
            bar(
                index,
                symbol=symbol,
                opening=opening,
                high=opening + Decimal("2"),
                low=opening - Decimal("1"),
                close=opening + Decimal("1"),
            )
        )
    return dataset(values)


def _scorer(strategy: str) -> SignalMLScorer:
    adapter = ConstantAdapter(0.75, MLFeatureBuilder.feature_names(strategy))
    artifact = model_artifact(
        adapter,
        model_id=f"model-{strategy}-monitoring",
        strategy_name=strategy,
    )
    return SignalMLScorer(
        mode=MLMode.SCORE_ONLY,
        inference_engine=InferenceEngine(artifact, adapter),
        threshold=0.55,
    )


@pytest.fixture(scope="module")
def monitoring_export(tmp_path_factory):
    root = tmp_path_factory.mktemp("lot8") / "data_local"
    settings = load_runtime_settings("PAPER", "balanced")
    engine = BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(settings.profile),
        regime_detector=BalancedRegimeDetector.from_profile(settings.profile),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(settings.profile),
        portfolio_engine=BalancedPortfolioEngine.from_profile(settings.profile),
        ml_scorers={"trend": _scorer("trend"), "momentum": _scorer("momentum")},
        code_version="lot8-test",
    )
    result = engine.run(
        (
            TrendFollowingStrategy(("AAPL", "MSFT"), "1d"),
            MomentumStrategy(("AAPL", "MSFT"), "1d", MomentumConfig(top_k=2)),
        ),
        (_rising("AAPL", 1), _rising("MSFT", 2)),
        settings.context,
        BacktestConfig(
            starting_cash=Decimal("100000"),
            primary_timeframe="1d",
            benchmark_symbol="AAPL",
            spread_bps=Decimal("5"),
            slippage_bps=Decimal("5"),
        ),
    )
    BacktestResultStore(root / "backtests").export(result)
    return root, result


@pytest.fixture
def dashboard_client(monitoring_export):
    root, _ = monitoring_export
    return TestClient(create_dashboard_app(data_root=root))


def test_dashboard_settings_are_local_only() -> None:
    assert DashboardSettings().host == "127.0.0.1"
    assert DashboardSettings(host="localhost").host == "localhost"
    with pytest.raises(MonitoringConfigurationError, match="local-only"):
        DashboardSettings(host="0.0.0.0")


def test_verified_source_caches_parquet_but_rechecks_integrity(monitoring_export) -> None:
    root, result = monitoring_export
    source = BacktestMonitoringSource(root / "backtests")
    first = source.load_run(result.run_id)
    parse_count = source.parquet_parse_count
    second = source.load_run(result.run_id)
    assert second is first
    assert source.parquet_parse_count == parse_count
    assert first.schema_version == "1.6"
    assert first.integrity_verified is True


def test_api_exposes_all_read_only_sections_and_full_decision_trace(
    dashboard_client, monitoring_export
) -> None:
    _, result = monitoring_export
    run_id = result.run_id
    routes = (
        "overview", "equity", "portfolio", "strategies", "regimes", "ml",
        "risk", "data-quality", "costs", "validation", "robustness", "decisions", "events", "health",
    )
    for route in routes:
        response = dashboard_client.get(f"/api/v1/{route}", params={"run_id": run_id})
        assert response.status_code == 200, (route, response.text)

    overview = dashboard_client.get("/api/v1/overview", params={"run_id": run_id}).json()
    assert overview["mode"] == "BACKTEST"
    assert overview["integrity"] == "VERIFIED"
    assert overview["cost_coverage_status"] == "INCOMPLETE"

    snapshot = dashboard_client.get("/api/v1/snapshot", params={"run_id": run_id}).json()
    assert snapshot["source_schema_version"] == "1.6"
    assert snapshot["portfolio"]["engine_name"] == "balanced-portfolio"
    assert snapshot["ml"]["mode"] == "SCORE_ONLY"
    assert snapshot["robustness"]["status"] == "UNAVAILABLE"
    assert {item["name"] for item in snapshot["strategies"]["strategies"]} == {
        "trend", "momentum"
    }
    assert snapshot["decision_traces"]
    trace = snapshot["decision_traces"][0]
    assert [item["stage"] for item in trace["steps"]] == [
        "Dataset", "Feature", "Regime", "Strategy", "Signal", "ML",
        "Activation", "Portfolio", "Cost Estimate", "Economic Gate", "Risk",
        "Order", "Fill", "Actual Cost", "Cost Reconciliation",
    ]
    assert all(
        item["status"] == "HEALTHY"
        for item in trace["steps"]
        if item["stage"] not in {
            "Cost Estimate", "Economic Gate", "Actual Cost", "Cost Reconciliation"
        }
    )
    assert all(
        item["status"] == "UNAVAILABLE"
        for item in trace["steps"]
        if item["stage"] in {
            "Cost Estimate", "Economic Gate", "Actual Cost", "Cost Reconciliation"
        }
    )
    events = dashboard_client.get(
        "/api/v1/events", params={"run_id": run_id, "event_type": "RISK_DECISION"}
    ).json()["events"]
    assert events
    assert all(item["event_type"] == "RISK_DECISION" for item in events)


def test_decision_history_filters_are_deterministic(dashboard_client, monitoring_export) -> None:
    _, result = monitoring_export
    first = dashboard_client.get(
        "/api/v1/decisions",
        params={"run_id": result.run_id, "component": "Risk"},
    ).json()["decisions"]
    second = dashboard_client.get(
        "/api/v1/decisions",
        params={"run_id": result.run_id, "component": "Risk"},
    ).json()["decisions"]
    assert first == second
    assert first
    assert all(item["component"] == "Risk" for item in first)


def test_cost_endpoint_keeps_unimplemented_components_unavailable(
    dashboard_client, monitoring_export
) -> None:
    _, result = monitoring_export
    costs = dashboard_client.get(
        "/api/v1/costs", params={"run_id": result.run_id}
    ).json()
    assert costs["trading"]["commission"]["status"] == "KNOWN"
    assert costs["trading"]["spread"]["status"] == "KNOWN"
    assert costs["trading"]["slippage"]["status"] == "KNOWN"
    assert costs["trading"]["transaction_tax"] == {
        "status": "UNAVAILABLE",
        "amount": None,
        "source": "transaction-cost component unavailable in this export",
    }
    assert costs["trading"]["fx_cost"]["amount"] is None
    assert costs["coverage_status"] == "INCOMPLETE"
    assert costs["net_pnl_estimated"] is None


def test_dashboard_html_is_responsive_escaped_and_has_no_trading_controls(
    dashboard_client, monitoring_export
) -> None:
    _, result = monitoring_export
    response = dashboard_client.get("/", params={"run_id": result.run_id})
    assert response.status_code == 200
    html = response.text
    for section in (
        "Overview", "Portfolio", "Strategies", "Regimes", "ML", "Risk",
        "Data Quality", "Costs", "Robustness", "System Health", "Decision Trace",
    ):
        assert section in html
    assert "READ ONLY · LOCAL" in html
    assert "<button" not in html.lower()
    assert "place_order" not in html
    assert "submit_order" not in html
    assert "force_live" not in html
    escaped = dashboard_client.get("/", params={"run_id": '<script>alert("x")</script>'})
    assert '<script>alert("x")</script>' not in escaped.text
    assert "&lt;script&gt;" in escaped.text


def test_unknown_path_traversal_and_corrupt_runs_fail_cleanly(
    dashboard_client, monitoring_export, tmp_path
) -> None:
    assert dashboard_client.get("/api/v1/overview", params={"run_id": "missing"}).status_code == 404
    for run_id in ("../outside", "..\\outside", str(Path.cwd().resolve())):
        response = dashboard_client.get("/api/v1/overview", params={"run_id": run_id})
        assert response.status_code == 404

    original_root, result = monitoring_export
    corrupt_root = tmp_path / "data_local"
    shutil.copytree(original_root / "backtests", corrupt_root / "backtests")
    summary = corrupt_root / "backtests" / result.run_id / "summary.json"
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    corrupt_client = TestClient(create_dashboard_app(data_root=corrupt_root))
    response = corrupt_client.get("/api/v1/overview", params={"run_id": result.run_id})
    assert response.status_code == 409
    assert "untrusted" in response.json()["error"]


def test_cached_dashboard_invalidates_and_refuses_a_run_changed_after_first_read(
    monitoring_export, tmp_path
) -> None:
    original_root, result = monitoring_export
    changed_root = tmp_path / "data_local"
    shutil.copytree(original_root / "backtests", changed_root / "backtests")
    client = TestClient(create_dashboard_app(data_root=changed_root))
    assert client.get(
        "/api/v1/overview", params={"run_id": result.run_id}
    ).status_code == 200
    fills = changed_root / "backtests" / result.run_id / "fills.parquet"
    fills.write_bytes(fills.read_bytes() + b"tampered")
    response = client.get("/api/v1/overview", params={"run_id": result.run_id})
    assert response.status_code == 409


def _downgrade(directory: Path, version: str) -> None:
    file_minimum_version = {
        "signals.parquet": "1.1",
        "risk_decisions.parquet": "1.2",
        "risk_states.parquet": "1.2",
        "regime_snapshots.parquet": "1.3",
        "regime_transitions.parquet": "1.3",
        "activation_decisions.parquet": "1.3",
        "ml_predictions.parquet": "1.4",
        "ml_decisions.parquet": "1.4",
        **{name: "1.5" for name in PORTFOLIO_FILES},
        **{name: "1.6" for name in COST_FILES},
    }
    removed = tuple(
        name for name, minimum in file_minimum_version.items() if float(version) < float(minimum)
    )
    for name in removed:
        (directory / name).unlink()
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = version
    count_minimum_version = {
        "signals": "1.1", "risk_decisions": "1.2", "risk_state_transitions": "1.2",
        "regime_snapshots": "1.3", "regime_transitions": "1.3",
        "activation_decisions": "1.3", "ml_predictions": "1.4",
        "ml_decisions": "1.4", "portfolio_opportunities": "1.5",
        "portfolio_decisions": "1.5", "portfolio_plans": "1.5",
        "portfolio_targets": "1.5", "portfolio_sleeves": "1.5",
        "cost_estimates": "1.6", "cost_actuals": "1.6",
        "economic_decisions": "1.6", "cost_reconciliations": "1.6",
    }
    for name, minimum in count_minimum_version.items():
        if float(version) < float(minimum):
            summary["counts"].pop(name, None)
    for name, minimum in (("risk", "1.2"), ("regime", "1.3"), ("ml", "1.4"), ("portfolio", "1.5")):
        if float(version) < float(minimum):
            summary.pop(name, None)
    if float(version) < 1.6:
        summary.pop("costs", None)
        summary.pop("validation", None)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = directory / "checksums.json"
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    for name in removed:
        checksums["files"].pop(name)
    checksums["files"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "version", ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6")
)
def test_dashboard_opens_all_legacy_schemas_with_unavailable_sections(
    monitoring_export, tmp_path, version
) -> None:
    original_root, result = monitoring_export
    legacy_root = tmp_path / "data_local"
    shutil.copytree(original_root / "backtests", legacy_root / "backtests")
    _downgrade(legacy_root / "backtests" / result.run_id, version)
    client = TestClient(create_dashboard_app(data_root=legacy_root))
    response = client.get("/api/v1/snapshot", params={"run_id": result.run_id})
    assert response.status_code == 200
    payload = response.json()
    assert payload["source_schema_version"] == version
    if float(version) < 1.3:
        assert payload["regimes"]["status"] == "UNAVAILABLE"
    if float(version) < 1.4:
        assert payload["ml"]["status"] == "unavailable / not used"
    if float(version) < 1.5:
        assert payload["portfolio"]["status"] == "UNAVAILABLE"
    else:
        assert payload["portfolio"]["engine_name"] == "balanced-portfolio"
    assert payload["validation"]["status"] == "UNAVAILABLE"
    if float(version) < 1.2:
        assert payload["risk"]["status"] == "UNAVAILABLE"


def test_same_run_produces_same_monitoring_snapshot(monitoring_export) -> None:
    root, result = monitoring_export
    service = MonitoringService(
        BacktestMonitoringSource(root / "backtests"),
        SQLiteMonitoringStore(root / "monitoring" / "determinism.db"),
    )
    first = service.inspect(result.run_id)
    second = service.inspect(result.run_id)
    assert first == second


def test_monitoring_source_raises_domain_errors_without_exposing_arbitrary_paths(
    monitoring_export
) -> None:
    root, _ = monitoring_export
    source = BacktestMonitoringSource(root / "backtests")
    with pytest.raises(MonitoringNotFoundError, match="invalid backtest run_id"):
        source.load_run("../secret")
    with pytest.raises(MonitoringNotFoundError, match="not found"):
        source.load_run("valid-but-missing")


def test_dashboard_and_monitoring_cli_are_read_only_and_local(
    monitoring_export, capsys
) -> None:
    root, result = monitoring_export
    args = build_parser().parse_args(("dashboard", "serve"))
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert main(("dashboard", "inspect", "--run-id", result.run_id, "--data-root", str(root), "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "BACKTEST"
    assert payload["overview"]["integrity"] == "VERIFIED"
    assert main(("monitoring", "health", "--data-root", str(root), "--json")) == 0
    assert json.loads(capsys.readouterr().out)["monitoring_store"] == "HEALTHY"
    assert main(("dashboard", "serve", "--host", "0.0.0.0", "--data-root", str(root))) == 2
    assert "local-only" in capsys.readouterr().err


def test_dashboard_http_surface_contains_no_mutating_routes(dashboard_client) -> None:
    for route in dashboard_client.app.routes:
        methods = getattr(route, "methods", None)
        if methods:
            assert methods <= {"GET", "HEAD"}
