from __future__ import annotations

import json
from decimal import Decimal

from fastapi.testclient import TestClient

from backtest_support import bar, dataset
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.models import BacktestConfig
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.backtesting.strategy import BuyAndHoldDemoStrategy
from trading_ai.cli import main
from trading_ai.core.config import load_profile, load_runtime_settings
from trading_ai.costs.economics import EconomicGate
from trading_ai.costs.engine import BalancedTransactionCostEngine
from trading_ai.monitoring.dashboard import create_dashboard_app
from trading_ai.risk.balanced import BalancedRiskEngine


def _export_cost_aware_run(data_root):
    profile = load_profile("balanced")
    costs = BalancedTransactionCostEngine.from_profile(profile)
    result = BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(profile),
        cost_engine=costs,
        economic_gate=EconomicGate(costs.bundle.config, costs.config_hash),
        code_version="lot-8.1-cli-test",
    ).run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("1")),
        (
            dataset(
                (
                    bar(0, opening="100", close="100"),
                    bar(1, opening="101", high="103", low="100", close="102"),
                )
            ),
        ),
        load_runtime_settings("PAPER", "balanced").context,
        BacktestConfig(
            starting_cash=Decimal("10000"),
            benchmark_symbol="AAPL",
            spread_bps=Decimal("5"),
            slippage_bps=Decimal("2"),
        ),
    )
    BacktestResultStore(data_root / "backtests").export(result)
    return result


def test_costs_cli_inspects_and_verifies_dated_source_provenance(capsys) -> None:
    assert main(("costs", "inspect", "--profile", "balanced", "--json")) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["engine"] == "balanced-transaction-cost"
    assert inspected["version"] == "1.0"
    assert inspected["tariff_status"] == "VERIFIED"
    assert inspected["tariff_source"].startswith("https://www.interactivebrokers.com/")
    assert len(inspected["config_hash"]) == 64

    assert main(
        (
            "costs",
            "verify-config",
            "--cost-profile",
            "ibkr_pro_tiered",
            "--json",
        )
    ) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["tariff_profile"] == "ibkr_pro_tiered"
    assert verified["tariff_status"] == "VERIFIED"


def test_costs_cli_estimate_is_point_in_time_and_reports_cash_requirement(capsys) -> None:
    assert main(
        (
            "costs",
            "estimate",
            "--symbol",
            "AAPL",
            "--side",
            "BUY",
            "--quantity",
            "10",
            "--price",
            "100",
            "--timeframe",
            "1d",
            "--timestamp",
            "2026-08-29T16:00:00Z",
            "--spread-bps",
            "5",
            "--slippage-bps",
            "2",
            "--json",
        )
    ) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["reference_price"] == "100"
    assert estimate["entry_costs"]["commission"]["amount"] == "1.00000000"
    assert estimate["cash_requirement"]["total_cash_required"] == "1002.20000000"
    assert estimate["round_trip_costs"]["total_variable_cost"]["amount"] == "3.40000000"


def test_costs_cli_keeps_aggressive_profile_locked(capsys) -> None:
    assert main(("costs", "inspect", "--profile", "aggressive", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["enabled"] is False
    assert payload["locked"] is True


def test_validation_cli_records_blocked_external_campaign_and_inspects_it(
    tmp_path, capsys
) -> None:
    data_root = tmp_path / "data_local"
    result = _export_cost_aware_run(data_root)

    exit_code = main(
        (
            "validation",
            "run",
            "--run-id",
            result.run_id,
            "--final-oos-confirmed",
            "--no-training-edge-overlap-confirmed",
            "--synthetic-mechanics-only",
            "--data-root",
            str(data_root),
            "--json",
        )
    )
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert report["status"] == "BLOCKED_EXTERNAL_DATA"
    assert report["real_data_campaign_status"] == "BLOCKED_EXTERNAL_DATA"
    assert report["synthetic_mechanics_only"] is True
    assert report["unlocks_paper_or_live"] is False

    assert main(
        (
            "validation",
            "inspect",
            "--validation-id",
            report["validation_id"],
            "--data-root",
            str(data_root),
            "--json",
        )
    ) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["validation_id"] == report["validation_id"]
    assert inspected["status"] == "BLOCKED_EXTERNAL_DATA"

    dashboard = TestClient(create_dashboard_app(data_root=data_root))
    overview = dashboard.get(
        "/api/v1/overview", params={"run_id": result.run_id}
    ).json()
    costs = dashboard.get(
        "/api/v1/costs", params={"run_id": result.run_id}
    ).json()
    validation = dashboard.get(
        "/api/v1/validation", params={"run_id": result.run_id}
    ).json()
    assert overview["net_completeness"] == "NET INCOMPLETE"
    assert costs["trading"]["commission"]["status"] == "ESTIMATED"
    assert costs["trading"]["spread"]["status"] == "KNOWN"
    assert costs["trading"]["slippage"]["status"] == "KNOWN"
    assert costs["operating"]["server_vps"]["status"] == "UNAVAILABLE"
    assert costs["net_pnl_estimated"] is None
    assert validation["status"] == "BLOCKED_EXTERNAL_DATA"
    snapshot = dashboard.get(
        "/api/v1/snapshot", params={"run_id": result.run_id}
    ).json()
    trace = snapshot["decision_traces"][0]
    cost_stages = {
        item["stage"]: item["status"]
        for item in trace["steps"]
        if item["stage"] in {
            "Cost Estimate",
            "Economic Gate",
            "Actual Cost",
            "Cost Reconciliation",
        }
    }
    assert set(cost_stages.values()) == {"HEALTHY"}


def test_validation_cli_rejects_invalid_identifiers_without_path_escape(
    tmp_path, capsys
) -> None:
    exit_code = main(
        (
            "validation",
            "inspect",
            "--validation-id",
            "../outside",
            "--data-root",
            str(tmp_path),
            "--json",
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "ERROR"
    assert "invalid validation_id" in payload["error"]


def test_validation_cli_cannot_relabel_an_in_memory_fixture_as_real_data(
    tmp_path, capsys
) -> None:
    data_root = tmp_path / "data_local"
    result = _export_cost_aware_run(data_root)

    exit_code = main(
        (
            "validation",
            "run",
            "--run-id",
            result.run_id,
            "--final-oos-confirmed",
            "--no-training-edge-overlap-confirmed",
            "--real-data",
            "--data-root",
            str(data_root),
            "--json",
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 3
    assert payload["real_data_campaign_status"] == "BLOCKED_EXTERNAL_DATA"
