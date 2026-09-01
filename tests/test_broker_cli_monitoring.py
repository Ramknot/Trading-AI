from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from trading_ai.brokers.models import PaperMode, PaperSessionManifest
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.cli import build_parser, main
from trading_ai.monitoring.dashboard import create_dashboard_app


def _paper_store(root: Path) -> LocalPaperStore:
    store = LocalPaperStore(root / "paper")
    store.create_session(
        PaperSessionManifest(
            session_id="paper-monitoring",
            created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
            code_sha="2b192ad38acf1a0c3a08f0173417597724c16ebd",
            mode=PaperMode.PAPER_READ_ONLY,
            broker_adapter_name="ibkr-tws-paper",
            broker_adapter_version="1.0",
            official_sdk_version="10.50",
            server_version="190",
            account_hash="a" * 64,
            account_masked="IBKR-****0001",
            config_hashes=(("broker", "b" * 64),),
            ml_model_ids=(),
        )
    )
    store.append(
        "paper-monitoring",
        "orders",
        {"internal_order_id": "order-1", "state": "ACKNOWLEDGED"},
        record_id="order-1",
    )
    store.append(
        "paper-monitoring",
        "executions",
        {"exec_id": "exec-1", "quantity": "1"},
        record_id="exec-1",
    )
    store.append(
        "paper-monitoring",
        "reconciliation",
        {"status": "IN_SYNC", "reconciliation_id": "recon-1"},
        record_id="recon-1",
    )
    return store


def test_broker_and_paper_cli_are_read_only(capsys, tmp_path) -> None:
    assert main(["broker", "inspect-config", "--json"]) == 0
    payload = capsys.readouterr().out
    assert '"paper_execution_armed": false' in payload
    assert '"connectable": false' in payload
    assert main(["broker", "contracts", "--json"]) == 0
    assert '"first_match_fallback": false' in capsys.readouterr().out
    assert main(["paper", "list", "--data-root", str(tmp_path), "--json"]) == 0
    assert capsys.readouterr().out.strip() == "[]"

    parser = build_parser()
    help_text = parser.format_help().lower()
    assert "broker" in help_text and "paper" in help_text
    paper = next(
        action for action in parser._actions if action.dest == "command"
    ).choices["paper"]
    subcommands = next(
        action for action in paper._actions if action.dest == "paper_command"
    ).choices
    assert set(subcommands) == {
        "connectivity-check", "list", "inspect", "replay", "shadow-audit"
    }
    assert not {"buy", "sell", "submit", "cancel", "arm", "live"} & set(subcommands)


def test_paper_evidence_cli_commands_read_verified_local_bundle(
    capsys, tmp_path
) -> None:
    _paper_store(tmp_path)
    common = ["--session-id", "paper-monitoring", "--data-root", str(tmp_path), "--json"]
    assert main(["paper", "inspect", *common]) == 0
    inspected = capsys.readouterr().out
    assert '"integrity": "VERIFIED"' in inspected
    assert "IBKR-****0001" in inspected

    assert main(["paper", "replay", *common]) == 0
    replay = capsys.readouterr().out
    assert '"broker_fills_reproduced": false' in replay
    assert '"execution_count": 1' in replay

    assert main(["paper", "shadow-audit", *common]) == 0
    audit = capsys.readouterr().out
    assert '"status": "UNAVAILABLE"' in audit
    assert '"decision_envelopes": 0' in audit


def test_connectivity_command_fails_closed_before_network_with_example_config(
    capsys, tmp_path
) -> None:
    exit_code = main(
        [
            "paper",
            "connectivity-check",
            "--config",
            "config/brokers/ibkr_paper.example.toml",
            "--session-id",
            "paper-connectivity-example",
            "--data-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert exit_code == 2
    payload = capsys.readouterr().out
    assert '"status": "ERROR"' in payload
    assert "create an ignored local config" in payload
    assert not (tmp_path / "paper" / "paper-connectivity-example").exists()


def test_dashboard_exposes_read_only_broker_evidence_endpoints(tmp_path) -> None:
    _paper_store(tmp_path)
    client = TestClient(create_dashboard_app(data_root=tmp_path))
    sessions = client.get("/api/v1/broker/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["paper_execution_armed"] is False
    assert sessions.json()["live_hard_locked"] is True
    assert sessions.json()["sessions"][0]["account_masked"] == "IBKR-****0001"
    for route in ("session", "orders", "executions", "reconciliation", "paper-audit"):
        response = client.get(
            f"/api/v1/broker/{route}", params={"session_id": "paper-monitoring"}
        )
        assert response.status_code == 200, (route, response.text)
    audit = client.get(
        "/api/v1/broker/paper-audit", params={"session_id": "paper-monitoring"}
    ).json()
    assert audit["read_only"] is True
    assert audit["paper_execution_armed"] is False
    assert audit["replay"]["broker_fills_reproduced"] is False

    html = client.get("/").text
    assert "Broker &amp; Paper Infrastructure" in html
    assert "READ ONLY · LOCAL" in html
    assert "<button" not in html.lower()
    assert "submit-order" not in html.lower()
    assert "paper-execution-arm" not in html.lower()


def test_dashboard_refuses_paper_path_traversal(tmp_path) -> None:
    client = TestClient(create_dashboard_app(data_root=tmp_path))
    for value in ("../outside", "..\\outside", str(tmp_path.resolve())):
        response = client.get("/api/v1/broker/session", params={"session_id": value})
        assert response.status_code == 409


def test_broker_modules_have_no_provider_broker_wrapper_or_heavy_ml_imports() -> None:
    forbidden = {
        "yfinance", "requests", "ib_insync", "tensorflow", "torch",
        "trading_ai.data.providers", "trading_ai.strategies", "trading_ai.ml",
    }
    for path in Path("src/trading_ai/brokers").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name == blocked or name.startswith(f"{blocked}.")
            for name in imported
            for blocked in forbidden
        ), f"{path} imports forbidden dependency"


def test_repository_contains_no_live_or_tls_unlock_in_broker_code() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/trading_ai/brokers").rglob("*.py")
    )
    assert "verify=False" not in combined
    assert "force_live" not in combined
    assert "paper_execution_armed = True" not in combined
    assert "partial_fit" not in combined
    assert "place_order(" in combined  # official transport exists, behind the boundary
    assert "PaperExecutionBoundary" in combined
