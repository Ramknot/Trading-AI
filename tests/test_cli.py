import json

from trading_ai.cli import main
from trading_ai.core.health import doctor


def test_doctor_balanced_paper_is_healthy() -> None:
    report = doctor("PAPER", "balanced")

    assert report.status == "OK"
    assert report.configuration_valid is True
    assert report.profile_enabled is True
    assert report.live_allowed is False


def test_cli_doctor_emits_json(capsys) -> None:
    exit_code = main(
        ["doctor", "--environment", "PAPER", "--profile", "balanced", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "OK"
    assert payload["environment"] == "PAPER"


def test_cli_blocks_aggressive(capsys) -> None:
    exit_code = main(
        ["doctor", "--environment", "PAPER", "--profile", "aggressive", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "BLOCKED"
    assert payload["profile_enabled"] is False


def test_cli_blocks_live(capsys) -> None:
    exit_code = main(
        ["doctor", "--environment", "LIVE", "--profile", "balanced", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "BLOCKED"
    assert payload["live_allowed"] is False


def test_cli_risk_inspect_reports_balanced_limits_and_hash(capsys) -> None:
    exit_code = main(["risk", "inspect", "--profile", "balanced", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["engine"] == "balanced-risk"
    assert payload["version"] == "1.0"
    assert payload["enabled"] is True
    assert payload["locked"] is False
    assert payload["limits"]["max_positions"] == 5
    assert payload["limits"]["max_portfolio_exposure"] == "0.6"
    assert len(payload["config_hash"]) == 64
    assert payload["volatility"]["feature"] == "rolling_vol_20"


def test_cli_risk_inspect_does_not_activate_aggressive(capsys) -> None:
    exit_code = main(["risk", "inspect", "--profile", "aggressive", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["profile"] == "aggressive"
    assert payload["enabled"] is False
    assert payload["locked"] is True
