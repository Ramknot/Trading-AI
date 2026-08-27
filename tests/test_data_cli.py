import json

from trading_ai.cli import main
from trading_ai.data.engine import DataEngine
from trading_ai.data.storage import ParquetDataStore


def test_data_cli_json_outputs(
    monkeypatch,
    tmp_path,
    fake_data_provider,
    capsys,
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))
    monkeypatch.setattr("trading_ai.cli._build_data_engine", lambda _: engine)
    arguments = [
        "--symbol",
        "AAPL",
        "--timeframe",
        "1d",
        "--start",
        "2024-07-01",
        "--end",
        "2024-07-03",
        "--json",
    ]

    assert main(["data", "fetch", *arguments]) == 0
    fetched = json.loads(capsys.readouterr().out)
    assert fetched[0]["symbol"] == "AAPL"
    assert fetched[0]["row_count"] == 2
    assert fetched[0]["quality_status"] == "PASS"

    assert main(["data", "validate", "--symbol", "AAPL", "--timeframe", "1d", "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["integrity_valid"] is True
    assert validated["quality_report"]["quality_status"] == "PASS"

    assert main(["data", "inspect", "--symbol", "AAPL", "--timeframe", "1d", "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["manifest"]["provider"] == "fake"


def test_data_cli_cache_miss_is_a_controlled_error(
    monkeypatch,
    tmp_path,
    fake_data_provider,
    capsys,
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))
    monkeypatch.setattr("trading_ai.cli._build_data_engine", lambda _: engine)

    exit_code = main(
        [
            "data",
            "fetch",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--start",
            "2024-07-01",
            "--end",
            "2024-07-03",
            "--cache-mode",
            "CACHE_ONLY",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "ERROR"
    assert "no exact cached dataset" in payload["error"]
