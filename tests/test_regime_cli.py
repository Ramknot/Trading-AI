from __future__ import annotations

import json

from trading_ai.cli import main
from trading_ai.data.engine import DataEngine
from trading_ai.data.models import CacheMode
from trading_ai.data.storage import ParquetDataStore


def _populate(data_root, fake_data_provider, market_start, market_end) -> None:
    DataEngine(fake_data_provider, ParquetDataStore(data_root)).fetch(
        profile_name="balanced",
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=market_end,
        cache_mode=CacheMode.REFRESH,
    )


def test_regime_inspect_uses_only_exact_cached_dataset(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
    capsys,
) -> None:
    data_root = tmp_path / "data_local"
    _populate(data_root, fake_data_provider, market_start, market_end)

    exit_code = main(
        [
            "regime",
            "inspect",
            "--profile",
            "balanced",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--start",
            market_start.date().isoformat(),
            "--end",
            market_end.date().isoformat(),
            "--data-root",
            str(data_root),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["detector_name"] == "balanced-regime"
    assert payload["detector_version"] == "1.0"
    assert payload["latest"]["symbol"] == "AAPL"
    assert payload["latest"]["structure_regime"] == "UNKNOWN"
    assert payload["network_access"] is False
    assert payload["dataset_id"]
    assert len(payload["dataset_checksum"]) == 64


def test_regime_latest_uses_latest_local_manifest_without_network(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
    capsys,
) -> None:
    data_root = tmp_path / "data_local"
    _populate(data_root, fake_data_provider, market_start, market_end)

    assert main(
        [
            "regime",
            "latest",
            "--profile",
            "balanced",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--data-root",
            str(data_root),
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network_access"] is False
    assert payload["latest"]["timestamp"]


def test_regime_inspect_cache_miss_fails_without_fetch(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "regime",
            "inspect",
            "--profile",
            "balanced",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
            "--data-root",
            str(tmp_path / "empty"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert "never download" in payload["error"]


def test_backtest_cli_accepts_mean_reversion_and_records_regime_policy(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
    capsys,
) -> None:
    data_root = tmp_path / "data_local"
    _populate(data_root, fake_data_provider, market_start, market_end)

    exit_code = main(
        [
            "backtest",
            "run",
            "--strategy",
            "mean-reversion",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--start",
            market_start.date().isoformat(),
            "--end",
            market_end.date().isoformat(),
            "--mean-reversion-lookback",
            "2",
            "--data-root",
            str(data_root),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["strategy"] == "mean-reversion"
    assert payload["regime"]["detector"] == "balanced-regime"
    assert payload["regime"]["policy"] == "balanced-strategy-policy"
    assert len(payload["regime"]["config_hash"]) == 64
