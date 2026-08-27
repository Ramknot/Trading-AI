import hashlib
import json
from datetime import datetime
from decimal import Decimal

import pyarrow.parquet as parquet
import pytest

from backtest_support import bar, dataset
from risk_support import PermissiveBacktestEngine as BacktestEngine
from trading_ai.backtesting.exceptions import BacktestDataError, BacktestStorageError
from trading_ai.backtesting.input import load_cached_dataset
from trading_ai.backtesting.models import BacktestConfig
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.backtesting.strategy import BuyAndHoldDemoStrategy
from trading_ai.cli import main
from trading_ai.data.engine import DataEngine
from trading_ai.data.models import CacheMode
from trading_ai.data.storage import ParquetDataStore


def _result(paper_context):
    return BacktestEngine(code_version="test-code").run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("1")),
        (dataset((bar(0), bar(1, opening="105"))),),
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("1000"), benchmark_symbol="AAPL"
        ),
    )


def test_result_export_writes_json_parquet_and_verifiable_hashes(
    tmp_path, paper_context
) -> None:
    result = _result(paper_context)
    store = BacktestResultStore(tmp_path / "data_local" / "backtests")

    directory = store.export(result)
    summary = store.inspect(result.run_id)

    assert directory == tmp_path / "data_local" / "backtests" / result.run_id
    assert store.verify_integrity(result.run_id) is True
    assert summary["result_hash"] == result.result_hash
    assert summary["dataset_references"][0]["dataset_id"].startswith("memory-")
    assert parquet.read_table(directory / "equity.parquet").num_rows == 2
    assert parquet.read_table(directory / "fills.parquet").num_rows == 1
    assert parquet.read_table(directory / "trades.parquet").num_rows == 0
    assert parquet.read_table(directory / "orders.parquet").num_rows == 1
    assert parquet.read_table(directory / "signals.parquet").num_rows == 0
    assert parquet.read_table(directory / "ledger.parquet").num_rows == 1
    assert parquet.read_table(directory / "risk_decisions.parquet").num_rows == 1
    assert parquet.read_table(directory / "risk_states.parquet").num_rows == 0
    assert summary["schema_version"] == "1.2"
    assert summary["risk"]["engine_name"] == "permissive-test-risk"


def test_result_store_detects_tampering(tmp_path, paper_context) -> None:
    result = _result(paper_context)
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)
    summary_path = directory / "summary.json"
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(BacktestStorageError, match="SHA-256 mismatch"):
        store.verify_integrity(result.run_id)


def test_result_store_keeps_lot2_schema_1_0_exports_inspectable(
    tmp_path, paper_context
) -> None:
    result = _result(paper_context)
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)
    for name in (
        "signals.parquet",
        "risk_decisions.parquet",
        "risk_states.parquet",
    ):
        (directory / name).unlink()
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "1.0"
    summary["counts"].pop("signals")
    summary["counts"].pop("risk_decisions")
    summary["counts"].pop("risk_state_transitions")
    summary.pop("risk")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums_path = directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for name in (
        "signals.parquet",
        "risk_decisions.parquet",
        "risk_states.parquet",
    ):
        checksums["files"].pop(name)
    checksums["files"]["summary.json"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    checksums_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    inspected = store.inspect(result.run_id)

    assert inspected["schema_version"] == "1.0"
    assert "signals" not in inspected["counts"]


def test_result_store_keeps_lot3_schema_1_1_exports_inspectable(
    tmp_path, paper_context
) -> None:
    result = _result(paper_context)
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)
    for name in ("risk_decisions.parquet", "risk_states.parquet"):
        (directory / name).unlink()
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "1.1"
    summary["counts"].pop("risk_decisions")
    summary["counts"].pop("risk_state_transitions")
    summary.pop("risk")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums_path = directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for name in ("risk_decisions.parquet", "risk_states.parquet"):
        checksums["files"].pop(name)
    checksums["files"]["summary.json"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    checksums_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    inspected = store.inspect(result.run_id)
    assert inspected["schema_version"] == "1.1"
    assert "signals" in inspected["counts"]
    assert "risk" not in inspected


def test_cached_dataset_adapter_preserves_manifest_and_action_provenance(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
) -> None:
    parquet_store = ParquetDataStore(tmp_path / "data_local")
    fetched = DataEngine(fake_data_provider, parquet_store).fetch(
        profile_name="balanced",
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=market_end,
        cache_mode=CacheMode.REFRESH,
    )

    loaded = load_cached_dataset(
        parquet_store,
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=market_end,
    )

    assert loaded.bars == fetched.bars
    assert loaded.corporate_actions == fetched.corporate_actions
    assert loaded.reference.dataset_id == fetched.manifest.dataset_id
    assert loaded.reference.checksum_sha256 == fetched.manifest.checksum_sha256
    assert loaded.reference.corporate_actions_dataset_id == (
        fetched.corporate_actions_manifest.dataset_id
        if fetched.corporate_actions_manifest is not None
        else None
    )


def test_derived_cached_dataset_retains_source_lineage(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
) -> None:
    parquet_store = ParquetDataStore(tmp_path / "data_local")
    fetched = DataEngine(fake_data_provider, parquet_store).fetch(
        profile_name="balanced",
        symbol="AAPL",
        timeframe="4h",
        start=market_start,
        end=market_end,
        cache_mode=CacheMode.REFRESH,
    )

    loaded = load_cached_dataset(
        parquet_store,
        symbol="AAPL",
        timeframe="4h",
        start=market_start,
        end=market_end,
    )

    assert fetched.manifest.derived_from
    assert loaded.reference.derived_from == fetched.manifest.derived_from


def test_cached_dataset_adapter_never_fetches_on_miss(tmp_path) -> None:
    with pytest.raises(BacktestDataError, match="never download"):
        load_cached_dataset(
            ParquetDataStore(tmp_path),
            symbol="AAPL",
            timeframe="1d",
            start=bar(0).timestamp,
            end=bar(1).timestamp,
        )


def test_cached_dataset_adapter_refuses_naive_request_timestamps(tmp_path) -> None:
    with pytest.raises(BacktestDataError, match="timezone-aware"):
        load_cached_dataset(
            ParquetDataStore(tmp_path),
            symbol="AAPL",
            timeframe="1d",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )


def test_backtest_cli_runs_only_from_cache_and_inspects_result(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
    capsys,
) -> None:
    data_root = tmp_path / "data_local"
    DataEngine(fake_data_provider, ParquetDataStore(data_root)).fetch(
        profile_name="balanced",
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=market_end,
        cache_mode=CacheMode.REFRESH,
    )
    common = [
        "--symbol",
        "AAPL",
        "--timeframe",
        "1d",
        "--start",
        "2024-07-01",
        "--end",
        "2024-07-03",
        "--data-root",
        str(data_root),
        "--json",
    ]

    assert main(["backtest", "run", *common]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "COMPLETED"
    assert payload["initial_cash"] == "100000"
    assert payload["benchmark"]["symbol"] == "AAPL"
    assert payload["dataset_ids"]
    run_id = payload["run_id"]

    assert main(
        [
            "backtest",
            "inspect",
            "--run-id",
            run_id,
            "--data-root",
            str(data_root),
            "--json",
        ]
    ) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["run_id"] == run_id
    assert inspected["counts"]["fills"] == 1


def test_backtest_cli_cache_miss_is_controlled(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "backtest",
            "run",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-10",
            "--data-root",
            str(tmp_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "ERROR"
    assert "never download" in payload["error"]


def test_strategy_cli_lists_versioned_non_optimized_baselines(capsys) -> None:
    assert main(["strategy", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert {item["name"] for item in payload} >= {
        "trend",
        "momentum",
        "breakout",
        "buy-and-hold",
    }
    trend = next(item for item in payload if item["name"] == "trend")
    assert trend["version"] == "1.0"
    assert "not optimized" in trend["warning"]


def test_backtest_cli_runs_lot3_trend_from_cache_without_fetching(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
    capsys,
) -> None:
    data_root = tmp_path / "data_local"
    DataEngine(fake_data_provider, ParquetDataStore(data_root)).fetch(
        profile_name="balanced",
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=market_end,
        cache_mode=CacheMode.REFRESH,
    )

    assert main(
        [
            "backtest",
            "run",
            "--strategy",
            "trend",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--start",
            "2024-07-01",
            "--end",
            "2024-07-03",
            "--fast-window",
            "1",
            "--slow-window",
            "2",
            "--slope-lookback",
            "1",
            "--data-root",
            str(data_root),
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["strategy"] == "trend"
    assert payload["strategy_parameters"]["feature_schema_version"] == "1.0"
    assert payload["dataset_ids"]
