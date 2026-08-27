import json
from datetime import datetime, timezone

import pytest

from trading_ai.data.exceptions import DataIntegrityError, DataStorageError
from trading_ai.data.models import DataKind, MarketDataRequest
from trading_ai.data.quality import normalize_provider_bars
from trading_ai.data.storage import ParquetDataStore, sha256_file


def test_parquet_round_trip_and_manifest(
    tmp_path,
    aapl_metadata,
    aapl_daily_rows,
    market_start,
    market_end,
) -> None:
    store = ParquetDataStore(tmp_path / "data_local")
    bars, _ = normalize_provider_bars(
        aapl_daily_rows,
        aapl_metadata,
        market_start,
        market_end,
        expected_timeframe="1d",
    )
    request = MarketDataRequest("AAPL", "1d", market_start, market_end)

    manifest = store.save_bars(
        bars=bars,
        request=request,
        metadata=aapl_metadata,
        provider="fake",
        provider_version="1",
        data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
    )
    loaded = store.read_bars(manifest)

    assert loaded == bars
    assert manifest.row_count == 2
    assert manifest.requested_start == market_start
    assert manifest.requested_end == market_end
    assert manifest.actual_start == bars[0].timestamp
    assert manifest.actual_end == bars[-1].timestamp
    assert manifest.timezone == "UTC"
    assert manifest.source_timezone == "America/New_York"
    assert manifest.exchange == "NMS"
    assert manifest.calendar == "NYSE"
    assert manifest.data_kind is DataKind.RAW_WITH_ADJUSTED_CLOSE
    assert manifest.file_path.startswith("market/fake/AAPL/1d/")
    assert len(manifest.checksum_sha256) == 64
    assert store.verify_integrity(manifest) is True


def test_manifest_json_round_trip(
    tmp_path,
    aapl_metadata,
    aapl_daily_rows,
    market_start,
    market_end,
) -> None:
    store = ParquetDataStore(tmp_path)
    bars, _ = normalize_provider_bars(
        aapl_daily_rows,
        aapl_metadata,
        market_start,
        market_end,
        expected_timeframe="1d",
    )
    manifest = store.save_bars(
        bars=bars,
        request=MarketDataRequest("AAPL", "1d", market_start, market_end),
        metadata=aapl_metadata,
        provider="fake",
        provider_version="1",
        data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
    )

    loaded_manifest = store.read_manifest(manifest.dataset_id)
    payload = json.loads(
        (store.manifest_directory / f"{manifest.dataset_id}.json").read_text(
            encoding="utf-8"
        )
    )

    assert loaded_manifest == manifest
    assert payload["schema_version"] == "1.0"
    assert payload["checksum_sha256"] == manifest.checksum_sha256


def test_corporate_actions_parquet_round_trip(
    tmp_path,
    aapl_metadata,
    aapl_actions,
    market_start,
    market_end,
) -> None:
    store = ParquetDataStore(tmp_path)

    manifest = store.save_corporate_actions(
        actions=reversed(aapl_actions),
        symbol="AAPL",
        start=market_start,
        end=market_end,
        metadata=aapl_metadata,
        provider="fake",
        provider_version="1",
    )
    loaded = store.read_corporate_actions(manifest)

    assert loaded == aapl_actions
    assert manifest.data_kind is DataKind.CORPORATE_ACTIONS
    assert manifest.file_path.startswith("corporate_actions/fake/AAPL/")


def test_sha256_detects_file_tampering(
    tmp_path,
    aapl_metadata,
    aapl_daily_rows,
    market_start,
    market_end,
) -> None:
    store = ParquetDataStore(tmp_path)
    bars, _ = normalize_provider_bars(
        aapl_daily_rows,
        aapl_metadata,
        market_start,
        market_end,
        expected_timeframe="1d",
    )
    manifest = store.save_bars(
        bars=bars,
        request=MarketDataRequest("AAPL", "1d", market_start, market_end),
        metadata=aapl_metadata,
        provider="fake",
        provider_version="1",
        data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
    )
    path = store.root / manifest.file_path
    original_hash = sha256_file(path)
    path.write_bytes(path.read_bytes() + b"tampered")

    assert original_hash == manifest.checksum_sha256
    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        store.verify_integrity(manifest)


def test_derived_dataset_requires_and_records_lineage(
    tmp_path,
    aapl_metadata,
    aapl_hourly_rows,
    market_start,
    market_end,
) -> None:
    from trading_ai.data.resampling import resample_1h_to_4h

    store = ParquetDataStore(tmp_path)
    source, _ = normalize_provider_bars(
        aapl_hourly_rows,
        aapl_metadata,
        market_start,
        market_end,
        expected_timeframe="1h",
    )
    derived = resample_1h_to_4h(source, aapl_metadata)
    manifest = store.save_bars(
        bars=derived,
        request=MarketDataRequest("AAPL", "4h", market_start, market_end),
        metadata=aapl_metadata,
        provider="derived",
        provider_version=None,
        data_kind=DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE,
        derived_from=("source-dataset-id",),
    )

    assert manifest.derived_from == ("source-dataset-id",)
    assert manifest.file_path.startswith("derived/AAPL/4h/")


def test_cache_lookup_is_exact_and_latest(
    tmp_path,
    aapl_metadata,
    aapl_daily_rows,
    market_start,
    market_end,
) -> None:
    store = ParquetDataStore(tmp_path)
    bars, _ = normalize_provider_bars(
        aapl_daily_rows,
        aapl_metadata,
        market_start,
        market_end,
        expected_timeframe="1d",
    )
    manifest = store.save_bars(
        bars=bars,
        request=MarketDataRequest("AAPL", "1d", market_start, market_end),
        metadata=aapl_metadata,
        provider="fake",
        provider_version="1",
        data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
    )

    assert store.find_exact(
        provider="fake",
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=market_end,
        data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
    ) == manifest
    assert store.find_exact(
        provider="fake",
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=datetime(2024, 7, 4, tzinfo=timezone.utc),
        data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
    ) is None
    assert store.find_latest("AAPL", "1d") == manifest


def test_storage_refuses_unsorted_market_bars(
    tmp_path,
    aapl_metadata,
    aapl_daily_rows,
    market_start,
    market_end,
) -> None:
    store = ParquetDataStore(tmp_path)
    bars, _ = normalize_provider_bars(
        aapl_daily_rows,
        aapl_metadata,
        market_start,
        market_end,
        expected_timeframe="1d",
    )

    with pytest.raises(DataStorageError, match="normalized and sorted"):
        store.save_bars(
            bars=reversed(bars),
            request=MarketDataRequest("AAPL", "1d", market_start, market_end),
            metadata=aapl_metadata,
            provider="fake",
            provider_version="1",
            data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
        )
