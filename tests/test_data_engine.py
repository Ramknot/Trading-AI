import pytest

from trading_ai.core.exceptions import ProfileDisabledError
from trading_ai.data.engine import DataEngine
from trading_ai.data.exceptions import (
    CacheMissError,
    DataProviderTemporaryError,
    DataValidationError,
)
from trading_ai.data.models import (
    CacheMode,
    DataKind,
    InstrumentMetadata,
    ProviderBar,
    QualityStatus,
)
from trading_ai.data.providers import FakeDataProvider
from trading_ai.data.storage import ParquetDataStore


def test_fake_provider_returns_bars_metadata_and_actions(
    fake_data_provider, market_start, market_end
) -> None:
    from trading_ai.data.models import MarketDataRequest

    response = fake_data_provider.fetch_bars(
        MarketDataRequest("AAPL", "1d", market_start, market_end)
    )
    actions = fake_data_provider.fetch_corporate_actions(
        "AAPL", market_start, market_end
    )

    assert response.metadata.symbol == "AAPL"
    assert len(response.bars) == 2
    assert len(actions) == 2
    assert fake_data_provider.calls["fetch_bars"] == 1


def test_cache_first_fetch_store_reuse(
    tmp_path, fake_data_provider, market_start, market_end
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))

    first = engine.fetch(
        symbol="AAPL", timeframe="1d", start=market_start, end=market_end
    )
    second = engine.fetch(
        symbol="AAPL", timeframe="1d", start=market_start, end=market_end
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.bars == second.bars
    assert second.corporate_actions == first.corporate_actions
    assert fake_data_provider.calls["fetch_bars"] == 1
    assert fake_data_provider.calls["fetch_corporate_actions"] == 1


def test_refresh_forces_provider_fetch(
    tmp_path, fake_data_provider, market_start, market_end
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))
    engine.fetch(symbol="AAPL", timeframe="1d", start=market_start, end=market_end)

    refreshed = engine.fetch(
        symbol="AAPL",
        timeframe="1d",
        start=market_start,
        end=market_end,
        cache_mode=CacheMode.REFRESH,
    )

    assert refreshed.cache_hit is False
    assert fake_data_provider.calls["fetch_bars"] == 2


def test_cache_only_never_calls_provider_on_miss(
    tmp_path, fake_data_provider, market_start, market_end
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))

    with pytest.raises(CacheMissError):
        engine.fetch(
            symbol="AAPL",
            timeframe="1d",
            start=market_start,
            end=market_end,
            cache_mode=CacheMode.CACHE_ONLY,
        )

    assert fake_data_provider.calls["fetch_bars"] == 0


def test_string_cache_mode_is_coerced_without_network_on_cache_only(
    tmp_path, fake_data_provider, market_start, market_end
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))

    with pytest.raises(CacheMissError):
        engine.fetch(
            symbol="AAPL",
            timeframe="1d",
            start=market_start,
            end=market_end,
            cache_mode="CACHE_ONLY",  # type: ignore[arg-type]
        )

    assert fake_data_provider.calls["fetch_bars"] == 0


def test_4h_is_derived_from_cached_1h_without_network(
    tmp_path, fake_data_provider, market_start, market_end
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))
    source = engine.fetch(
        symbol="AAPL", timeframe="1h", start=market_start, end=market_end
    )

    result = engine.fetch(
        symbol="AAPL",
        timeframe="4h",
        start=market_start,
        end=market_end,
        cache_mode=CacheMode.CACHE_ONLY,
    )

    assert len(result.bars) == 4
    assert result.manifest.provider == "derived"
    assert result.manifest.data_kind is DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE
    assert result.manifest.derived_from == (source.manifest.dataset_id,)
    assert fake_data_provider.calls["fetch_bars"] == 1


def test_inspect_and_validate_cached_dataset(
    tmp_path, fake_data_provider, market_start, market_end
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))
    engine.fetch(symbol="AAPL", timeframe="1d", start=market_start, end=market_end)

    inspected = engine.inspect_cached("AAPL", "1d")
    validated = engine.validate_cached("AAPL", "1d")

    assert inspected.integrity_valid is True
    assert inspected.quality_report.quality_status is QualityStatus.PASS
    assert validated == inspected


def test_provider_partial_period_is_recorded_and_warned(
    tmp_path,
    aapl_metadata,
    aapl_hourly_rows,
    market_start,
    market_end,
) -> None:
    partial = aapl_hourly_rows[:-1]
    provider = FakeDataProvider(
        datasets={("AAPL", "1h"): partial},
        metadata_by_symbol={"AAPL": aapl_metadata},
    )
    engine = DataEngine(provider, ParquetDataStore(tmp_path))

    result = engine.fetch(
        symbol="AAPL", timeframe="1h", start=market_start, end=market_end
    )

    assert result.manifest.requested_end == market_end
    assert result.manifest.actual_end == partial[-1].timestamp
    assert result.quality_report.missing_expected_bar_count == 1
    assert result.quality_report.quality_status is QualityStatus.WARNING
    assert "missing" in " ".join(result.manifest.warnings)


def test_derived_quality_propagates_source_gap_warnings(
    tmp_path,
    aapl_metadata,
    aapl_hourly_rows,
    market_start,
    market_end,
) -> None:
    provider = FakeDataProvider(
        datasets={("AAPL", "1h"): aapl_hourly_rows[:3] + aapl_hourly_rows[4:]},
        metadata_by_symbol={"AAPL": aapl_metadata},
    )
    engine = DataEngine(provider, ParquetDataStore(tmp_path))

    result = engine.fetch(
        symbol="AAPL", timeframe="4h", start=market_start, end=market_end
    )

    assert result.quality_report.quality_status is QualityStatus.WARNING
    assert "source 1h" in " ".join(result.quality_report.warnings)


def test_profile_controls_symbol_and_timeframe(
    tmp_path, fake_data_provider, market_start, market_end
) -> None:
    engine = DataEngine(fake_data_provider, ParquetDataStore(tmp_path))

    with pytest.raises(DataValidationError, match="not in profile"):
        engine.fetch(
            symbol="BTC-USD", timeframe="1d", start=market_start, end=market_end
        )
    with pytest.raises(DataValidationError, match="not in profile"):
        engine.fetch(
            symbol="AAPL", timeframe="15m", start=market_start, end=market_end
        )
    with pytest.raises(ProfileDisabledError):
        engine.fetch(
            profile_name="aggressive",
            symbol="AAPL",
            timeframe="1h",
            start=market_start,
            end=market_end,
        )


def test_multi_asset_load_is_globally_sorted_and_profile_driven(
    tmp_path,
    aapl_metadata,
    aapl_daily_rows,
    market_start,
    market_end,
) -> None:
    msft_metadata = InstrumentMetadata(
        symbol="MSFT",
        exchange="NMS",
        exchange_timezone="America/New_York",
        calendar="NYSE",
        source="fake",
        currency="USD",
    )
    msft_rows = tuple(
        ProviderBar(
            symbol="MSFT",
            timeframe=row.timeframe,
            timestamp=row.timestamp,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            adjusted_close=row.adjusted_close,
            source=row.source,
        )
        for row in aapl_daily_rows
    )
    provider = FakeDataProvider(
        datasets={
            ("AAPL", "1d"): aapl_daily_rows,
            ("MSFT", "1d"): msft_rows,
        },
        metadata_by_symbol={"AAPL": aapl_metadata, "MSFT": msft_metadata},
    )
    engine = DataEngine(provider, ParquetDataStore(tmp_path))

    bars = engine.load_bars(("MSFT", "AAPL"), "1d", market_start, market_end)

    assert [bar.symbol for bar in bars] == ["AAPL", "AAPL", "MSFT", "MSFT"]


def test_temporary_provider_failures_have_bounded_retries(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
) -> None:
    class FlakyProvider(FakeDataProvider):
        def __init__(self) -> None:
            super().__init__(
                datasets=fake_data_provider.datasets,
                metadata_by_symbol=fake_data_provider.metadata_by_symbol,
                actions_by_symbol=fake_data_provider.actions_by_symbol,
            )
            self.attempts = 0

        def fetch_bars(self, request):
            self.attempts += 1
            if self.attempts < 3:
                raise DataProviderTemporaryError("temporary test failure")
            return super().fetch_bars(request)

    provider = FlakyProvider()
    engine = DataEngine(
        provider,
        ParquetDataStore(tmp_path),
        max_retries=2,
        retry_delay_seconds=0,
    )

    result = engine.fetch(
        symbol="AAPL", timeframe="1d", start=market_start, end=market_end
    )

    assert result.manifest.row_count == 2
    assert provider.attempts == 3


def test_retry_limit_is_enforced(tmp_path, fake_data_provider, market_start, market_end) -> None:
    class FailingProvider(FakeDataProvider):
        def fetch_bars(self, request):
            self.calls["attempt"] += 1
            raise DataProviderTemporaryError("still unavailable")

    provider = FailingProvider(
        datasets=fake_data_provider.datasets,
        metadata_by_symbol=fake_data_provider.metadata_by_symbol,
    )
    engine = DataEngine(
        provider,
        ParquetDataStore(tmp_path),
        max_retries=1,
        retry_delay_seconds=0,
    )

    with pytest.raises(DataProviderTemporaryError):
        engine.fetch(
            symbol="AAPL", timeframe="1d", start=market_start, end=market_end
        )
    assert provider.calls["attempt"] == 2
