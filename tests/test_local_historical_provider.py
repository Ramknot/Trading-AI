from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_ai.data.exceptions import DataValidationError
from trading_ai.data.models import InstrumentMetadata, MarketDataRequest
from trading_ai.data.providers.local import LocalHistoricalFileProvider


START = datetime(2024, 1, 2, 21, tzinfo=timezone.utc)


def _metadata() -> InstrumentMetadata:
    return InstrumentMetadata(
        symbol="AAPL",
        exchange="NMS",
        exchange_timezone="America/New_York",
        calendar="NYSE",
        source="explicit-local-fixture",
        currency="USD",
    )


def test_local_csv_provider_is_explicit_offline_and_provider_neutral(tmp_path) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    pd.DataFrame(
        {
            "timestamp": [START.isoformat(), (START + timedelta(days=1)).isoformat()],
            "open": [100, 101], "high": [102, 103], "low": [99, 100],
            "close": [101, 102], "volume": [1000, 1100],
            "adjusted_close": [100.5, 101.5],
        }
    ).to_csv(root / "aapl.csv", index=False)
    provider = LocalHistoricalFileProvider(
        root=root,
        datasets={("AAPL", "1d"): "aapl.csv"},
        metadata_by_symbol={"AAPL": _metadata()},
    )
    result = provider.fetch_bars(
        MarketDataRequest("AAPL", "1d", START, START + timedelta(days=2))
    )
    assert provider.name == "local_historical_file"
    assert len(result.bars) == 2
    assert result.bars[0].timestamp.utcoffset() == timedelta(0)
    assert result.bars[0].adjusted_close is not None
    assert result.warnings and "provenance" in result.warnings[0].lower()


def test_local_provider_rejects_naive_timestamps_and_path_escape(tmp_path) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2024-01-02T21:00:00"],
            "open": [100], "high": [102], "low": [99], "close": [101],
            "volume": [1000],
        }
    ).to_parquet(root / "naive.parquet", index=False)
    provider = LocalHistoricalFileProvider(
        root=root,
        datasets={("AAPL", "1d"): "naive.parquet"},
        metadata_by_symbol={"AAPL": _metadata()},
    )
    with pytest.raises(DataValidationError, match="timezone-aware"):
        provider.fetch_bars(
            MarketDataRequest("AAPL", "1d", START, START + timedelta(days=2))
        )
    with pytest.raises(DataValidationError, match="escapes"):
        LocalHistoricalFileProvider(
            root=root,
            datasets={("AAPL", "1d"): "../outside.csv"},
            metadata_by_symbol={"AAPL": _metadata()},
        )
