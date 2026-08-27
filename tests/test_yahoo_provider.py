import os
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from trading_ai.data.engine import DataEngine
from trading_ai.data.exceptions import DataProviderError, DataProviderTemporaryError
from trading_ai.data.models import CorporateActionType, MarketDataRequest
from trading_ai.data.providers import YahooFinanceProvider
from trading_ai.data.storage import ParquetDataStore


class _DummyTicker:
    def __init__(self, symbol: str, *, exchange: str = "NMS") -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.actions = pd.DataFrame(
            {
                "Dividends": [0.25, 0.0],
                "Stock Splits": [0.0, 4.0],
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2024, 7, 1, 9, 30, tzinfo=timezone.utc),
                    datetime(2024, 7, 2, 9, 30, tzinfo=timezone.utc),
                ]
            ),
        )

    def get_history_metadata(self):
        if self.exchange == "PAR":
            return {
                "exchangeName": "PAR",
                "exchangeTimezoneName": "Europe/Paris",
                "currency": "EUR",
            }
        return {
            "exchangeName": self.exchange,
            "exchangeTimezoneName": "America/New_York",
            "currency": "USD",
        }


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [1000, 1100],
        },
        index=pd.DatetimeIndex(
            [
                datetime(2024, 7, 1, 4, tzinfo=timezone.utc),
                datetime(2024, 7, 2, 4, tzinfo=timezone.utc),
            ]
        ),
    )


def test_yahoo_adapter_converts_to_provider_neutral_rows(monkeypatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: _DummyTicker(symbol))
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: _daily_frame())
    provider = YahooFinanceProvider()

    result = provider.fetch_bars(
        MarketDataRequest(
            "AAPL",
            "1d",
            datetime(2024, 7, 1, tzinfo=timezone.utc),
            datetime(2024, 7, 3, tzinfo=timezone.utc),
        )
    )

    assert result.metadata.calendar == "NYSE"
    assert result.metadata.exchange_timezone == "America/New_York"
    assert result.bars[0].open == Decimal("100.0")
    assert result.bars[0].close == Decimal("101.0")
    assert result.bars[0].adjusted_close == Decimal("100.5")
    assert result.bars[0].source == "yahoo"
    assert not hasattr(result.bars[0], "to_pandas")


def test_yahoo_adapter_maps_paris_calendar(monkeypatch) -> None:
    import yfinance

    monkeypatch.setattr(
        yfinance, "Ticker", lambda symbol: _DummyTicker(symbol, exchange="PAR")
    )

    metadata = YahooFinanceProvider().metadata("MC.PA")

    assert metadata.calendar == "XPAR"
    assert metadata.exchange_timezone == "Europe/Paris"


def test_yahoo_adapter_rejects_unknown_exchange_calendar(monkeypatch) -> None:
    import yfinance

    monkeypatch.setattr(
        yfinance, "Ticker", lambda symbol: _DummyTicker(symbol, exchange="UNKNOWN")
    )

    with pytest.raises(DataProviderError, match="no configured market calendar"):
        YahooFinanceProvider().metadata("UNKNOWN")


def test_yahoo_adapter_keeps_dividends_and_splits_separate(monkeypatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: _DummyTicker(symbol))
    provider = YahooFinanceProvider()

    actions = provider.fetch_corporate_actions(
        "AAPL",
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        datetime(2024, 7, 3, tzinfo=timezone.utc),
    )

    assert [action.action_type for action in actions] == [
        CorporateActionType.DIVIDEND,
        CorporateActionType.SPLIT,
    ]
    assert [action.value for action in actions] == [Decimal("0.25"), Decimal("4.0")]


def test_yahoo_specific_failure_is_wrapped(monkeypatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: _DummyTicker(symbol))
    monkeypatch.setattr(
        yfinance,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider detail")),
    )

    with pytest.raises(DataProviderError, match="Yahoo bars failure") as raised:
        YahooFinanceProvider().fetch_bars(
            MarketDataRequest(
                "AAPL",
                "1d",
                datetime(2024, 7, 1, tzinfo=timezone.utc),
                datetime(2024, 7, 3, tzinfo=timezone.utc),
            )
        )

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_yahoo_timeout_is_classified_as_temporary(monkeypatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: _DummyTicker(symbol))
    monkeypatch.setattr(
        yfinance,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    with pytest.raises(DataProviderTemporaryError):
        YahooFinanceProvider().fetch_bars(
            MarketDataRequest(
                "AAPL",
                "1d",
                datetime(2024, 7, 1, tzinfo=timezone.utc),
                datetime(2024, 7, 3, tzinfo=timezone.utc),
            )
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("TRADING_AI_RUN_NETWORK_TESTS") != "1",
    reason="set TRADING_AI_RUN_NETWORK_TESTS=1 for the opt-in Yahoo smoke test",
)
def test_real_yahoo_daily_smoke(tmp_path) -> None:
    engine = DataEngine(
        YahooFinanceProvider(),
        ParquetDataStore(tmp_path / "data_local"),
    )

    result = engine.fetch(
        symbol="AAPL",
        timeframe="1d",
        start=datetime(2025, 1, 2, tzinfo=timezone.utc),
        end=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert result.bars
    assert result.manifest.provider == "yahoo"
    assert result.manifest.actual_start is not None
    assert result.manifest.checksum_sha256
    assert engine.store.verify_integrity(result.manifest) is True
