from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_ai.core.models import MarketBar
from trading_ai.data.base import DataEngine as DataEngineContract
from trading_ai.data.base import DataProvider
from trading_ai.data.exceptions import DataError, DataProviderError, DataValidationError
from trading_ai.data.models import (
    CorporateActionType,
    Dividend,
    MarketDataRequest,
    ProviderBar,
    QualityStatus,
    StockSplit,
)
from trading_ai.data.providers import FakeDataProvider
from trading_ai.data.quality import normalize_provider_bars


def test_data_provider_is_an_abstract_boundary() -> None:
    with pytest.raises(TypeError):
        DataProvider()  # type: ignore[abstract]
    assert isinstance(FakeDataProvider(), DataProvider)


def test_data_engine_contract_remains_abstract() -> None:
    with pytest.raises(TypeError):
        DataEngineContract()  # type: ignore[abstract]


def test_market_request_rejects_naive_bounds() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketDataRequest(
            "AAPL",
            "1d",
            datetime(2024, 1, 1),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )


def test_normalization_converts_to_utc_and_preserves_adjusted_close(
    aapl_metadata,
) -> None:
    eastern = timezone(timedelta(hours=-4))
    row = ProviderBar(
        "AAPL",
        "1h",
        datetime(2024, 7, 1, 9, 30, tzinfo=eastern),
        100,
        102,
        99,
        101,
        10,
        100.5,
        "fake",
    )
    bars, report = normalize_provider_bars(
        (row,),
        aapl_metadata,
        datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
        datetime(2024, 7, 1, 14, 30, tzinfo=timezone.utc),
        expected_timeframe="1h",
    )

    assert bars[0].timestamp == datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc)
    assert bars[0].close == Decimal("101")
    assert bars[0].adjusted_close == Decimal("100.5")
    assert report.timezone_valid is True
    assert report.quality_status is QualityStatus.PASS


def test_normalization_rejects_naive_provider_timestamp(aapl_metadata) -> None:
    row = ProviderBar(
        "AAPL", "1h", datetime(2024, 7, 1, 9, 30), 100, 102, 99, 101, 10
    )
    with pytest.raises(DataValidationError) as raised:
        normalize_provider_bars(
            (row,),
            aapl_metadata,
            datetime(2024, 7, 1, tzinfo=timezone.utc),
            datetime(2024, 7, 2, tzinfo=timezone.utc),
            expected_timeframe="1h",
        )

    assert raised.value.quality_report is not None
    assert raised.value.quality_report.timezone_valid is False
    assert raised.value.quality_report.invalid_bar_count == 1


def test_normalization_rejects_rows_outside_requested_interval(
    aapl_metadata, aapl_hourly_rows
) -> None:
    with pytest.raises(DataValidationError) as raised:
        normalize_provider_bars(
            (aapl_hourly_rows[0],),
            aapl_metadata,
            datetime(2024, 7, 1, 14, 30, tzinfo=timezone.utc),
            datetime(2024, 7, 1, 15, 30, tzinfo=timezone.utc),
            expected_timeframe="1h",
        )

    assert raised.value.quality_report.invalid_bar_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("open", 0),
        ("high", 98),
        ("low", 102),
        ("close", -1),
        ("volume", -1),
        ("open", None),
        ("high", float("nan")),
        ("adjusted_close", 0),
    ),
)
def test_invalid_ohlcv_is_never_silently_repaired(
    field: str, value: object, aapl_metadata
) -> None:
    values = {
        "symbol": "AAPL",
        "timeframe": "1h",
        "timestamp": datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
        "open": 100,
        "high": 102,
        "low": 99,
        "close": 101,
        "volume": 10,
        "adjusted_close": 100.5,
        "source": "fake",
    }
    values[field] = value
    with pytest.raises(DataValidationError) as raised:
        normalize_provider_bars(
            (ProviderBar(**values),),
            aapl_metadata,
            datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
            datetime(2024, 7, 1, 14, 30, tzinfo=timezone.utc),
            expected_timeframe="1h",
        )

    assert raised.value.quality_report.invalid_bar_count == 1
    assert raised.value.quality_report.quality_status is QualityStatus.FAIL


def test_rows_are_sorted_deterministically(aapl_metadata, aapl_hourly_rows) -> None:
    reversed_rows = tuple(reversed(aapl_hourly_rows[:3]))
    bars, report = normalize_provider_bars(
        reversed_rows,
        aapl_metadata,
        datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
        datetime(2024, 7, 1, 16, 30, tzinfo=timezone.utc),
        expected_timeframe="1h",
    )

    assert [bar.timestamp for bar in bars] == sorted(bar.timestamp for bar in bars)
    assert report.sorted is True
    assert report.quality_status is QualityStatus.WARNING
    assert "reordered" in " ".join(report.warnings)


def test_duplicate_bar_is_reported_and_rejected(aapl_metadata, aapl_hourly_rows) -> None:
    duplicate_rows = aapl_hourly_rows[:2] + (aapl_hourly_rows[0],)
    with pytest.raises(DataValidationError) as raised:
        normalize_provider_bars(
            duplicate_rows,
            aapl_metadata,
            datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
            datetime(2024, 7, 1, 15, 30, tzinfo=timezone.utc),
            expected_timeframe="1h",
        )

    assert raised.value.quality_report.duplicate_count == 1


@pytest.mark.parametrize(
    "change",
    (
        {"symbol": "MSFT"},
        {"timeframe": "1d"},
    ),
)
def test_provider_identity_mismatch_is_rejected(change, aapl_metadata) -> None:
    values = {
        "symbol": "AAPL",
        "timeframe": "1h",
        "timestamp": datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
        "open": 100,
        "high": 102,
        "low": 99,
        "close": 101,
        "volume": 10,
        "source": "fake",
    }
    values.update(change)
    with pytest.raises(DataValidationError):
        normalize_provider_bars(
            (ProviderBar(**values),),
            aapl_metadata,
            datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc),
            datetime(2024, 7, 1, 14, 30, tzinfo=timezone.utc),
            expected_timeframe="1h",
        )


def test_market_bar_is_validated_and_immutable() -> None:
    bar = MarketBar(
        "AAPL",
        "1d",
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        Decimal("100"),
        Decimal("102"),
        Decimal("99"),
        Decimal("101"),
        Decimal("10"),
        Decimal("100.5"),
        "fake",
    )
    with pytest.raises(FrozenInstanceError):
        bar.close = Decimal("1")  # type: ignore[misc]


def test_corporate_actions_are_explicit_and_immutable(aapl_actions) -> None:
    dividend, split = aapl_actions
    assert isinstance(dividend, Dividend)
    assert dividend.action_type is CorporateActionType.DIVIDEND
    assert isinstance(split, StockSplit)
    assert split.action_type is CorporateActionType.SPLIT
    with pytest.raises(FrozenInstanceError):
        dividend.value = Decimal("1")  # type: ignore[misc]


def test_provider_errors_share_project_data_error_base() -> None:
    assert issubclass(DataProviderError, DataError)
