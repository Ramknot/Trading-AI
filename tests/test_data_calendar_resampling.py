from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_ai.data.calendar import MarketCalendarService
from trading_ai.data.exceptions import DataValidationError
from trading_ai.data.quality import normalize_provider_bars
from trading_ai.data.resampling import resample_1h_to_4h


def test_weekend_is_not_reported_as_missing(aapl_metadata) -> None:
    calendar = MarketCalendarService()
    friday_open = datetime(2024, 7, 5, 13, 30, tzinfo=timezone.utc)
    timestamps = tuple(friday_open + timedelta(hours=index) for index in range(7))

    missing, spans, outside = calendar.gap_counts(
        timestamps,
        aapl_metadata,
        datetime(2024, 7, 5, tzinfo=timezone.utc),
        datetime(2024, 7, 8, tzinfo=timezone.utc),
        "1h",
    )

    assert (missing, spans, outside) == (0, 0, 0)


def test_gap_during_open_session_is_reported(aapl_metadata) -> None:
    calendar = MarketCalendarService()
    session_open = datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc)
    timestamps = tuple(
        session_open + timedelta(hours=index)
        for index in range(7)
        if index != 3
    )

    missing, spans, outside = calendar.gap_counts(
        timestamps,
        aapl_metadata,
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        datetime(2024, 7, 2, tzinfo=timezone.utc),
        "1h",
    )

    assert (missing, spans, outside) == (1, 1, 0)


def test_exchange_holiday_is_not_an_expected_session(aapl_metadata) -> None:
    calendar = MarketCalendarService()
    keys = calendar.expected_keys(
        aapl_metadata,
        datetime(2024, 7, 3, tzinfo=timezone.utc),
        datetime(2024, 7, 6, tzinfo=timezone.utc),
        "1h",
    )

    assert keys
    assert all(key.date().isoformat() != "2024-07-04" for key in keys)


def test_market_calendars_distinguish_new_york_and_paris(
    aapl_metadata, paris_metadata
) -> None:
    calendar = MarketCalendarService()
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    end = datetime(2024, 7, 2, tzinfo=timezone.utc)

    nyse_first = calendar.expected_keys(aapl_metadata, start, end, "1h")[0]
    paris_first = calendar.expected_keys(paris_metadata, start, end, "1h")[0]

    assert nyse_first != paris_first
    assert nyse_first.hour == 13 and nyse_first.minute == 30
    assert paris_first.hour == 7 and paris_first.minute == 0


def test_quality_report_counts_missing_expected_bar(
    aapl_metadata, aapl_hourly_rows, market_start, market_end
) -> None:
    bars, report = normalize_provider_bars(
        aapl_hourly_rows[:4] + aapl_hourly_rows[5:],
        aapl_metadata,
        market_start,
        market_end,
        expected_timeframe="1h",
    )

    assert len(bars) == 13
    assert report.missing_expected_bar_count == 1
    assert report.unexpected_gap_count == 1


def test_resampling_uses_correct_ohlcv_and_session_anchors(
    aapl_metadata, aapl_hourly_rows
) -> None:
    derived = resample_1h_to_4h(aapl_hourly_rows, aapl_metadata)

    assert len(derived) == 4
    first = derived[0]
    assert first.timestamp == datetime(2024, 7, 1, 13, 30, tzinfo=timezone.utc)
    assert first.open == Decimal("100")
    assert first.high == Decimal("105")
    assert first.low == Decimal("99")
    assert first.close == Decimal("104")
    assert first.volume == Decimal("100")
    assert first.adjusted_close == Decimal("103.5")
    second = derived[1]
    assert second.timestamp == datetime(2024, 7, 1, 17, 30, tzinfo=timezone.utc)
    assert second.volume == Decimal("180")


def test_resampling_never_mixes_sessions(aapl_metadata, aapl_hourly_rows) -> None:
    derived = resample_1h_to_4h(aapl_hourly_rows, aapl_metadata)

    assert [bar.timestamp.date().isoformat() for bar in derived] == [
        "2024-07-01",
        "2024-07-01",
        "2024-07-02",
        "2024-07-02",
    ]
    assert derived[2].open == Decimal("100")


def test_partial_session_produces_deterministic_partial_bucket(
    aapl_metadata, aapl_hourly_rows
) -> None:
    derived = resample_1h_to_4h(aapl_hourly_rows[:2], aapl_metadata)

    assert len(derived) == 1
    assert derived[0].open == Decimal("100")
    assert derived[0].close == Decimal("102")
    assert derived[0].volume == Decimal("30")


def test_resampling_rejects_off_session_bar(aapl_metadata, aapl_hourly_rows) -> None:
    invalid = aapl_hourly_rows[0].__class__(
        symbol="AAPL",
        timeframe="1h",
        timestamp=datetime(2024, 7, 1, 1, tzinfo=timezone.utc),
        open=100,
        high=102,
        low=99,
        close=101,
        volume=10,
        source="fake",
    )
    bars, _ = normalize_provider_bars(
        (invalid,),
        aapl_metadata,
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        datetime(2024, 7, 2, tzinfo=timezone.utc),
        expected_timeframe="1h",
    )
    with pytest.raises(DataValidationError, match="outside an exchange session"):
        resample_1h_to_4h(bars, aapl_metadata)
