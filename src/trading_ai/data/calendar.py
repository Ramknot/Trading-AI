"""Exchange-calendar isolation for expected bars, gaps, and session anchoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Hashable, Iterable
from zoneinfo import ZoneInfo

from trading_ai.data.exceptions import DataValidationError
from trading_ai.data.models import InstrumentMetadata


@dataclass(frozen=True, slots=True)
class SessionWindow:
    session_date: date
    market_open: datetime
    market_close: datetime


@lru_cache(maxsize=512)
def _load_session_windows(
    calendar_name: str,
    start_date: date,
    end_date: date,
) -> tuple[SessionWindow, ...]:
    import pandas_market_calendars as market_calendars

    calendar = market_calendars.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    sessions: list[SessionWindow] = []
    for session_label, row in schedule.iterrows():
        sessions.append(
            SessionWindow(
                session_date=session_label.date(),
                market_open=row["market_open"].to_pydatetime().astimezone(timezone.utc),
                market_close=row["market_close"].to_pydatetime().astimezone(timezone.utc),
            )
        )
    return tuple(sessions)


class MarketCalendarService:
    """Wrap pandas-market-calendars so its objects do not leak into the domain."""

    def _sessions(
        self,
        metadata: InstrumentMetadata,
        start: datetime,
        end: datetime,
    ) -> tuple[SessionWindow, ...]:
        try:
            if (
                start.tzinfo is None
                or start.utcoffset() is None
                or end.tzinfo is None
                or end.utcoffset() is None
            ):
                raise ValueError("calendar bounds must be timezone-aware")
            local_zone = ZoneInfo(metadata.exchange_timezone)
            start_date = (start.astimezone(local_zone).date() - timedelta(days=1))
            end_date = (end.astimezone(local_zone).date() + timedelta(days=1))
            return _load_session_windows(metadata.calendar, start_date, end_date)
        except Exception as exc:
            raise DataValidationError(
                f"unable to load market calendar {metadata.calendar!r}"
            ) from exc

    def expected_keys(
        self,
        metadata: InstrumentMetadata,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> tuple[Hashable, ...]:
        sessions = self._sessions(metadata, start, end)
        if timeframe == "1d":
            return tuple(
                session.session_date
                for session in sessions
                if session.market_close > start and session.market_open < end
            )
        if timeframe not in {"1h", "4h"}:
            raise DataValidationError(f"unsupported calendar timeframe {timeframe!r}")
        step = timedelta(hours=1 if timeframe == "1h" else 4)
        expected: list[datetime] = []
        for session in sessions:
            cursor = session.market_open
            while cursor < session.market_close:
                if start <= cursor < end:
                    expected.append(cursor)
                cursor += step
        return tuple(expected)

    def bar_key(
        self,
        timestamp: datetime,
        timeframe: str,
        metadata: InstrumentMetadata,
    ) -> Hashable:
        if timeframe == "1d":
            return timestamp.astimezone(ZoneInfo(metadata.exchange_timezone)).date()
        return timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)

    def session_for_timestamp(
        self, timestamp: datetime, metadata: InstrumentMetadata
    ) -> SessionWindow | None:
        start = timestamp - timedelta(days=2)
        end = timestamp + timedelta(days=2)
        for session in self._sessions(metadata, start, end):
            if session.market_open <= timestamp < session.market_close:
                return session
        return None

    def gap_counts(
        self,
        actual_timestamps: Iterable[datetime],
        metadata: InstrumentMetadata,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> tuple[int, int, int]:
        """Return missing expected bars, missing spans, and off-session bars."""

        expected = self.expected_keys(metadata, start, end, timeframe)
        actual = {
            self.bar_key(timestamp, timeframe, metadata)
            for timestamp in actual_timestamps
        }
        missing_flags = [key not in actual for key in expected]
        missing_count = sum(missing_flags)
        missing_spans = 0
        previous_missing = False
        for is_missing in missing_flags:
            if is_missing and not previous_missing:
                missing_spans += 1
            previous_missing = is_missing
        expected_set = set(expected)
        unexpected_count = sum(key not in expected_set for key in actual)
        return missing_count, missing_spans, unexpected_count
