"""Clock observation only: no trading, NTP adjustment or threshold calibration."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from trading_ai.brokers.models import _utc
from trading_ai.brokers.exceptions import BrokerUnavailableError
from trading_ai.core.hashing import stable_hash


@dataclass(frozen=True)
class HeartbeatClockSample:
    session_id: str
    request_id: str
    requested_at_utc: datetime
    received_at_utc: datetime
    request_monotonic: float
    receive_monotonic: float
    server_epoch: int
    source: str = "IBKR_CURRENT_TIME"
    server_timestamp_resolution_seconds: float = 1.0
    sample_id: str = field(init=False)
    server_time_utc: datetime = field(init=False)
    round_trip_ms: float = field(init=False)
    raw_offset_seconds: float = field(init=False)
    raw_server_time_offset_seconds: float = field(init=False)
    midpoint_offset_estimate_seconds: float = field(init=False)
    offset_uncertainty_seconds: float = field(init=False)
    certain_clock_offset_seconds: float = field(init=False)
    wall_clock_step_seconds: float = field(init=False)

    def __post_init__(self):
        _utc(self.requested_at_utc, "requested_at_utc")
        _utc(self.received_at_utc, "received_at_utc")
        if not self.session_id or not self.request_id:
            raise ValueError("clock sample requires session/request lineage")
        if (not all(math.isfinite(x) for x in (self.request_monotonic, self.receive_monotonic))
                or self.receive_monotonic < self.request_monotonic
                or not math.isfinite((self.receive_monotonic - self.request_monotonic) * 1000)):
            raise ValueError("CLOCK_RTT_INVALID")
        if type(self.server_epoch) is not int or self.server_timestamp_resolution_seconds != 1.0:
            raise ValueError("CURRENT_TIME requires integer epoch seconds and one-second resolution")
        server = datetime.fromtimestamp(self.server_epoch, timezone.utc)
        elapsed = self.receive_monotonic - self.request_monotonic
        wall_elapsed = (self.received_at_utc - self.requested_at_utc).total_seconds()
        midpoint = self.requested_at_utc + (self.received_at_utc - self.requested_at_utc) / 2
        raw = (self.received_at_utc - server).total_seconds()
        values = dict(server_time_utc=server, round_trip_ms=elapsed * 1000,
                      raw_offset_seconds=raw, raw_server_time_offset_seconds=abs(raw),
                      midpoint_offset_estimate_seconds=(midpoint - server).total_seconds(),
                      wall_clock_step_seconds=wall_elapsed - elapsed,
                      offset_uncertainty_seconds=elapsed / 2 + 1.0 + abs(wall_elapsed - elapsed) / 2)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "certain_clock_offset_seconds", max(
            0.0, abs(self.midpoint_offset_estimate_seconds) - self.offset_uncertainty_seconds))
        object.__setattr__(self, "sample_id", "clock-" + stable_hash((
            self.session_id, self.request_id, self.requested_at_utc, self.received_at_utc,
            self.request_monotonic, self.receive_monotonic, self.server_epoch,
        ))[:24])


class ClockRequestTracker:
    """Single-flight association for an SDK callback without a request ID.

    Timeout/overlap/ambiguous replies poison this connection's clock channel.
    No retry on that channel: reconnect is required. A repeated integer epoch
    while another request is pending is ambiguous, not a new precise sample.
    """
    def __init__(self, session_id: str, *, timeout_seconds: float, now=None, monotonic=None):
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self._lock = threading.RLock()
        self._sequence = 0
        self._pending = None
        self.latest: HeartbeatClockSample | None = None
        self.error: str | None = None

    def reset_connection(self):
        with self._lock:
            self._pending = None
            self.latest = None
            self.error = None

    def begin(self) -> str:
        with self._lock:
            if self.error or self._pending:
                self.invalidate("CLOCK_REQUEST_OVERLAP_OR_UNCERTAIN")
                raise BrokerUnavailableError(self.error)
            self._sequence += 1
            request_id = f"clock-request-{self._sequence:08d}"
            self._pending = (request_id, self.now(), self.monotonic())
            return request_id

    def invalidate(self, reason: str):
        with self._lock:
            self.error = reason
            self._pending = None

    def receive(self, epoch: int, *, received_at=None, receive_monotonic=None) -> HeartbeatClockSample | None:
        with self._lock:
            if self.error:
                return None
            if type(epoch) is not int:
                self.invalidate("CLOCK_MEASUREMENT_INVALID")
                return None
            if self._pending is None:
                if self.latest is not None and epoch == self.latest.server_epoch:
                    return None  # duplicate of the completed response, no second sample
                self.invalidate("CLOCK_RESPONSE_UNSOLICITED")
                return None
            request_id, sent_at, sent_mono = self._pending
            received_at = received_at if received_at is not None else self.now()
            receive_monotonic = receive_monotonic if receive_monotonic is not None else self.monotonic()
            if self.latest is not None and epoch <= self.latest.server_epoch:
                self.invalidate("CLOCK_RESPONSE_AMBIGUOUS")
                return None
            if receive_monotonic - sent_mono > self.timeout_seconds:
                self.invalidate("CLOCK_RESPONSE_EXPIRED")
                return None
            try:
                sample = HeartbeatClockSample(self.session_id, request_id, sent_at, received_at,
                                              sent_mono, receive_monotonic, epoch)
            except (ValueError, OverflowError, OSError):
                self.invalidate("CLOCK_MEASUREMENT_INVALID")
                return None
            self._pending = None
            self.latest = sample
            return sample


def clock_summary(samples: tuple[HeartbeatClockSample, ...], warning: float, hard: float) -> dict:
    """Nearest-rank p95; absolute offsets for aggregate maxima (signed in samples)."""
    def p95(values):
        return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None
    raw = [s.raw_server_time_offset_seconds for s in samples]
    certain = [s.certain_clock_offset_seconds for s in samples]
    rtts = [s.round_trip_ms for s in samples]
    warns = [s for s in samples if s.certain_clock_offset_seconds > warning]
    fails = [s for s in samples if s.certain_clock_offset_seconds > hard]
    peak = max(samples, key=lambda s: s.certain_clock_offset_seconds) if samples else None
    warning_peak = max(warns, key=lambda s: s.certain_clock_offset_seconds) if warns else None
    return dict(
        clock_gate_metric="certain_clock_offset_seconds", clock_sample_count=len(samples),
        clock_certain_offset_current_seconds=certain[-1] if certain else None,
        clock_certain_offset_max_seconds=max(certain) if certain else None,
        clock_certain_offset_p95_seconds=p95(certain),
        clock_warning_threshold_seconds=warning, clock_hard_threshold_seconds=hard,
        clock_raw_offset_current_seconds=raw[-1] if raw else None,
        clock_raw_offset_max_seconds=max(raw) if raw else None, clock_raw_offset_p95_seconds=p95(raw),
        clock_midpoint_estimate_max_seconds=max(abs(s.midpoint_offset_estimate_seconds) for s in samples) if samples else None,
        clock_rtt_current_ms=rtts[-1] if rtts else None,
        clock_rtt_max_ms=max(rtts) if rtts else None, clock_rtt_p95_ms=p95(rtts),
        clock_warning_count=len(warns), clock_hard_failure_count=len(fails),
        clock_warning_first_at=warns[0].received_at_utc if warns else None,
        clock_warning_last_at=warns[-1].received_at_utc if warns else None,
        clock_warning_max_sample_id=warning_peak.sample_id if warning_peak else None,
        clock_warning_max_value_seconds=warning_peak.certain_clock_offset_seconds if warning_peak else None,
        clock_peak_sample_id=peak.sample_id if peak else None,
        clock_peak_at=peak.received_at_utc if peak else None,
        clock_hard_failure_sample_id=peak.sample_id if fails else None,
    )
