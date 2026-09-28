"""Offline-only read-only fault fixture; never exposed by production CLI."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math

from trading_ai.brokers.fake import FakeBroker
from trading_ai.brokers.exceptions import BrokerUnavailableError, PaperExecutionLockedError
from trading_ai.brokers.models import BrokerConnectionState
from trading_ai.brokers.clock import ClockRequestTracker


class FakeReadOnlyBroker(FakeBroker):
    """Scheduled test callbacks use an injected clock; no wall-clock waiting."""
    sdk_version = "FAKE"
    server_version = "FAKE"

    def __init__(self, *, clock, scheduled=(), **kwargs):
        super().__init__(**kwargs)
        self.clock = clock
        self.scheduled = list(sorted(scheduled, key=lambda x: x[0]))
        self.last_heartbeat = None
        self.reader_stale = False
        self.reader_failed = False
        self.connect_fails = False
        self.clock_drift = 0.0
        self.clock_rtt_seconds = 0.0
        self.server_clock_offset_seconds = None  # optional physical RTT/second-quantization fixture
        self._clock_wall = clock.now()
        self._clock_tracker = ClockRequestTracker(self.session_id, timeout_seconds=10,
            now=lambda: self._clock_wall, monotonic=clock.monotonic)
        self.submit_order_calls = self.cancel_order_calls = 0
        self.sync_calls = self.connect_calls = 0

    def _faults(self):
        while self.scheduled and self.clock.monotonic() >= self.scheduled[0][0]:
            _, callback = self.scheduled.pop(0)
            callback(self)

    def connect(self):
        self.connect_calls += 1
        self._faults()
        if self.connect_fails:
            raise BrokerUnavailableError("fake connection failure")
        super().connect()
        self._clock_tracker.reset_connection()

    @property
    def account_identity(self):
        return self.account

    def heartbeat(self):
        self._faults()
        if self._connection is not BrokerConnectionState.CONNECTED:
            raise BrokerUnavailableError("fake disconnected")
        rtt = self.clock_rtt_seconds
        sent = self.clock.now()
        received = sent + timedelta(seconds=rtt)
        if self.server_clock_offset_seconds is None and self.clock_drift is not None:
            # An explicit raw-offset fixture, including a fractional local UTC
            # phase; the server epoch itself remains integer seconds.
            received = datetime.fromtimestamp(math.floor(received.timestamp()) + self.clock_drift % 1, timezone.utc)
            sent = received - timedelta(seconds=rtt)
        self._clock_wall = sent
        request_id = self._clock_tracker.begin()
        if self.clock_drift is None:
            self._clock_tracker.invalidate("SERVER_TIME_UNAVAILABLE")
        elif not self.reader_stale and not self.reader_failed:
            self.clock.sleep(rtt)
            epoch = (math.floor(received.timestamp() - self.clock_drift) if self.server_clock_offset_seconds is None
                     else math.floor((sent + timedelta(seconds=rtt / 2)).timestamp() - self.server_clock_offset_seconds))
            self._clock_tracker.receive(epoch, received_at=received, receive_monotonic=self.clock.monotonic())
            self.last_heartbeat = received
        return request_id

    @property
    def latest_clock_sample(self):
        return self._clock_tracker.latest

    @property
    def clock_measurement_error(self):
        return self._clock_tracker.error

    def expire_clock_request(self):
        self._clock_tracker.invalidate("CLOCK_REQUEST_TIMED_OUT")

    @property
    def last_server_time_at(self):
        return self.last_heartbeat

    def health(self):
        self._faults()
        return replace(super().health(), observed_at=self.clock.now(),
                       last_heartbeat_at=self.last_heartbeat, stale=self.reader_stale,
                       clock_drift_seconds=self.latest_clock_sample.raw_server_time_offset_seconds if self.latest_clock_sample else None,
                       critical_errors=("CALLBACK_READER_FAILED",) if self.reader_failed else ())

    def account_snapshot(self):
        return replace(super().account_snapshot(), observed_at=self.clock.now())

    def sync_state(self):
        self._faults()
        self.sync_calls += 1
        return super().sync_state()

    def submit_approved(self, *args, **kwargs):
        self.submit_order_calls += 1
        raise PaperExecutionLockedError("fake soak can never submit")

    def cancel_order(self, *args, **kwargs):
        self.cancel_order_calls += 1
        raise PaperExecutionLockedError("fake soak can never cancel")
