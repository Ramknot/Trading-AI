"""Offline-only read-only fault fixture; never exposed by production CLI."""
from dataclasses import replace
from datetime import timedelta

from trading_ai.brokers.fake import FakeBroker
from trading_ai.brokers.exceptions import BrokerUnavailableError, PaperExecutionLockedError
from trading_ai.brokers.models import BrokerConnectionState


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

    @property
    def account_identity(self):
        return self.account

    def heartbeat(self):
        self._faults()
        if self._connection is not BrokerConnectionState.CONNECTED:
            raise BrokerUnavailableError("fake disconnected")
        if not self.reader_stale and not self.reader_failed:
            # Server replies may arrive within the same simulated tick.
            self.last_heartbeat = max(self.clock.now(), (self.last_heartbeat or self.clock.now()) + timedelta(microseconds=1))

    @property
    def last_server_time_at(self):
        return self.last_heartbeat

    def health(self):
        self._faults()
        return replace(super().health(), observed_at=self.clock.now(),
                       last_heartbeat_at=self.last_heartbeat, stale=self.reader_stale,
                       clock_drift_seconds=self.clock_drift,
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
