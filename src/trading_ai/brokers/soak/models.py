"""Immutable soak contracts; no order submission capability."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

from trading_ai.brokers.models import (
    BrokerAccountSnapshot, BrokerCommissionReport, BrokerHealth,
    ReconciliationStatus, _utc,
)
from trading_ai.brokers.reconciliation import ReconciliationState
from trading_ai.core.hashing import stable_hash


class SoakState(str, Enum):
    STARTING = "STARTING"
    CONNECTING = "CONNECTING"
    PAPER_VERIFIED = "PAPER_VERIFIED"
    RECONCILING = "RECONCILING"
    SOAK_RUNNING = "SOAK_RUNNING"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class SoakConfig:
    duration_seconds: float = 3600
    snapshot_seconds: float = 30
    smoke_minimum_seconds: float = 900
    soak_minimum_seconds: float = 3600
    heartbeat_seconds: float = 10
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 10
    cash_tolerance: Decimal = Decimal("0.01")
    equity_tolerance: Decimal = Decimal("0.01")
    clock_warning_seconds: float = 2

    def __post_init__(self):
        for name in ("duration_seconds", "snapshot_seconds", "smoke_minimum_seconds",
                     "soak_minimum_seconds", "heartbeat_seconds", "reconnect_delay_seconds",
                     "clock_warning_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid soak {name}")
        if self.snapshot_seconds < 30 or self.heartbeat_seconds < 5:
            raise ValueError("soak polling must not be aggressive")
        if self.smoke_minimum_seconds < 600 or self.soak_minimum_seconds < 3600:
            raise ValueError("smoke requires >=10min; soak requires >=60min")
        if self.soak_minimum_seconds <= self.smoke_minimum_seconds:
            raise ValueError("soak must be longer than smoke")
        if type(self.reconnect_attempts) is not int or not 0 <= self.reconnect_attempts <= 5:
            raise ValueError("reconnect attempts must be bounded")
        for value in (self.cash_tolerance, self.equity_tolerance):
            if not value.is_finite() or value < 0:
                raise ValueError("invalid reconciliation tolerance")

    @property
    def config_hash(self):
        return stable_hash(self)


def load_soak_config(path: Path | str) -> SoakConfig:
    try:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("read-only soak configuration unavailable or invalid") from exc
    for name in ("cash_tolerance", "equity_tolerance"):
        if name in raw:
            raw[name] = Decimal(str(raw[name]))
    try:
        return SoakConfig(**raw)
    except TypeError as exc:
        raise ValueError("unsupported read-only soak configuration field/type") from exc


@dataclass(frozen=True)
class ReadOnlySnapshot:
    session_id: str
    sequence: int
    timestamp: datetime
    account: BrokerAccountSnapshot
    state: ReconciliationState
    commissions: tuple[BrokerCommissionReport, ...]
    health: BrokerHealth
    adapter_version: str
    sdk_version: str | None
    server_version: str | None
    latency_seconds: float
    missing: tuple[str, ...] = ()
    ownership: str = "BROKER_BOOTSTRAP_READ_ONLY"

    def __post_init__(self):
        _utc(self.timestamp, "timestamp")
        if self.sequence < 0 or not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("invalid snapshot sequence/latency")

    @property
    def snapshot_id(self):
        return "soak-snapshot-" + stable_hash(self)[:24]


@dataclass(frozen=True)
class SoakReconciliation:
    timestamp: datetime
    baseline_id: str
    snapshot_id: str
    status: ReconciliationStatus
    reasons: tuple[str, ...]
    external_activity: bool

    def __post_init__(self):
        _utc(self.timestamp, "timestamp")


@dataclass(frozen=True)
class GateResult:
    status: str
    evidence_level: str
    reasons: tuple[str, ...]
    name: str = "paper-read-only-reconciliation"
    version: str = "1.1"


@dataclass(frozen=True)
class PaperReadOnlySoakReport:
    session_id: str
    start: datetime
    end: datetime
    duration_seconds: float
    observed_seconds: float
    uptime_seconds: float
    snapshots_count: int
    reconnects: int
    disconnect_count: int
    reconnect_duration_seconds: float
    stale_events: int
    heartbeat_gap_max_seconds: float
    snapshot_latency_max_seconds: float
    broker_callback_error_count: int
    drift_events: int
    critical_drift: int
    external_activity: int
    clock_drift_max_seconds: float | None
    snapshot_completeness: float
    health_status: str
    reconciliation_initial: str
    reconciliation_final: str
    account_verified: bool
    integrity: str
    state: SoakState
    gate: GateResult
    config_hash: str
    previous_session_id: str | None
    warnings: tuple[str, ...]
    submit_order_calls: int = 0
    cancel_order_calls: int = 0
    paper_execution_armed: bool = False
    live_hard_locked: bool = True
    decision_data_freshness: str = "NOT_EVALUATED"
    soak_schema_version: str = "1.1"
    clock_gate_metric: str | None = None
    clock_sample_count: int | None = None
    clock_warning_threshold_seconds: float | None = None
    clock_hard_threshold_seconds: float | None = None
    clock_raw_offset_current_seconds: float | None = None
    clock_raw_offset_max_seconds: float | None = None
    clock_raw_offset_p95_seconds: float | None = None
    clock_midpoint_estimate_max_seconds: float | None = None
    clock_rtt_current_ms: float | None = None
    clock_rtt_max_ms: float | None = None
    clock_rtt_p95_ms: float | None = None
    clock_warning_count: int | None = None
    clock_hard_failure_count: int | None = None
    clock_warning_first_at: datetime | None = None
    clock_warning_last_at: datetime | None = None
    clock_warning_max_sample_id: str | None = None
    clock_warning_max_value_seconds: float | None = None
    clock_peak_sample_id: str | None = None
    clock_peak_at: datetime | None = None
    clock_hard_failure_sample_id: str | None = None

    def __post_init__(self):
        _utc(self.start, "start")
        _utc(self.end, "end")
        if self.end < self.start or self.paper_execution_armed or not self.live_hard_locked:
            raise ValueError("invalid read-only report")
        if self.clock_sample_count is not None:
            if self.clock_drift_max_seconds != self.clock_raw_offset_max_seconds:
                raise ValueError("clock maxima must cover the same evaluated samples")
            if "CLOCK_DRIFT_WARNING" in self.warnings:
                if (not self.clock_warning_count or not self.clock_warning_max_sample_id
                    or self.clock_warning_max_value_seconds is None
                    or self.clock_warning_threshold_seconds is None
                    or self.clock_warning_max_value_seconds <= self.clock_warning_threshold_seconds):
                    raise ValueError("clock warning requires persisted sample provenance")
            if "BROKER_CLOCK_DRIFT" in self.gate.reasons:
                if (not self.clock_hard_failure_count or not self.clock_hard_failure_sample_id
                    or self.clock_raw_offset_max_seconds is None or self.clock_hard_threshold_seconds is None
                    or self.clock_raw_offset_max_seconds <= self.clock_hard_threshold_seconds):
                    raise ValueError("clock failure requires persisted sample provenance")

    @property
    def report_hash(self):
        return stable_hash(self)
