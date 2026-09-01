"""Immutable broker, order-lifecycle, and Paper-session contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from trading_ai.core.hashing import stable_hash
from trading_ai.core.models import OrderSide, OrderType


ZERO = Decimal("0")


def _text(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be normalized to UTC")


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")


def _json_object(value: str, name: str) -> None:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")


class BrokerEnvironment(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"
    UNKNOWN = "UNKNOWN"


class BrokerConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    STALE = "STALE"


class PaperSessionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED_UNVERIFIED = "CONNECTED_UNVERIFIED"
    PAPER_VERIFIED = "PAPER_VERIFIED"
    RECONCILING = "RECONCILING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"
    STOPPING = "STOPPING"


class PaperMode(str, Enum):
    CONNECTIVITY_CHECK = "CONNECTIVITY_CHECK"
    PAPER_READ_ONLY = "PAPER_READ_ONLY"
    PAPER_EXECUTION_ARMED = "PAPER_EXECUTION_ARMED"


class BrokerOrderState(str, Enum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


class BrokerEventType(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    ACCOUNT = "ACCOUNT"
    POSITIONS = "POSITIONS"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACK = "ORDER_ACK"
    ORDER_STATUS = "ORDER_STATUS"
    ORDER_REJECT = "ORDER_REJECT"
    ORDER_CANCEL = "ORDER_CANCEL"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    EXECUTION_CORRECTION = "EXECUTION_CORRECTION"
    COMMISSION_REPORT = "COMMISSION_REPORT"
    ERROR = "ERROR"
    WARNING = "WARNING"
    SESSION_STALE = "SESSION_STALE"
    RECONCILIATION_RESULT = "RECONCILIATION_RESULT"
    EXTERNAL_BROKER_ACTIVITY = "EXTERNAL_BROKER_ACTIVITY"
    EMERGENCY_HALT = "EMERGENCY_HALT"


class BrokerErrorSeverity(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    WARNING = "WARNING"
    REJECT = "REJECT"
    CONNECTIVITY = "CONNECTIVITY"
    CRITICAL = "CRITICAL"


class ReconciliationStatus(str, Enum):
    IN_SYNC = "IN_SYNC"
    DRIFT = "DRIFT"
    CRITICAL_DRIFT = "CRITICAL_DRIFT"
    UNKNOWN = "UNKNOWN"


class DataFreshnessStatus(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class CommissionKnowledge(str, Enum):
    KNOWN = "KNOWN"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class BrokerAccountIdentity:
    broker: str
    account_hash: str
    account_masked: str
    environment: BrokerEnvironment
    base_currency: str
    capabilities: tuple[str, ...]
    environment_verified: bool
    environment_evidence: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.broker, "broker"),
            (self.account_masked, "account_masked"),
            (self.base_currency, "base_currency"),
            (self.environment_evidence, "environment_evidence"),
        ):
            _text(value, name)
        _sha256(self.account_hash, "account_hash")
        if self.capabilities != tuple(sorted(set(self.capabilities))):
            raise ValueError("capabilities must be sorted and unique")
        if self.environment is not BrokerEnvironment.PAPER and self.environment_verified:
            raise ValueError("only an explicit PAPER identity may be verified")


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    symbol: str
    quantity: Decimal
    average_cost: Decimal
    currency: str

    def __post_init__(self) -> None:
        _text(self.symbol, "symbol")
        _text(self.currency, "currency")
        if not self.quantity.is_finite():
            raise ValueError("position quantity must be finite")
        if not self.average_cost.is_finite() or self.average_cost < ZERO:
            raise ValueError("average_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BrokerAccountSnapshot:
    observed_at: datetime
    account: BrokerAccountIdentity
    cash: Decimal
    net_liquidation: Decimal
    positions: tuple[BrokerPosition, ...]

    def __post_init__(self) -> None:
        _utc(self.observed_at, "observed_at")
        for value, name in ((self.cash, "cash"), (self.net_liquidation, "net_liquidation")):
            if not value.is_finite():
                raise ValueError(f"{name} must be finite")
        symbols = tuple(item.symbol for item in self.positions)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("positions must be sorted by unique symbol")


@dataclass(frozen=True, slots=True)
class BrokerOrderRecord:
    internal_order_id: str
    client_order_key: str
    session_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    filled_quantity: Decimal
    state: BrokerOrderState
    risk_decision_id: str
    created_at: datetime
    updated_at: datetime
    tif: str = "DAY"
    limit_price: Decimal | None = None
    broker_order_id: str | None = None
    perm_id: str | None = None
    rejection_code: str | None = None
    external: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.internal_order_id, "internal_order_id"),
            (self.client_order_key, "client_order_key"),
            (self.session_id, "session_id"),
            (self.symbol, "symbol"),
            (self.risk_decision_id, "risk_decision_id"),
            (self.tif, "tif"),
        ):
            _text(value, name)
        _utc(self.created_at, "created_at")
        _utc(self.updated_at, "updated_at")
        if self.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError("Lot 9 supports MARKET and LIMIT orders only")
        if not self.quantity.is_finite() or self.quantity <= ZERO:
            raise ValueError("order quantity must be positive and finite")
        if (
            not self.filled_quantity.is_finite()
            or self.filled_quantity < ZERO
            or self.filled_quantity > self.quantity
        ):
            raise ValueError("filled quantity must be in [0, quantity]")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders require limit_price")
        if self.limit_price is not None and (
            not self.limit_price.is_finite() or self.limit_price <= ZERO
        ):
            raise ValueError("limit_price must be positive and finite")
        for value in (self.broker_order_id, self.perm_id, self.rejection_code):
            if value is not None:
                _text(value, "optional order identifier")


@dataclass(frozen=True, slots=True)
class BrokerExecution:
    exec_id: str
    internal_order_id: str
    client_order_key: str
    broker_order_id: str
    perm_id: str | None
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    broker_timestamp: datetime | None
    received_at: datetime
    broker_timestamp_raw: str | None = None
    correction_of: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.exec_id, "exec_id"),
            (self.internal_order_id, "internal_order_id"),
            (self.client_order_key, "client_order_key"),
            (self.broker_order_id, "broker_order_id"),
            (self.symbol, "symbol"),
        ):
            _text(value, name)
        if self.broker_timestamp is not None:
            _utc(self.broker_timestamp, "broker_timestamp")
        _utc(self.received_at, "received_at")
        if not self.quantity.is_finite() or self.quantity <= ZERO:
            raise ValueError("execution quantity must be positive and finite")
        if not self.price.is_finite() or self.price <= ZERO:
            raise ValueError("execution price must be positive and finite")
        if self.correction_of is not None:
            _text(self.correction_of, "correction_of")
        if self.broker_timestamp_raw is not None:
            _text(self.broker_timestamp_raw, "broker_timestamp_raw")


@dataclass(frozen=True, slots=True)
class BrokerCommissionReport:
    exec_id: str
    status: CommissionKnowledge
    received_at: datetime
    amount: Decimal | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        _text(self.exec_id, "exec_id")
        _utc(self.received_at, "received_at")
        if self.status is CommissionKnowledge.UNAVAILABLE:
            if self.amount is not None:
                raise ValueError("unavailable commission must not carry zero or an amount")
        elif self.amount is None or not self.amount.is_finite() or self.amount < ZERO:
            raise ValueError("known commission requires a non-negative finite amount")
        if self.amount is not None and self.currency is None:
            raise ValueError("known commission requires a currency")


@dataclass(frozen=True, slots=True)
class BrokerEvent:
    event_id: str
    session_id: str
    event_type: BrokerEventType
    received_at: datetime
    source: str
    source_version: str
    broker_timestamp: datetime | None = None
    related_ids: tuple[tuple[str, str], ...] = ()
    payload_json: str = "{}"

    def __post_init__(self) -> None:
        for value, name in (
            (self.event_id, "event_id"),
            (self.session_id, "session_id"),
            (self.source, "source"),
            (self.source_version, "source_version"),
        ):
            _text(value, name)
        _utc(self.received_at, "received_at")
        if self.broker_timestamp is not None:
            _utc(self.broker_timestamp, "broker_timestamp")
        if self.related_ids != tuple(sorted(set(self.related_ids))):
            raise ValueError("related_ids must be sorted and unique")
        _json_object(self.payload_json, "payload_json")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


@dataclass(frozen=True, slots=True)
class BrokerHealth:
    observed_at: datetime
    connection_state: BrokerConnectionState
    session_state: PaperSessionState
    stale: bool
    last_heartbeat_at: datetime | None
    critical_errors: tuple[str, ...] = ()
    clock_drift_seconds: float | None = None

    def __post_init__(self) -> None:
        _utc(self.observed_at, "observed_at")
        if self.last_heartbeat_at is not None:
            _utc(self.last_heartbeat_at, "last_heartbeat_at")
        if self.critical_errors != tuple(sorted(set(self.critical_errors))):
            raise ValueError("critical_errors must be sorted and unique")
        if self.clock_drift_seconds is not None and self.clock_drift_seconds < 0:
            raise ValueError("clock_drift_seconds must not be negative")


@dataclass(frozen=True, slots=True)
class PaperEmergencyHalt:
    active: bool
    reason: str
    changed_at: datetime

    def __post_init__(self) -> None:
        _text(self.reason, "halt reason")
        _utc(self.changed_at, "changed_at")


@dataclass(frozen=True, slots=True)
class PaperSessionManifest:
    session_id: str
    created_at: datetime
    code_sha: str
    mode: PaperMode
    broker_adapter_name: str
    broker_adapter_version: str
    official_sdk_version: str | None
    server_version: str | None
    account_hash: str
    account_masked: str
    config_hashes: tuple[tuple[str, str], ...]
    ml_model_ids: tuple[str, ...]
    paper_execution_armed: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.session_id, "session_id"),
            (self.code_sha, "code_sha"),
            (self.broker_adapter_name, "broker_adapter_name"),
            (self.broker_adapter_version, "broker_adapter_version"),
            (self.account_masked, "account_masked"),
        ):
            _text(value, name)
        _utc(self.created_at, "created_at")
        _sha256(self.account_hash, "account_hash")
        if self.config_hashes != tuple(sorted(set(self.config_hashes))):
            raise ValueError("config_hashes must be sorted and unique")
        for _, digest in self.config_hashes:
            _sha256(digest, "component config hash")
        if self.ml_model_ids != tuple(sorted(set(self.ml_model_ids))):
            raise ValueError("ml_model_ids must be sorted and unique")
        if self.paper_execution_armed:
            raise ValueError("Lot 9 session manifests must keep Paper execution unarmed")

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True, slots=True)
class PaperDecisionEnvelope:
    envelope_id: str
    session_id: str
    timestamp: datetime
    symbol: str
    order_id: str
    risk_decision_id: str
    data_snapshot_id: str
    feature_snapshot_id: str
    regime_snapshot_id: str
    signal_id: str
    ml_decision_id: str | None
    activation_decision_id: str
    portfolio_plan_id: str
    cost_estimate_id: str
    economic_decision_id: str
    account_snapshot_hash: str
    broker_health_hash: str
    reconciliation_hash: str
    payload_json: str = "{}"

    def __post_init__(self) -> None:
        for field in (
            "envelope_id", "session_id", "symbol", "order_id", "risk_decision_id",
            "data_snapshot_id", "feature_snapshot_id", "regime_snapshot_id", "signal_id",
            "activation_decision_id", "portfolio_plan_id", "cost_estimate_id",
            "economic_decision_id",
        ):
            _text(str(getattr(self, field)), field)
        _utc(self.timestamp, "timestamp")
        for field in ("account_snapshot_hash", "broker_health_hash", "reconciliation_hash"):
            _sha256(str(getattr(self, field)), field)
        _json_object(self.payload_json, "payload_json")


@dataclass(frozen=True, slots=True)
class PaperOutcomeEnvelope:
    envelope_id: str
    decision_envelope_id: str
    session_id: str
    completed_at: datetime
    order_id: str
    broker_order_id: str
    final_state: BrokerOrderState
    execution_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    estimated_cost: Decimal | None
    broker_reported_cost: Decimal | None
    resulting_cash: Decimal | None
    resulting_positions: tuple[BrokerPosition, ...]
    resulting_equity: Decimal | None
    reconciliation_hash: str

    def __post_init__(self) -> None:
        for field in (
            "envelope_id", "decision_envelope_id", "session_id", "order_id",
            "broker_order_id",
        ):
            _text(str(getattr(self, field)), field)
        _utc(self.completed_at, "completed_at")
        if self.execution_ids != tuple(sorted(set(self.execution_ids))):
            raise ValueError("execution_ids must be sorted and unique")
        if self.event_ids != tuple(sorted(set(self.event_ids))):
            raise ValueError("event_ids must be sorted and unique")
        position_symbols = tuple(item.symbol for item in self.resulting_positions)
        if position_symbols != tuple(sorted(set(position_symbols))):
            raise ValueError("resulting_positions must be sorted by unique symbol")
        for field in (
            "estimated_cost", "broker_reported_cost", "resulting_cash", "resulting_equity"
        ):
            value = getattr(self, field)
            if value is not None and not value.is_finite():
                raise ValueError(f"{field} must be finite when available")
        _sha256(self.reconciliation_hash, "reconciliation_hash")


@dataclass(frozen=True, slots=True)
class ExpectedObservedMetrics:
    decision_to_submit_ms: int | None
    submit_to_ack_ms: int | None
    submit_to_fill_ms: int | None
    expected_fill_price: Decimal | None
    observed_average_fill_price: Decimal | None
    estimated_slippage: Decimal | None
    observed_slippage: Decimal | None
    estimated_commission: Decimal | None
    broker_commission: Decimal | None
    rejects: int
    cancels: int
    partial_fills: int
    reconnects: int
    drifts: int

    def __post_init__(self) -> None:
        for field in (
            "decision_to_submit_ms", "submit_to_ack_ms", "submit_to_fill_ms",
            "rejects", "cancels", "partial_fills", "reconnects", "drifts",
        ):
            value = getattr(self, field)
            if value is not None and value < 0:
                raise ValueError(f"{field} must not be negative")
        for field in (
            "expected_fill_price", "observed_average_fill_price", "estimated_slippage",
            "observed_slippage", "estimated_commission", "broker_commission",
        ):
            value = getattr(self, field)
            if value is not None and not value.is_finite():
                raise ValueError(f"{field} must be finite")


@dataclass(frozen=True, slots=True)
class PaperSafetySnapshot:
    session_id: str
    observed_at: datetime
    mode: PaperMode
    session_state: PaperSessionState
    account: BrokerAccountIdentity | None
    connection_state: BrokerConnectionState
    reconciliation_status: ReconciliationStatus
    reconciliation_hash: str
    data_freshness: DataFreshnessStatus
    risk_health_ok: bool
    market_session_open: bool | None
    configs_frozen: bool
    emergency_halt: PaperEmergencyHalt

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _utc(self.observed_at, "observed_at")
        _sha256(self.reconciliation_hash, "reconciliation_hash")
