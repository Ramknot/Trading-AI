"""Immutable, provider-neutral contracts for observability and costs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any


ZERO = Decimal("0")


def _text(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be normalized to UTC")


def _json_object(value: str, name: str) -> None:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")


class SystemStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


class MonitoringEventType(str, Enum):
    DATA_QUALITY = "DATA_QUALITY"
    FEATURE = "FEATURE"
    REGIME = "REGIME"
    SIGNAL = "SIGNAL"
    ML_PREDICTION = "ML_PREDICTION"
    ML_DECISION = "ML_DECISION"
    ACTIVATION_DECISION = "ACTIVATION_DECISION"
    PORTFOLIO_DECISION = "PORTFOLIO_DECISION"
    RISK_DECISION = "RISK_DECISION"
    ORDER_INTENT = "ORDER_INTENT"
    FILL = "FILL"
    POSITION_UPDATE = "POSITION_UPDATE"
    EQUITY_UPDATE = "EQUITY_UPDATE"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    COST_ESTIMATE = "COST_ESTIMATE"
    ECONOMIC_DECISION = "ECONOMIC_DECISION"
    COST_ACTUAL = "COST_ACTUAL"
    COST_RECONCILIATION = "COST_RECONCILIATION"
    VALIDATION_RESULT = "VALIDATION_RESULT"
    EVIDENCE_VERIFIED = "EVIDENCE_VERIFIED"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    EVIDENCE_REASSESSMENT = "EVIDENCE_REASSESSMENT"
    PAPER_READINESS_REVIEW = "PAPER_READINESS_REVIEW"
    ECONOMIC_RECOMPUTATION_STARTED = "ECONOMIC_RECOMPUTATION_STARTED"
    ECONOMIC_RECOMPUTATION_COMPLETED = "ECONOMIC_RECOMPUTATION_COMPLETED"
    DECISION_INVARIANCE_CHECK = "DECISION_INVARIANCE_CHECK"
    PAPER_READINESS_V3 = "PAPER_READINESS_V3"
    HUMAN_READINESS_REVIEW = "HUMAN_READINESS_REVIEW"
    BROKER_EVENT = "BROKER_EVENT"
    BROKER_RECONCILIATION = "BROKER_RECONCILIATION"
    PAPER_SESSION = "PAPER_SESSION"
    PAPER_SHADOW_AUDIT = "PAPER_SHADOW_AUDIT"


class CostKnowledge(str, Enum):
    KNOWN = "KNOWN"
    ESTIMATED = "ESTIMATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class CostCoverageStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class CostComponent:
    status: CostKnowledge
    amount: Decimal | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.status is CostKnowledge.UNAVAILABLE:
            if self.amount is not None:
                raise ValueError("an unavailable cost must not carry a zero or amount")
        else:
            if self.amount is None or not self.amount.is_finite() or self.amount < ZERO:
                raise ValueError("known or estimated costs need a non-negative amount")
        if self.source is not None:
            _text(self.source, "cost source")

    @classmethod
    def known(cls, amount: Decimal, source: str) -> CostComponent:
        return cls(CostKnowledge.KNOWN, amount, source)

    @classmethod
    def estimated(cls, amount: Decimal, source: str) -> CostComponent:
        return cls(CostKnowledge.ESTIMATED, amount, source)

    @classmethod
    def unavailable(cls, source: str | None = None) -> CostComponent:
        return cls(CostKnowledge.UNAVAILABLE, None, source)

    @classmethod
    def not_applicable(cls, source: str) -> CostComponent:
        return cls(CostKnowledge.NOT_APPLICABLE, ZERO, source)


@dataclass(frozen=True, slots=True)
class TradingCostBreakdown:
    commission: CostComponent
    spread: CostComponent
    slippage: CostComponent
    exchange_fees: CostComponent
    transaction_tax: CostComponent
    fx_cost: CostComponent
    financing_cost: CostComponent
    other_variable_cost: CostComponent
    total_variable_cost: CostComponent

    @property
    def components(self) -> tuple[tuple[str, CostComponent], ...]:
        return (
            ("commission", self.commission),
            ("spread", self.spread),
            ("slippage", self.slippage),
            ("exchange_fees", self.exchange_fees),
            ("transaction_tax", self.transaction_tax),
            ("fx_cost", self.fx_cost),
            ("financing_cost", self.financing_cost),
            ("other_variable_cost", self.other_variable_cost),
        )


@dataclass(frozen=True, slots=True)
class OperatingCostBreakdown:
    market_data_subscription: CostComponent
    server_vps: CostComponent
    software_subscriptions: CostComponent
    other_fixed_cost: CostComponent

    @property
    def components(self) -> tuple[tuple[str, CostComponent], ...]:
        return (
            ("market_data_subscription", self.market_data_subscription),
            ("server_vps", self.server_vps),
            ("software_subscriptions", self.software_subscriptions),
            ("other_fixed_cost", self.other_fixed_cost),
        )


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    run_id: str
    timestamp: datetime
    trading: TradingCostBreakdown
    operating: OperatingCostBreakdown
    gross_pnl: Decimal | None
    known_trading_costs: Decimal
    estimated_trading_costs: Decimal
    known_operating_costs: Decimal
    estimated_operating_costs: Decimal
    net_pnl_known: Decimal | None
    net_pnl_estimated: Decimal | None
    coverage_status: CostCoverageStatus
    warnings: tuple[str, ...] = ()
    net_trading_pnl_before_operating: Decimal | None = None
    operating_costs_total: Decimal | None = None
    net_economic_pnl: Decimal | None = None
    tariff_profile_id: str | None = None
    tariff_status: str | None = None

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        _utc(self.timestamp, "timestamp")
        for field_name in (
            "known_trading_costs",
            "estimated_trading_costs",
            "known_operating_costs",
            "estimated_operating_costs",
        ):
            value = getattr(self, field_name)
            if not value.is_finite() or value < ZERO:
                raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MonitoringEvent:
    event_id: str
    timestamp: datetime
    event_type: MonitoringEventType
    run_id: str
    session_id: str
    source_component: str
    component_version: str
    related_ids: tuple[tuple[str, str], ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    payload_json: str = "{}"
    symbol: str | None = None
    strategy_name: str | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "event_id", "run_id", "session_id", "source_component", "component_version"
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        for field_name in ("related_ids", "provenance"):
            values = getattr(self, field_name)
            if values != tuple(sorted(values)):
                raise ValueError(f"{field_name} must be deterministically sorted")
        _json_object(self.payload_json, "payload_json")
        for optional in (self.symbol, self.strategy_name, self.status):
            if optional is not None:
                _text(optional, "optional event field")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


@dataclass(frozen=True, slots=True)
class HealthComponent:
    name: str
    status: SystemStatus
    message: str
    observed_at: datetime

    def __post_init__(self) -> None:
        _text(self.name, "health component name")
        _text(self.message, "health message")
        _utc(self.observed_at, "observed_at")


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    run_id: str
    timestamp: datetime
    status: SystemStatus
    components: tuple[HealthComponent, ...]

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        _utc(self.timestamp, "timestamp")
        names = [item.name for item in self.components]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("health components must have unique sorted names")


@dataclass(frozen=True, slots=True)
class DecisionTraceStep:
    stage: str
    status: SystemStatus
    entity_id: str | None
    reason_codes: tuple[str, ...] = ()
    human_reasons: tuple[str, ...] = ()
    details_json: str = "{}"

    def __post_init__(self) -> None:
        _text(self.stage, "trace stage")
        if self.entity_id is not None:
            _text(self.entity_id, "trace entity_id")
        _json_object(self.details_json, "details_json")

    @property
    def details(self) -> dict[str, Any]:
        return json.loads(self.details_json)


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    trace_id: str
    run_id: str
    timestamp: datetime
    symbol: str
    strategy_name: str | None
    steps: tuple[DecisionTraceStep, ...]

    def __post_init__(self) -> None:
        for field_name in ("trace_id", "run_id", "symbol"):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        if self.strategy_name is not None:
            _text(self.strategy_name, "strategy_name")
        stages = [item.stage for item in self.steps]
        if len(stages) != len(set(stages)):
            raise ValueError("decision trace stages must be unique")


@dataclass(frozen=True, slots=True)
class MonitoringSnapshot:
    snapshot_id: str
    run_id: str
    timestamp: datetime
    mode: str
    status: SystemStatus
    source_schema_version: str
    source_fingerprint: str
    sections_json: str

    def __post_init__(self) -> None:
        for field_name in (
            "snapshot_id", "run_id", "mode", "source_schema_version", "source_fingerprint"
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.timestamp, "timestamp")
        _json_object(self.sections_json, "sections_json")

    @property
    def sections(self) -> dict[str, Any]:
        return json.loads(self.sections_json)
