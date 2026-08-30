"""Immutable research-validation reports that never unlock execution modes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum


def _text(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    BLOCKED_EXTERNAL_DATA = "BLOCKED_EXTERNAL_DATA"


class CriterionStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ValidationCriterion:
    name: str
    status: CriterionStatus
    observed: str
    required: str
    reason: str

    def __post_init__(self) -> None:
        for field_name in ("name", "observed", "required", "reason"):
            _text(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class CostStressResult:
    multiplier: Decimal
    stressed_variable_costs: Decimal | None
    stressed_net_pnl: Decimal | None
    stressed_net_return: float | None
    status: CriterionStatus


@dataclass(frozen=True, slots=True)
class SubperiodResult:
    index: int
    start: datetime
    end: datetime
    net_return: float
    closed_trades: int

    def __post_init__(self) -> None:
        _utc(self.start, "subperiod start")
        _utc(self.end, "subperiod end")


@dataclass(frozen=True, slots=True)
class SymbolRobustnessResult:
    symbol: str
    closed_trades: int
    net_pnl: Decimal
    share_of_positive_pnl: float | None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    validation_id: str
    run_id: str
    created_at: datetime
    gate_name: str
    gate_version: str
    config_hash: str
    status: ValidationStatus
    implementation_status: str
    real_data_campaign_status: ValidationStatus
    synthetic_mechanics_only: bool
    final_oos: bool
    dataset_ids: tuple[str, ...]
    dataset_checksums: tuple[str, ...]
    tariff_profile_id: str | None
    tariff_status: str | None
    tariff_period_verified: bool
    cost_coverage: str
    criteria: tuple[ValidationCriterion, ...]
    stress_results: tuple[CostStressResult, ...]
    subperiods: tuple[SubperiodResult, ...]
    symbols: tuple[SymbolRobustnessResult, ...]
    warnings: tuple[str, ...]
    survivorship_bias_warning: str = "SURVIVORSHIP_BIAS_NOT_RESOLVED"
    unlocks_paper_or_live: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "validation_id", "run_id", "gate_name", "gate_version", "config_hash",
            "implementation_status", "cost_coverage", "survivorship_bias_warning",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.created_at, "created_at")
        if self.unlocks_paper_or_live:
            raise ValueError("validation reports must never unlock PAPER or LIVE")
        if len(self.config_hash) != 64:
            raise ValueError("validation config_hash must be SHA-256")
