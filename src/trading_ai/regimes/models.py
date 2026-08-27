"""Immutable models for two-axis regimes and strategy eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


ZERO = Decimal("0")
ONE = Decimal("1")


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{field_name} must be a SHA-256 digest")


class StructureRegime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class VolatilityRegime(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class ActivationStatus(str, Enum):
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """Two independent regime dimensions known at exactly one bar."""

    snapshot_id: str
    symbol: str
    timestamp: datetime
    timeframe: str
    structure_regime: StructureRegime
    volatility_regime: VolatilityRegime
    detector_name: str
    detector_version: str
    config_hash: str
    bars_in_current_structure_regime: int
    evidence: tuple[tuple[str, str], ...]
    reason_codes: tuple[str, ...]
    candidate_structure_regime: StructureRegime = StructureRegime.UNKNOWN
    confirmation_progress: int = 0
    transition_from: StructureRegime | None = None
    transition_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "snapshot_id",
            "symbol",
            "timeframe",
            "detector_name",
            "detector_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_aware(self.timestamp, "timestamp")
        _require_sha256(self.config_hash, "config_hash")
        if self.bars_in_current_structure_regime < 1:
            raise ValueError("bars_in_current_structure_regime must be positive")
        if self.confirmation_progress < 0:
            raise ValueError("confirmation_progress must not be negative")
        if tuple(sorted(self.evidence)) != self.evidence:
            raise ValueError("regime evidence must be deterministically sorted")
        evidence_names = [name for name, _ in self.evidence]
        if len(evidence_names) != len(set(evidence_names)):
            raise ValueError("regime evidence names must be unique")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("regime reason_codes must be sorted and unique")
        if any(not code.strip() for code in self.reason_codes):
            raise ValueError("regime reason_codes must not be empty")
        if self.transition_from is None and self.transition_reason is not None:
            raise ValueError("transition_reason requires transition_from")
        if self.transition_reason is not None:
            _require_text(self.transition_reason, "transition_reason")


@dataclass(frozen=True, slots=True)
class RegimeTransition:
    transition_id: str
    symbol: str
    timestamp: datetime
    timeframe: str
    from_structure: StructureRegime
    to_structure: StructureRegime
    from_volatility: VolatilityRegime
    to_volatility: VolatilityRegime
    reason: str

    def __post_init__(self) -> None:
        for field_name in ("transition_id", "symbol", "timeframe", "reason"):
            _require_text(getattr(self, field_name), field_name)
        _require_aware(self.timestamp, "timestamp")
        if (
            self.from_structure is self.to_structure
            and self.from_volatility is self.to_volatility
        ):
            raise ValueError("a RegimeTransition must change at least one axis")


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    """Regime eligibility decision; never a risk or execution authorization."""

    decision_id: str
    timestamp: datetime
    symbol: str
    strategy_name: str
    strategy_version: str
    signal_id: str
    regime_snapshot_id: str
    structure_regime: StructureRegime
    volatility_regime: VolatilityRegime
    status: ActivationStatus
    allocation_multiplier: Decimal
    proposed_quantity: Decimal
    adjusted_quantity: Decimal
    reason_codes: tuple[str, ...]
    human_readable_reasons: tuple[str, ...]
    policy_name: str
    policy_version: str
    policy_config_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id",
            "symbol",
            "strategy_name",
            "strategy_version",
            "signal_id",
            "regime_snapshot_id",
            "policy_name",
            "policy_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_aware(self.timestamp, "timestamp")
        _require_sha256(self.policy_config_hash, "policy_config_hash")
        if not ZERO <= self.allocation_multiplier <= ONE:
            raise ValueError("allocation_multiplier must be between 0 and 1")
        if self.proposed_quantity <= ZERO or not self.proposed_quantity.is_finite():
            raise ValueError("proposed_quantity must be positive and finite")
        if (
            self.adjusted_quantity < ZERO
            or not self.adjusted_quantity.is_finite()
            or self.adjusted_quantity > self.proposed_quantity
        ):
            raise ValueError(
                "adjusted_quantity must be non-negative and never exceed the proposal"
            )
        if self.adjusted_quantity != self.proposed_quantity * self.allocation_multiplier:
            raise ValueError("adjusted_quantity must equal proposal times multiplier")
        if self.status is ActivationStatus.ALLOW and self.allocation_multiplier != ONE:
            raise ValueError("ALLOW requires allocation_multiplier = 1")
        if self.status is ActivationStatus.REDUCE and not (
            ZERO < self.allocation_multiplier < ONE
        ):
            raise ValueError("REDUCE requires a multiplier strictly between 0 and 1")
        if self.status is ActivationStatus.BLOCK and self.allocation_multiplier != ZERO:
            raise ValueError("BLOCK requires allocation_multiplier = 0")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("activation reason_codes must be sorted and unique")
        if any(not reason.strip() for reason in self.human_readable_reasons):
            raise ValueError("activation reasons must not be empty")


@dataclass(frozen=True, slots=True)
class RegimeReport:
    bars_by_structure_regime: tuple[tuple[str, int], ...]
    bars_by_volatility_regime: tuple[tuple[str, int], ...]
    transition_count: int
    signals_by_regime: tuple[tuple[str, int], ...]
    activation_allow: int
    activation_reduce: int
    activation_block: int

    def __post_init__(self) -> None:
        for field_name in (
            "bars_by_structure_regime",
            "bars_by_volatility_regime",
            "signals_by_regime",
        ):
            values = getattr(self, field_name)
            if tuple(sorted(values)) != values:
                raise ValueError(f"{field_name} must be deterministically sorted")
            if any(count < 0 for _, count in values):
                raise ValueError(f"{field_name} counts must not be negative")
        for field_name in (
            "transition_count",
            "activation_allow",
            "activation_reduce",
            "activation_block",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")
