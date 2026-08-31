"""Offline evidence registry, tariff compatibility, and Paper cost scenarios.

This module is deliberately analytical.  It never changes a tariff used by a
backtest, never scrapes a source at runtime, and never authorizes Paper or LIVE.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.hashing import stable_hash
from trading_ai.robustness.exceptions import RobustnessError


DEFAULT_EVIDENCE_V2_PATH = (
    PROJECT_ROOT / "config" / "robustness" / "evidence_registry_v2.toml"
)
DEFAULT_PAPER_OPERATING_PATH = (
    PROJECT_ROOT / "config" / "robustness" / "paper_operating_v1.toml"
)


class EvidenceSourceType(str, Enum):
    CURRENT_OFFICIAL_SOURCE = "CURRENT_OFFICIAL_SOURCE"
    HISTORICAL_OFFICIAL_SOURCE = "HISTORICAL_OFFICIAL_SOURCE"
    ARCHIVED_OFFICIAL_SOURCE = "ARCHIVED_OFFICIAL_SOURCE"
    REGULATORY_SOURCE = "REGULATORY_SOURCE"
    BROKER_STATEMENT = "BROKER_STATEMENT"
    ESTIMATE = "ESTIMATE"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    ESTIMATED_CONSERVATIVE = "ESTIMATED_CONSERVATIVE"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICTED = "CONFLICTED"


class TariffCompatibilityStatus(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    COMPATIBLE_CONSERVATIVE = "COMPATIBLE_CONSERVATIVE"
    NUMERICALLY_DIFFERENT = "NUMERICALLY_DIFFERENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceReassessmentMode(str, Enum):
    EVIDENCE_ONLY_RECLASSIFICATION = "EVIDENCE_ONLY_RECLASSIFICATION"
    ECONOMIC_RECOMPUTATION_REQUIRED = "ECONOMIC_RECOMPUTATION_REQUIRED"
    DECISION_CORE_CHANGED = "DECISION_CORE_CHANGED"


class InvarianceStatus(str, Enum):
    IDENTICAL = "IDENTICAL"
    DIFFERENT = "DIFFERENT"
    NOT_EVALUATED = "NOT_EVALUATED"


class EconomicEvidenceStatus(str, Enum):
    COMPLETE_VERIFIED = "COMPLETE_VERIFIED"
    COMPLETE_CONSERVATIVE = "COMPLETE_CONSERVATIVE"
    COMPLETE_ESTIMATED = "COMPLETE_ESTIMATED"
    INCOMPLETE = "INCOMPLETE"


_OFFICIAL_DOMAINS = (
    "interactivebrokers.com",
    "sec.gov",
    "finra.org",
    "bofip.impots.gouv.fr",
)


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RobustnessError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RobustnessError(f"{name} must be numeric") from exc
    if not result.is_finite():
        raise RobustnessError(f"{name} must be finite")
    return result


def _official_host(reference: str) -> bool:
    host = (urlparse(reference).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in _OFFICIAL_DOMAINS)


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise RobustnessError(f"{name} must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    provider: str
    market: str
    fee_type: str
    plan: str
    source_type: EvidenceSourceType
    verification_status: EvidenceVerificationStatus
    effective_from: datetime
    effective_to: datetime
    retrieved_at: datetime
    source_reference: str
    original_source_reference: str | None
    archive_timestamp: datetime | None
    archive_digest: str | None
    supporting_references: tuple[str, ...]
    normalized_rules: tuple[tuple[str, str], ...]
    notes: str
    evidence_hash: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.evidence_id, "evidence_id"),
            (self.provider, "provider"),
            (self.market, "market"),
            (self.fee_type, "fee_type"),
            (self.plan, "plan"),
            (self.source_reference, "source_reference"),
            (self.notes, "notes"),
        ):
            if not value.strip():
                raise RobustnessError(f"{name} must not be empty")
        _utc(self.effective_from, "evidence effective_from")
        _utc(self.effective_to, "evidence effective_to")
        _utc(self.retrieved_at, "evidence retrieved_at")
        if self.effective_from >= self.effective_to:
            raise RobustnessError("evidence period must be positive")
        _sha(self.evidence_hash, "evidence_hash")
        if self.normalized_rules != tuple(sorted(self.normalized_rules)):
            raise RobustnessError("normalized evidence rules must be sorted")
        if len({name for name, _ in self.normalized_rules}) != len(self.normalized_rules):
            raise RobustnessError("normalized evidence rule names must be unique")
        if self.source_type is EvidenceSourceType.ARCHIVED_OFFICIAL_SOURCE:
            if (urlparse(self.source_reference).hostname or "").lower() != "web.archive.org":
                raise RobustnessError("archived official evidence must use web.archive.org")
            if not self.original_source_reference or not _official_host(
                self.original_source_reference
            ):
                raise RobustnessError("archive must identify its original official URL")
            if self.archive_timestamp is None or not self.archive_digest:
                raise RobustnessError("archive evidence requires timestamp and digest")
            _utc(self.archive_timestamp, "archive_timestamp")
        elif self.source_type in {
            EvidenceSourceType.CURRENT_OFFICIAL_SOURCE,
            EvidenceSourceType.HISTORICAL_OFFICIAL_SOURCE,
            EvidenceSourceType.REGULATORY_SOURCE,
        } and not _official_host(self.source_reference):
            raise RobustnessError("official evidence source domain is not approved")
        if self.source_type is EvidenceSourceType.CURRENT_OFFICIAL_SOURCE and self.archive_timestamp:
            raise RobustnessError("current evidence cannot carry an archive timestamp")

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return self.effective_from < end and start < self.effective_to


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    fee_type: str
    market: str
    plan: str
    evidence_ids: tuple[str, str]
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceRegistryV2:
    registry_version: str
    acquired_at: datetime
    records: tuple[EvidenceRecord, ...]
    conflicts: tuple[EvidenceConflict, ...]
    registry_hash: str

    def __post_init__(self) -> None:
        if self.registry_version != "2.0":
            raise RobustnessError("unsupported evidence registry version")
        _utc(self.acquired_at, "registry acquired_at")
        _sha(self.registry_hash, "registry_hash")
        if tuple(item.evidence_id for item in self.records) != tuple(
            sorted(item.evidence_id for item in self.records)
        ):
            raise RobustnessError("evidence records must be deterministically sorted")
        if len({item.evidence_id for item in self.records}) != len(self.records):
            raise RobustnessError("duplicate evidence_id")

    def records_for(
        self, *, fee_type: str, market: str, plan: str
    ) -> tuple[EvidenceRecord, ...]:
        return tuple(
            item
            for item in self.records
            if item.fee_type == fee_type
            and item.market == market
            and item.plan == plan
        )


@dataclass(frozen=True, slots=True)
class AppliedTariffAssumption:
    provider: str
    market: str
    fee_type: str
    plan: str
    normalized_rules: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class TariffCompatibilityAssessment:
    fee_type: str
    market: str
    plan: str
    period_start: datetime
    period_end: datetime
    status: TariffCompatibilityStatus
    applied_rules_hash: str
    evidence_rule_hashes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    period_coverage_complete: bool
    mathematical_demonstration: str
    warnings: tuple[str, ...]


def _record_semantic(raw: dict[str, Any]) -> dict[str, Any]:
    """Exclude acquisition time so repeated offline loading is deterministic."""

    return {key: value for key, value in raw.items() if key != "retrieved_at"}


def _conflicts(records: tuple[EvidenceRecord, ...]) -> tuple[EvidenceConflict, ...]:
    result: list[EvidenceConflict] = []
    for index, left in enumerate(records):
        if left.verification_status is not EvidenceVerificationStatus.VERIFIED:
            continue
        for right in records[index + 1 :]:
            same_scope = (
                left.provider == right.provider
                and left.market == right.market
                and left.fee_type == right.fee_type
                and left.plan == right.plan
            )
            if (
                same_scope
                and left.overlaps(right.effective_from, right.effective_to)
                and left.normalized_rules != right.normalized_rules
                and right.verification_status is EvidenceVerificationStatus.VERIFIED
            ):
                result.append(
                    EvidenceConflict(
                        fee_type=left.fee_type,
                        market=left.market,
                        plan=left.plan,
                        evidence_ids=tuple(sorted((left.evidence_id, right.evidence_id))),
                        reason="CONFLICTING_OFFICIAL_RULES_WITH_OVERLAPPING_SCOPE",
                    )
                )
    return tuple(sorted(result, key=lambda item: item.evidence_ids))


def load_evidence_registry_v2(
    path: Path = DEFAULT_EVIDENCE_V2_PATH,
) -> EvidenceRegistryV2:
    """Load normalized evidence without network access or source mutation."""

    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        records: list[EvidenceRecord] = []
        for item in raw.get("evidence", ()):
            semantic = _record_semantic(dict(item))
            rules = tuple(sorted((str(key), str(value)) for key, value in item.get("rules", {}).items()))
            records.append(
                EvidenceRecord(
                    evidence_id=str(item["evidence_id"]),
                    provider=str(item["provider"]),
                    market=str(item["market"]),
                    fee_type=str(item["fee_type"]),
                    plan=str(item["plan"]),
                    source_type=EvidenceSourceType(str(item["source_type"])),
                    verification_status=EvidenceVerificationStatus(
                        str(item["verification_status"])
                    ),
                    effective_from=_utc(item["effective_from"], "effective_from"),
                    effective_to=_utc(item["effective_to"], "effective_to"),
                    retrieved_at=_utc(item["retrieved_at"], "retrieved_at"),
                    source_reference=str(item["source_reference"]),
                    original_source_reference=(
                        str(item["original_source_reference"])
                        if item.get("original_source_reference")
                        else None
                    ),
                    archive_timestamp=(
                        _utc(item["archive_timestamp"], "archive_timestamp")
                        if item.get("archive_timestamp")
                        else None
                    ),
                    archive_digest=(
                        str(item["archive_digest"])
                        if item.get("archive_digest")
                        else None
                    ),
                    supporting_references=tuple(
                        sorted(str(value) for value in item.get("supporting_references", ()))
                    ),
                    normalized_rules=rules,
                    notes=str(item["notes"]),
                    evidence_hash=stable_hash(
                        {**semantic, "rules": rules}
                    ),
                )
            )
        ordered = tuple(sorted(records, key=lambda item: item.evidence_id))
        conflicts = _conflicts(ordered)
        registry_semantic = {
            "registry_version": str(raw["registry_version"]),
            "records": tuple(
                {
                    "evidence_id": item.evidence_id,
                    "evidence_hash": item.evidence_hash,
                }
                for item in ordered
            ),
            "conflicts": conflicts,
        }
        return EvidenceRegistryV2(
            registry_version=str(raw["registry_version"]),
            acquired_at=_utc(raw["acquired_at"], "registry acquired_at"),
            records=ordered,
            conflicts=conflicts,
            registry_hash=stable_hash(registry_semantic),
        )
    except RobustnessError:
        raise
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RobustnessError(f"invalid evidence registry V2: {exc}") from exc


def _coverage_complete(
    records: Iterable[EvidenceRecord], start: datetime, end: datetime
) -> bool:
    cursor = start
    for item in sorted(records, key=lambda record: (record.effective_from, record.effective_to)):
        if item.effective_to <= cursor:
            continue
        if item.effective_from > cursor:
            return False
        cursor = max(cursor, item.effective_to)
        if cursor >= end:
            return True
    return cursor >= end


def _rule_decimals(rules: tuple[tuple[str, str], ...]) -> dict[str, Decimal]:
    numeric = {
        "fixed_per_order",
        "per_unit",
        "proportional_bps",
        "minimum_per_order",
        "maximum_notional_fraction",
    }
    result: dict[str, Decimal] = {}
    for name, value in rules:
        if name in numeric:
            result[name] = _decimal(value, name)
    return result


def _conservative(applied: dict[str, Decimal], evidenced: dict[str, Decimal]) -> bool:
    lower_bound_fields = (
        "fixed_per_order",
        "per_unit",
        "proportional_bps",
        "minimum_per_order",
    )
    for name in lower_bound_fields:
        if applied.get(name, Decimal("0")) < evidenced.get(name, Decimal("0")):
            return False
    applied_cap = applied.get("maximum_notional_fraction")
    evidence_cap = evidenced.get("maximum_notional_fraction")
    if evidence_cap is None:
        return applied_cap is None
    return applied_cap is None or applied_cap >= evidence_cap


class TariffEvidenceComparator:
    """Compare frozen numeric assumptions with dated normalized evidence."""

    @staticmethod
    def compare(
        assumption: AppliedTariffAssumption,
        registry: EvidenceRegistryV2,
        *,
        period_start: datetime,
        period_end: datetime,
    ) -> TariffCompatibilityAssessment:
        start = _utc(period_start, "comparison period_start")
        end = _utc(period_end, "comparison period_end")
        scoped = registry.records_for(
            fee_type=assumption.fee_type,
            market=assumption.market,
            plan=assumption.plan,
        )
        scoped_conflicts = tuple(
            item
            for item in registry.conflicts
            if item.fee_type == assumption.fee_type
            and item.market == assumption.market
            and item.plan == assumption.plan
        )
        historical = tuple(
            item
            for item in scoped
            if item.source_type
            not in {
                EvidenceSourceType.CURRENT_OFFICIAL_SOURCE,
                EvidenceSourceType.ESTIMATE,
                EvidenceSourceType.UNAVAILABLE,
            }
            and item.verification_status is EvidenceVerificationStatus.VERIFIED
            and item.overlaps(start, end)
        )
        complete = _coverage_complete(historical, start, end)
        warnings: list[str] = []
        if scoped_conflicts:
            warnings.append("CONFLICTING_OFFICIAL_EVIDENCE")
        if not complete:
            warnings.append("HISTORICAL_PERIOD_EVIDENCE_GAP")
        applied_hash = stable_hash(assumption.normalized_rules)
        evidence_hashes = tuple(sorted({stable_hash(item.normalized_rules) for item in historical}))
        evidence_ids = tuple(sorted(item.evidence_id for item in historical))
        if scoped_conflicts or not historical or not complete:
            status = TariffCompatibilityStatus.INSUFFICIENT_EVIDENCE
            demonstration = "Dated historical evidence does not cover the entire tested interval."
        elif all(item.normalized_rules == assumption.normalized_rules for item in historical):
            status = TariffCompatibilityStatus.EXACT_MATCH
            demonstration = (
                "Every dated historical interval has the same normalized fixed, per-unit, "
                "minimum, proportional, and cap rules as the frozen tariff."
            )
        else:
            applied = _rule_decimals(assumption.normalized_rules)
            evidence_numeric = [_rule_decimals(item.normalized_rules) for item in historical]
            if evidence_numeric and all(_conservative(applied, item) for item in evidence_numeric):
                status = TariffCompatibilityStatus.COMPATIBLE_CONSERVATIVE
                demonstration = (
                    "For every covered interval, the applied additive rates and minimum are "
                    "not lower and its cap is not tighter; the modeled commission cannot be "
                    "below the evidenced commission for positive quantity/notional."
                )
            else:
                status = TariffCompatibilityStatus.NUMERICALLY_DIFFERENT
                demonstration = (
                    "At least one normalized rate, minimum, or cap differs without a valid "
                    "component-wise conservative dominance proof."
                )
        return TariffCompatibilityAssessment(
            fee_type=assumption.fee_type,
            market=assumption.market,
            plan=assumption.plan,
            period_start=start,
            period_end=end,
            status=status,
            applied_rules_hash=applied_hash,
            evidence_rule_hashes=evidence_hashes,
            evidence_ids=evidence_ids,
            period_coverage_complete=complete,
            mathematical_demonstration=demonstration,
            warnings=tuple(warnings),
        )


@dataclass(frozen=True, slots=True)
class OperatingCostRange:
    component: str
    status: EvidenceVerificationStatus
    currency: str
    monthly_low: Decimal | None
    monthly_central: Decimal | None
    monthly_high: Decimal | None
    source_type: EvidenceSourceType
    source_reference: str
    notes: str

    def __post_init__(self) -> None:
        if self.status is EvidenceVerificationStatus.UNAVAILABLE:
            if any(value is not None for value in (self.monthly_low, self.monthly_central, self.monthly_high)):
                raise RobustnessError("UNAVAILABLE operating component cannot have an amount")
            return
        values = (self.monthly_low, self.monthly_central, self.monthly_high)
        if any(value is None for value in values):
            raise RobustnessError("estimated operating ranges require low/central/high")
        assert all(value is not None for value in values)
        if not (Decimal("0") <= values[0] <= values[1] <= values[2]):
            raise RobustnessError("operating range must be non-negative and ordered")


@dataclass(frozen=True, slots=True)
class PaperOperatingScenario:
    scenario_id: str
    scenario_version: str
    deployment_mode: str
    components: tuple[OperatingCostRange, ...]
    scenario_hash: str

    @property
    def monthly_totals(self) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        if any(item.status is EvidenceVerificationStatus.UNAVAILABLE for item in self.components):
            return None, None, None
        return tuple(
            sum((getattr(item, field) or Decimal("0")) for item in self.components)
            for field in ("monthly_low", "monthly_central", "monthly_high")
        )  # type: ignore[return-value]


def load_paper_operating_scenarios(
    path: Path = DEFAULT_PAPER_OPERATING_PATH,
) -> tuple[PaperOperatingScenario, ...]:
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        scenarios: list[PaperOperatingScenario] = []
        version = str(raw["version"])
        for name, values in sorted(raw.get("scenarios", {}).items()):
            components: list[OperatingCostRange] = []
            for component, item in sorted(values.get("components", {}).items()):
                status = EvidenceVerificationStatus(str(item["status"]))
                components.append(
                    OperatingCostRange(
                        component=str(component),
                        status=status,
                        currency=str(item["currency"]),
                        monthly_low=(
                            _decimal(item["monthly_low"], "monthly_low")
                            if "monthly_low" in item
                            else None
                        ),
                        monthly_central=(
                            _decimal(item["monthly_central"], "monthly_central")
                            if "monthly_central" in item
                            else None
                        ),
                        monthly_high=(
                            _decimal(item["monthly_high"], "monthly_high")
                            if "monthly_high" in item
                            else None
                        ),
                        source_type=EvidenceSourceType(str(item["source_type"])),
                        source_reference=str(item["source_reference"]),
                        notes=str(item["notes"]),
                    )
                )
            semantic = {
                "scenario_id": str(name),
                "version": version,
                "deployment_mode": str(values["deployment_mode"]),
                "components": tuple(components),
            }
            scenarios.append(
                PaperOperatingScenario(
                    scenario_id=str(name),
                    scenario_version=version,
                    deployment_mode=str(values["deployment_mode"]),
                    components=tuple(components),
                    scenario_hash=stable_hash(semantic),
                )
            )
        return tuple(scenarios)
    except RobustnessError:
        raise
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RobustnessError(f"invalid Paper operating scenarios: {exc}") from exc
