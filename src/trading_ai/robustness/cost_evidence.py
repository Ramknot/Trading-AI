"""Dated official cost evidence for research diagnostics, never execution rates."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.hashing import stable_hash
from trading_ai.robustness.exceptions import RobustnessError


DEFAULT_EVIDENCE_PATH = (
    PROJECT_ROOT / "config" / "robustness" / "historical_cost_evidence.toml"
)


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    HISTORICAL_TARIFF_UNVERIFIED = "HISTORICAL_TARIFF_UNVERIFIED"
    ESTIMATED = "ESTIMATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceKind(str, Enum):
    ARCHIVED_OFFICIAL_SOURCE = "ARCHIVED_OFFICIAL_SOURCE"
    CURRENT_OFFICIAL_SOURCE = "CURRENT_OFFICIAL_SOURCE"


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RobustnessError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class DatedCostEvidence:
    evidence_id: str
    subject: str
    period_start: datetime
    period_end: datetime
    status: EvidenceStatus
    evidence_kind: EvidenceKind
    source_name: str
    source_reference: str
    evidence_hash: str
    warning: str | None = None

    def covers(self, timestamp: datetime) -> bool:
        return self.period_start <= timestamp.astimezone(timezone.utc) < self.period_end


@dataclass(frozen=True, slots=True)
class HistoricalTaxRate:
    rule_id: str
    rate_bps: Decimal
    period_start: datetime
    period_end: datetime
    status: EvidenceStatus
    evidence_kind: EvidenceKind
    source_name: str
    source_reference: str
    evidence_hash: str

    def covers(self, timestamp: datetime) -> bool:
        return self.period_start <= timestamp.astimezone(timezone.utc) < self.period_end


@dataclass(frozen=True, slots=True)
class HistoricalTaxEligibility:
    symbol: str
    issuer: str
    period_start: datetime
    period_end: datetime
    eligible: bool
    status: EvidenceStatus
    evidence_kind: EvidenceKind
    source_reference: str
    evidence_hash: str

    def covers(self, timestamp: datetime) -> bool:
        return self.period_start <= timestamp.astimezone(timezone.utc) < self.period_end


@dataclass(frozen=True, slots=True)
class OperatingCostEvidence:
    component: str
    status: EvidenceStatus
    amount: Decimal | None
    source_reference: str

    def __post_init__(self) -> None:
        if self.status in {EvidenceStatus.UNAVAILABLE, EvidenceStatus.ESTIMATED} and self.amount is None:
            return
        if self.status is EvidenceStatus.NOT_APPLICABLE and self.amount != Decimal("0"):
            raise RobustnessError("NOT_APPLICABLE operating cost requires explicit zero")
        if self.amount is None or self.amount < 0:
            raise RobustnessError("known operating cost requires a non-negative amount")


@dataclass(frozen=True, slots=True)
class HistoricalCostEvidenceRegistry:
    version: str
    verified_at: datetime
    registry_hash: str
    broker_tariffs: tuple[DatedCostEvidence, ...]
    tax_rates: tuple[HistoricalTaxRate, ...]
    tax_eligibility: tuple[HistoricalTaxEligibility, ...]
    exchange_fee_status: EvidenceStatus
    fx_cost_status: EvidenceStatus
    operating_scenarios: tuple[tuple[str, tuple[OperatingCostEvidence, ...]], ...]

    def tax_rate_at(self, timestamp: datetime) -> HistoricalTaxRate | None:
        matches = [item for item in self.tax_rates if item.covers(timestamp)]
        if len(matches) > 1:
            raise RobustnessError("overlapping historical tax rate schedules")
        return matches[0] if matches else None

    def eligibility_at(
        self, symbol: str, timestamp: datetime
    ) -> HistoricalTaxEligibility | None:
        matches = [
            item for item in self.tax_eligibility
            if item.symbol == symbol and item.covers(timestamp)
        ]
        if len(matches) > 1:
            raise RobustnessError("overlapping tax eligibility schedules")
        return matches[0] if matches else None


def _component(name: str, raw: dict[str, Any]) -> OperatingCostEvidence:
    return OperatingCostEvidence(
        component=name,
        status=EvidenceStatus(str(raw["status"])),
        amount=Decimal(str(raw["amount"])) if "amount" in raw else None,
        source_reference=str(raw["source_reference"]),
    )


def load_historical_cost_evidence(
    path: Path = DEFAULT_EVIDENCE_PATH,
) -> HistoricalCostEvidenceRegistry:
    """Load static source records; this function never contacts the network."""

    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        tariffs = []
        for item in raw.get("broker_tariffs", ()):
            semantic = dict(item)
            tariffs.append(
                DatedCostEvidence(
                    evidence_id=f"tariff-evidence-{stable_hash(semantic)[:20]}",
                    subject=str(item["profile_id"]),
                    period_start=_utc(item["period_start"], "tariff period_start"),
                    period_end=_utc(item["period_end"], "tariff period_end"),
                    status=EvidenceStatus(str(item["status"])),
                    evidence_kind=EvidenceKind(str(item["evidence_kind"])),
                    source_name=str(item["source_name"]),
                    source_reference=str(item["source_reference"]),
                    evidence_hash=stable_hash(semantic),
                    warning=str(item["warning"]) if item.get("warning") else None,
                )
            )
        rates = []
        for item in raw.get("tax_rates", ()):
            semantic = dict(item)
            rates.append(
                HistoricalTaxRate(
                    rule_id=str(item["rule_id"]),
                    rate_bps=Decimal(str(item["rate_bps"])),
                    period_start=_utc(item["period_start"], "tax period_start"),
                    period_end=_utc(item["period_end"], "tax period_end"),
                    status=EvidenceStatus(str(item["status"])),
                    evidence_kind=EvidenceKind(str(item["evidence_kind"])),
                    source_name=str(item["source_name"]),
                    source_reference=str(item["source_reference"]),
                    evidence_hash=stable_hash(semantic),
                )
            )
        eligibility = []
        for item in raw.get("tax_eligibility", ()):
            semantic = dict(item)
            eligibility.append(
                HistoricalTaxEligibility(
                    symbol=str(item["symbol"]),
                    issuer=str(item["issuer"]),
                    period_start=_utc(item["period_start"], "eligibility period_start"),
                    period_end=_utc(item["period_end"], "eligibility period_end"),
                    eligible=bool(item["eligible"]),
                    status=EvidenceStatus(str(item["status"])),
                    evidence_kind=EvidenceKind(str(item["evidence_kind"])),
                    source_reference=str(item["source_reference"]),
                    evidence_hash=stable_hash(semantic),
                )
            )
        scenarios = []
        for name, values in sorted(raw.get("operating_scenarios", {}).items()):
            scenarios.append(
                (
                    str(name),
                    tuple(
                        _component(component, values[component])
                        for component in sorted(values)
                    ),
                )
            )
        registry = HistoricalCostEvidenceRegistry(
            version=str(raw["version"]),
            verified_at=_utc(raw["verified_at"], "evidence verified_at"),
            registry_hash=stable_hash(raw),
            broker_tariffs=tuple(tariffs),
            tax_rates=tuple(rates),
            tax_eligibility=tuple(eligibility),
            exchange_fee_status=EvidenceStatus(str(raw["exchange_fees"]["status"])),
            fx_cost_status=EvidenceStatus(str(raw["fx_cost"]["status"])),
            operating_scenarios=tuple(scenarios),
        )
        # Force overlap validation at every declared boundary without looking at
        # market data or a future holdout.
        for rate in registry.tax_rates:
            registry.tax_rate_at(rate.period_start)
        for entry in registry.tax_eligibility:
            registry.eligibility_at(entry.symbol, entry.period_start)
        return registry
    except RobustnessError:
        raise
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RobustnessError(f"invalid historical cost evidence: {exc}") from exc
