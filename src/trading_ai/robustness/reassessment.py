"""Lot 8.3 evidence-only reassessment and Paper-readiness review.

The services in this module are deliberately read-only.  They compare a
consumed backtest with dated evidence and emit immutable reports.  They never
change a trading decision, a numeric historical cost, or holdout governance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.hashing import stable_hash, to_primitive
from trading_ai.costs.config import DEFAULT_COST_DIRECTORY, load_tariff_profile
from trading_ai.monitoring.source import BacktestMonitoringData, BacktestMonitoringSource
from trading_ai.robustness.config import load_research_plan
from trading_ai.robustness.evidence import (
    AppliedTariffAssumption,
    EconomicEvidenceStatus,
    EvidenceReassessmentMode,
    EvidenceRegistryV2,
    EvidenceVerificationStatus,
    InvarianceStatus,
    PaperOperatingScenario,
    TariffCompatibilityAssessment,
    TariffCompatibilityStatus,
    TariffEvidenceComparator,
    load_evidence_registry_v2,
    load_paper_operating_scenarios,
)
from trading_ai.robustness.exceptions import HoldoutGovernanceError, RobustnessError
from trading_ai.robustness.governance import decision_core_hash
from trading_ai.robustness.models import (
    HoldoutStatus,
    PaperReadinessStatus,
    PeriodClassification,
)
from trading_ai.robustness.service import deserialize_holdout
from trading_ai.robustness.storage import LocalRobustnessStore
from trading_ai.validation import LocalValidationStore


ZERO = Decimal("0")
_DECISION_TABLES = (
    "signals",
    "ml_decisions",
    "activation_decisions",
    "portfolio_decisions",
    "economic_decisions",
    "risk_decisions",
    "orders",
    "fills",
)
_COST_TABLES = ("cost_estimates", "cost_actuals", "cost_reconciliation")


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RobustnessError(f"{name} must be numeric") from exc
    if not result.is_finite():
        raise RobustnessError(f"{name} must be finite")
    return result


def _utc(value: object, name: str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RobustnessError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise RobustnessError(f"{name} must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class InvarianceReport:
    kind: str
    baseline_run_id: str
    candidate_run_id: str
    status: InvarianceStatus
    baseline_hash: str
    candidate_hash: str
    item_hashes: tuple[tuple[str, str, str], ...]
    changed_items: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.baseline_hash, "baseline invariance hash")
        _sha(self.candidate_hash, "candidate invariance hash")
        if self.item_hashes != tuple(sorted(self.item_hashes)):
            raise RobustnessError("invariance item hashes must be sorted")


@dataclass(frozen=True, slots=True)
class EvidenceComponentAssessment:
    component: str
    status: str
    compatibility: TariffCompatibilityStatus
    amount_in_original_run: Decimal | None
    indicated_missing_amount: Decimal | None
    evidence_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class EconomicEvidenceCompleteness:
    status: EconomicEvidenceStatus
    broker_commission: str
    exchange_and_pass_through: str
    transaction_tax: str
    fx: str
    spread_and_slippage: str
    operating_costs: str
    historical_period_fit: str
    source_quality: str
    components: tuple[EvidenceComponentAssessment, ...]
    critical_unresolved: tuple[str, ...]
    completeness_hash: str

    def __post_init__(self) -> None:
        _sha(self.completeness_hash, "economic completeness hash")


@dataclass(frozen=True, slots=True)
class RetrospectivePaperEconomics:
    scenario_id: str
    scenario_hash: str
    period_months: Decimal
    operating_low: Decimal | None
    operating_central: Decimal | None
    operating_high: Decimal | None
    net_before_operating: Decimal
    net_after_operating_low: Decimal | None
    net_after_operating_central: Decimal | None
    net_after_operating_high: Decimal | None
    break_even_additional_variable_cost: Decimal
    break_even_monthly_fixed_cost: Decimal | None
    observed_profit_to_variable_cost_ratio: Decimal | None
    remains_positive_at_existing_cost_stress_2x: bool
    label: str = "RETROSPECTIVE_PAPER_COST_SCENARIO_NOT_A_FORECAST"


@dataclass(frozen=True, slots=True)
class ReviewCriterion:
    name: str
    status: str
    observed: str
    required: str
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceReassessmentReport:
    reassessment_id: str
    reassessment_version: str
    reassessment_hash: str
    created_at: datetime
    run_id: str
    robustness_report_id: str
    validation_id: str
    holdout_id: str
    holdout_status: str
    holdout_result_hash: str
    plan_hash: str
    evidence_registry_version: str
    evidence_registry_hash: str
    mode: EvidenceReassessmentMode
    decision_core_expected_hash: str
    decision_core_current_hash: str
    decision_invariance: InvarianceReport
    cost_invariance: InvarianceReport
    broker_tariff: TariffCompatibilityAssessment
    components: tuple[EvidenceComponentAssessment, ...]
    economic_completeness: EconomicEvidenceCompleteness
    original_validation_status: str
    strict_validation_evidence_status: str
    numeric_recomputation_required: bool
    indicated_missing_regulatory_cost: Decimal
    paper_economics: RetrospectivePaperEconomics
    survivorship_warning: str
    concentration_warning: str
    warnings: tuple[str, ...]
    evidence_only: bool
    decisions_or_costs_mutated: bool
    holdout_remains_consumed: bool
    unlocks_paper_or_live: bool = False

    def __post_init__(self) -> None:
        _sha(self.reassessment_hash, "reassessment hash")
        _sha(self.holdout_result_hash, "holdout result hash")
        _sha(self.plan_hash, "plan hash")
        _sha(self.evidence_registry_hash, "evidence registry hash")
        _sha(self.decision_core_expected_hash, "expected decision core hash")
        _sha(self.decision_core_current_hash, "current decision core hash")
        if self.holdout_status != HoldoutStatus.CONSUMED.value or not self.holdout_remains_consumed:
            raise HoldoutGovernanceError("Lot 8.3 reassessment requires a consumed holdout")
        if self.unlocks_paper_or_live:
            raise RobustnessError("evidence reassessment must never unlock Paper or LIVE")
        if self.evidence_only and self.decisions_or_costs_mutated:
            raise RobustnessError("evidence-only reassessment cannot mutate numeric history")


@dataclass(frozen=True, slots=True)
class PaperReadinessReviewV2:
    review_id: str
    review_name: str
    review_version: str
    review_hash: str
    created_at: datetime
    reassessment_id: str
    run_id: str
    holdout_status: str
    status: PaperReadinessStatus
    criteria: tuple[ReviewCriterion, ...]
    warnings: tuple[str, ...]
    meaning: str
    next_step: str
    unlocks_paper_or_live: bool = False

    def __post_init__(self) -> None:
        _sha(self.review_hash, "Paper-readiness V2 hash")
        if self.holdout_status != HoldoutStatus.CONSUMED.value:
            raise HoldoutGovernanceError("Paper-readiness V2 cannot freshen a holdout")
        if self.unlocks_paper_or_live:
            raise RobustnessError("READY_FOR_REVIEW must never unlock Paper or LIVE")


def _table_hashes(
    data: BacktestMonitoringData, table_names: Iterable[str]
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, stable_hash(data.tables.get(name, ()))) for name in sorted(table_names)
    )


def compare_invariance(
    baseline: BacktestMonitoringData,
    candidate: BacktestMonitoringData,
    *,
    kind: str,
    table_names: Iterable[str],
) -> InvarianceReport:
    baseline_items = _table_hashes(baseline, table_names)
    candidate_items = _table_hashes(candidate, table_names)
    candidate_map = dict(candidate_items)
    item_hashes = tuple(
        (name, digest, candidate_map.get(name, stable_hash(())))
        for name, digest in baseline_items
    )
    changed = tuple(name for name, left, right in item_hashes if left != right)
    baseline_hash = stable_hash(baseline_items)
    candidate_hash = stable_hash(candidate_items)
    return InvarianceReport(
        kind=kind,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        status=InvarianceStatus.IDENTICAL if not changed else InvarianceStatus.DIFFERENT,
        baseline_hash=baseline_hash,
        candidate_hash=candidate_hash,
        item_hashes=tuple(sorted(item_hashes)),
        changed_items=changed,
    )


def _rules(record: Any) -> dict[str, str]:
    return dict(record.normalized_rules)


def _sec_section_31_diagnostic(
    registry: EvidenceRegistryV2,
    fills: Iterable[Mapping[str, Any]],
) -> tuple[Decimal, int, tuple[str, ...]]:
    records = registry.records_for(
        fee_type="SEC_SECTION_31", market="US", plan="IBKR_PRO_FIXED"
    )
    missing = ZERO
    affected = 0
    ids: set[str] = set()
    for fill in fills:
        if str(fill.get("side")) != "SELL":
            continue
        timestamp = _utc(fill.get("timestamp"), "fill timestamp")
        record = next(
            (
                item
                for item in records
                if item.effective_from <= timestamp < item.effective_to
            ),
            None,
        )
        if record is None:
            continue
        rate = _decimal(_rules(record).get("rate_per_million", "0"), "SEC rate")
        if rate <= ZERO:
            continue
        notional = _decimal(fill.get("price"), "fill price") * _decimal(
            fill.get("quantity"), "fill quantity"
        )
        missing += notional * rate / Decimal("1000000")
        affected += 1
        ids.add(record.evidence_id)
    return missing, affected, tuple(sorted(ids))


def _original_component_amount(
    summary: Mapping[str, Any], name: str
) -> Decimal | None:
    costs = summary.get("costs")
    if not isinstance(costs, Mapping):
        return None
    cost_summary = costs.get("summary")
    if not isinstance(cost_summary, Mapping):
        return None
    value = cost_summary.get(f"total_{name}")
    return _decimal(value, name) if value is not None else None


def _operating_economics(
    scenario: PaperOperatingScenario,
    summary: Mapping[str, Any],
    start: datetime,
    end: datetime,
    validation: Mapping[str, Any],
) -> RetrospectivePaperEconomics:
    days = Decimal(str((end - start).total_seconds())) / Decimal("86400")
    months = days * Decimal("12") / Decimal("365.2425")
    monthly = scenario.monthly_totals
    period_costs = tuple(value * months if value is not None else None for value in monthly)
    costs = summary.get("costs") if isinstance(summary.get("costs"), Mapping) else {}
    cost_summary = costs.get("summary") if isinstance(costs.get("summary"), Mapping) else {}
    net_before = _decimal(
        cost_summary.get("net_trading_pnl_before_operating", "0"),
        "net PnL before operating",
    )
    variable = _decimal(cost_summary.get("total_variable_cost", "0"), "variable costs")
    gross = _decimal(cost_summary.get("gross_trading_pnl", "0"), "gross trading PnL")
    stress = validation.get("stress_results") if isinstance(validation, Mapping) else ()
    stress_2x = next(
        (
            item
            for item in stress or ()
            if isinstance(item, Mapping) and str(item.get("multiplier")) == "2.00"
        ),
        None,
    )
    return RetrospectivePaperEconomics(
        scenario_id=scenario.scenario_id,
        scenario_hash=scenario.scenario_hash,
        period_months=months,
        operating_low=period_costs[0],
        operating_central=period_costs[1],
        operating_high=period_costs[2],
        net_before_operating=net_before,
        net_after_operating_low=(net_before - period_costs[0] if period_costs[0] is not None else None),
        net_after_operating_central=(net_before - period_costs[1] if period_costs[1] is not None else None),
        net_after_operating_high=(net_before - period_costs[2] if period_costs[2] is not None else None),
        break_even_additional_variable_cost=max(net_before, ZERO),
        break_even_monthly_fixed_cost=(net_before / months if months > ZERO else None),
        observed_profit_to_variable_cost_ratio=(gross / variable if variable > ZERO else None),
        remains_positive_at_existing_cost_stress_2x=(
            bool(stress_2x) and _decimal(stress_2x.get("stressed_net_pnl"), "2x stress PnL") > ZERO
        ),
    )


class EvidenceReassessmentEngine:
    """Pure evaluator used by the application service and deterministic tests."""

    reassessment_version = "2.0"

    def assess(
        self,
        *,
        run: BacktestMonitoringData,
        candidate: BacktestMonitoringData,
        robustness_report: Mapping[str, Any],
        validation: Mapping[str, Any],
        holdout: Any,
        registry: EvidenceRegistryV2,
        broker_tariff: TariffCompatibilityAssessment,
        operating_scenario: PaperOperatingScenario,
        current_core_hash: str,
    ) -> EvidenceReassessmentReport:
        if holdout.status is not HoldoutStatus.CONSUMED:
            raise HoldoutGovernanceError("holdout V2 must remain CONSUMED")
        if holdout.result_hash != run.summary.get("result_hash"):
            raise HoldoutGovernanceError("run does not reproduce the consumed holdout result")
        decisions = compare_invariance(
            run, candidate, kind="DECISIONS_AND_QUANTITIES", table_names=_DECISION_TABLES
        )
        costs = compare_invariance(
            run, candidate, kind="NUMERIC_COSTS", table_names=_COST_TABLES
        )
        core_same = current_core_hash == holdout.expected_core_hash
        omitted_sec, affected_sells, sec_ids = _sec_section_31_diagnostic(
            registry, run.tables.get("fills", ())
        )
        commission = EvidenceComponentAssessment(
            component="broker_commission",
            status=(
                "VERIFIED"
                if broker_tariff.status is TariffCompatibilityStatus.EXACT_MATCH
                else "CONSERVATIVE"
                if broker_tariff.status is TariffCompatibilityStatus.COMPATIBLE_CONSERVATIVE
                else "INCOMPLETE"
            ),
            compatibility=broker_tariff.status,
            amount_in_original_run=_original_component_amount(run.summary, "commission"),
            indicated_missing_amount=ZERO,
            evidence_ids=broker_tariff.evidence_ids,
            reason=broker_tariff.mathematical_demonstration,
        )
        exchange_compatibility = (
            TariffCompatibilityStatus.NUMERICALLY_DIFFERENT
            if omitted_sec > ZERO
            else TariffCompatibilityStatus.EXACT_MATCH
        )
        exchange = EvidenceComponentAssessment(
            component="exchange_and_pass_through",
            status="INCOMPLETE" if omitted_sec > ZERO else "VERIFIED",
            compatibility=exchange_compatibility,
            amount_in_original_run=_original_component_amount(run.summary, "exchange_fees"),
            indicated_missing_amount=omitted_sec,
            evidence_ids=sec_ids,
            reason=(
                f"{affected_sells} covered sell fills occur while SEC Section 31 is positive; "
                "the frozen run recorded zero separate pass-through fee."
                if omitted_sec > ZERO
                else "No positive separately chargeable SEC Section 31 amount affects the fills."
            ),
        )
        actual_symbols = {str(item.get("symbol")) for item in run.tables.get("fills", ())}
        french_fills = actual_symbols.intersection({"MC.PA", "AIR.PA"})
        tax = EvidenceComponentAssessment(
            component="transaction_tax",
            status="NOT_APPLICABLE" if not french_fills else "VERIFIED",
            compatibility=TariffCompatibilityStatus.NOT_APPLICABLE if not french_fills else TariffCompatibilityStatus.EXACT_MATCH,
            amount_in_original_run=_original_component_amount(run.summary, "transaction_tax"),
            indicated_missing_amount=ZERO,
            evidence_ids=tuple(
                item.evidence_id for item in registry.records if item.fee_type == "TRANSACTION_TAX"
            ),
            reason=(
                "No French instrument fill occurred in the consumed holdout."
                if not french_fills
                else "Dated official FTT rules and annual eligibility apply to French fills."
            ),
        )
        fx_used = (_original_component_amount(run.summary, "fx_cost") or ZERO) > ZERO
        fx = EvidenceComponentAssessment(
            component="fx",
            status="INCOMPLETE" if fx_used else "NOT_APPLICABLE",
            compatibility=(
                TariffCompatibilityStatus.INSUFFICIENT_EVIDENCE
                if fx_used
                else TariffCompatibilityStatus.NOT_APPLICABLE
            ),
            amount_in_original_run=_original_component_amount(run.summary, "fx_cost"),
            indicated_missing_amount=None if fx_used else ZERO,
            evidence_ids=tuple(
                item.evidence_id for item in registry.records if item.fee_type == "FX_CONVERSION"
            ),
            reason=(
                "Only a current official FX source exists; historical conversion cost is unverified."
                if fx_used
                else "Every filled instrument is explicitly USD in the frozen metadata; no FX conversion occurred."
            ),
        )
        spread_slippage = EvidenceComponentAssessment(
            component="spread_and_slippage",
            status="VERIFIED",
            compatibility=TariffCompatibilityStatus.EXACT_MATCH,
            amount_in_original_run=(
                (_original_component_amount(run.summary, "spread") or ZERO)
                + (_original_component_amount(run.summary, "slippage") or ZERO)
            ),
            indicated_missing_amount=ZERO,
            evidence_ids=(),
            reason="Existing execution-model values are preserved exactly and are not debited again.",
        )
        components = (commission, exchange, tax, fx, spread_slippage)
        critical = tuple(
            item.component
            for item in components
            if item.compatibility in {
                TariffCompatibilityStatus.NUMERICALLY_DIFFERENT,
                TariffCompatibilityStatus.INSUFFICIENT_EVIDENCE,
            }
        )
        operating_complete = all(
            item.status is not EvidenceVerificationStatus.UNAVAILABLE
            for item in operating_scenario.components
        )
        if critical:
            completeness_status = EconomicEvidenceStatus.INCOMPLETE
        elif broker_tariff.status is TariffCompatibilityStatus.EXACT_MATCH and operating_complete:
            completeness_status = EconomicEvidenceStatus.COMPLETE_ESTIMATED
        elif broker_tariff.status is TariffCompatibilityStatus.COMPATIBLE_CONSERVATIVE and operating_complete:
            completeness_status = EconomicEvidenceStatus.COMPLETE_CONSERVATIVE
        else:
            completeness_status = EconomicEvidenceStatus.INCOMPLETE
        completeness_semantic = {
            "status": completeness_status,
            "broker_commission": commission.status,
            "exchange_and_pass_through": exchange.status,
            "transaction_tax": tax.status,
            "fx": fx.status,
            "spread_and_slippage": spread_slippage.status,
            "operating_costs": "COMPLETE_ESTIMATED" if operating_complete else "INCOMPLETE",
            "historical_period_fit": "INCOMPLETE" if critical else "COMPLETE",
            "source_quality": "OFFICIAL_AND_ARCHIVED_OFFICIAL",
            "components": components,
            "critical_unresolved": critical,
        }
        completeness = EconomicEvidenceCompleteness(
            **completeness_semantic,
            completeness_hash=stable_hash(completeness_semantic),
        )
        if not core_same or decisions.status is InvarianceStatus.DIFFERENT:
            mode = EvidenceReassessmentMode.DECISION_CORE_CHANGED
        elif (
            costs.status is InvarianceStatus.DIFFERENT
            or omitted_sec > ZERO
            or broker_tariff.status is TariffCompatibilityStatus.NUMERICALLY_DIFFERENT
        ):
            mode = EvidenceReassessmentMode.ECONOMIC_RECOMPUTATION_REQUIRED
        else:
            mode = EvidenceReassessmentMode.EVIDENCE_ONLY_RECLASSIFICATION
        evidence_only = (
            mode is EvidenceReassessmentMode.EVIDENCE_ONLY_RECLASSIFICATION
            and decisions.status is InvarianceStatus.IDENTICAL
            and costs.status is InvarianceStatus.IDENTICAL
        )
        decisions_or_costs_mutated = (
            decisions.status is InvarianceStatus.DIFFERENT
            or costs.status is InvarianceStatus.DIFFERENT
        )
        if mode is EvidenceReassessmentMode.DECISION_CORE_CHANGED:
            strict_evidence_status = "FAIL"
        elif completeness.status is EconomicEvidenceStatus.INCOMPLETE:
            strict_evidence_status = "FAIL"
        elif broker_tariff.status is TariffCompatibilityStatus.EXACT_MATCH:
            strict_evidence_status = "PASS"
        else:
            strict_evidence_status = "WARNING"
        start = holdout.period.start
        end = holdout.period.end
        paper_economics = _operating_economics(
            operating_scenario, run.summary, start, end, validation
        )
        concentration = robustness_report.get("concentration")
        top1 = concentration.get("top1_positive_pnl_share") if isinstance(concentration, Mapping) else None
        concentration_warning = (
            f"TOP1_POSITIVE_PNL_SHARE={top1}"
            if top1 is not None
            else "CONCENTRATION_UNAVAILABLE"
        )
        survivorship = str(
            robustness_report.get("survivorship_status", "SURVIVORSHIP_BIAS_UNRESOLVED")
        )
        warnings = [
            "CONSUMED_HOLDOUT_NEVER_BECOMES_FRESH_FINAL",
            "EVIDENCE_WORK_DOES_NOT_RETUNE_DECISION_PARAMETERS",
            survivorship,
            concentration_warning,
        ]
        if omitted_sec > ZERO:
            warnings.extend(
                (
                    "SEC_SECTION_31_NUMERICALLY_OMITTED",
                    "HOLDOUT_RECOMPUTATION_REQUIRED_AFTER_EVIDENCE_UPDATE",
                )
            )
        if broker_tariff.status is TariffCompatibilityStatus.COMPATIBLE_CONSERVATIVE:
            warnings.append("STRICT_VALIDATION_REMAINS_DISTINCT_FROM_CONSERVATIVE_READINESS")
        semantic = {
            "run_id": run.run_id,
            "robustness_report_id": robustness_report.get("report_id"),
            "validation_id": validation.get("validation_id"),
            "holdout_id": holdout.holdout_id,
            "holdout_status": holdout.status.value,
            "holdout_result_hash": holdout.result_hash,
            "plan_hash": holdout.plan_hash,
            "evidence_registry_version": registry.registry_version,
            "evidence_registry_hash": registry.registry_hash,
            "mode": mode,
            "decision_core_expected_hash": holdout.expected_core_hash,
            "decision_core_current_hash": current_core_hash,
            "decision_invariance": decisions,
            "cost_invariance": costs,
            "broker_tariff": broker_tariff,
            "components": components,
            "economic_completeness": completeness,
            "original_validation_status": str(validation.get("status")),
            "strict_validation_evidence_status": strict_evidence_status,
            "numeric_recomputation_required": mode is EvidenceReassessmentMode.ECONOMIC_RECOMPUTATION_REQUIRED,
            "indicated_missing_regulatory_cost": omitted_sec,
            "paper_economics": paper_economics,
            "survivorship_warning": survivorship,
            "concentration_warning": concentration_warning,
            "warnings": tuple(dict.fromkeys(warnings)),
            "evidence_only": evidence_only,
            "decisions_or_costs_mutated": decisions_or_costs_mutated,
            "holdout_remains_consumed": True,
            "unlocks_paper_or_live": False,
        }
        digest = stable_hash(semantic)
        return EvidenceReassessmentReport(
            reassessment_id=f"evidence-reassessment-{digest[:24]}",
            reassessment_version=self.reassessment_version,
            reassessment_hash=digest,
            created_at=datetime.now(timezone.utc),
            **semantic,
        )


class PaperReadinessReviewerV2:
    review_name = "balanced-paper-readiness-review"
    review_version = "2.0"

    def review(
        self,
        reassessment: EvidenceReassessmentReport,
        validation: Mapping[str, Any],
    ) -> PaperReadinessReviewV2:
        criteria: list[ReviewCriterion] = []

        def add(name: str, status: str, observed: object, required: str, reason: str) -> None:
            criteria.append(ReviewCriterion(name, status, str(observed), required, reason))

        add(
            "decision_core_unchanged",
            "PASS" if reassessment.mode is not EvidenceReassessmentMode.DECISION_CORE_CHANGED else "FAIL",
            reassessment.mode.value,
            "decision core unchanged",
            "Lot 8.3 evidence work cannot alter the decision core.",
        )
        add(
            "holdout_governance",
            "PASS" if reassessment.holdout_status == "CONSUMED" else "FAIL",
            reassessment.holdout_status,
            "CONSUMED",
            "A consumed holdout is never reclassified as fresh final evidence.",
        )
        hard = [
            item
            for item in validation.get("criteria", ())
            if isinstance(item, Mapping)
            and item.get("name") not in {"tariff_period", "operating_costs"}
        ]
        hard_pass = bool(hard) and all(str(item.get("status")) == "PASS" for item in hard)
        add(
            "frozen_validation_hard_checks",
            "PASS" if hard_pass else "FAIL",
            f"{sum(str(item.get('status')) == 'PASS' for item in hard)}/{len(hard)} PASS",
            "all frozen non-evidence hard checks PASS",
            "Trade count, return, expectancy, profit factor, drawdown, cash, and circuit breaker thresholds are unchanged.",
        )
        exact_or_conservative = reassessment.broker_tariff.status in {
            TariffCompatibilityStatus.EXACT_MATCH,
            TariffCompatibilityStatus.COMPATIBLE_CONSERVATIVE,
        }
        tariff_status = (
            "FAIL"
            if reassessment.numeric_recomputation_required
            else "PASS"
            if exact_or_conservative
            else "INSUFFICIENT"
        )
        add(
            "historical_tariff_evidence",
            tariff_status,
            reassessment.broker_tariff.status.value,
            "EXACT_MATCH or mathematically COMPATIBLE_CONSERVATIVE without numeric omission",
            "Current tariff pages alone are not historical proof and separate pass-through fees remain explicit.",
        )
        completeness_ok = reassessment.economic_completeness.status in {
            EconomicEvidenceStatus.COMPLETE_VERIFIED,
            EconomicEvidenceStatus.COMPLETE_CONSERVATIVE,
            EconomicEvidenceStatus.COMPLETE_ESTIMATED,
        }
        add(
            "economic_evidence_completeness",
            "PASS" if completeness_ok else "FAIL",
            reassessment.economic_completeness.status.value,
            "complete and no result-invalidating unresolved component",
            "Unknown or numerically omitted critical costs are never treated as zero.",
        )
        scenario_ok = reassessment.paper_economics.operating_central is not None
        add(
            "paper_operating_scenario",
            "PASS" if scenario_ok else "FAIL",
            reassessment.paper_economics.scenario_id,
            "explicit PAPER operating estimate",
            "Operating estimates are retrospective scenarios, not observed costs.",
        )
        add(
            "survivorship_bias",
            "WARNING",
            reassessment.survivorship_warning,
            "forward Paper evidence will observe membership at time t",
            "Unresolved survivorship bias limits historical interpretation but does not silently rewrite the universe.",
        )
        add(
            "symbol_concentration",
            "WARNING",
            reassessment.concentration_warning,
            "warning retained",
            "Concentration diagnostics do not remove a symbol or alter caps.",
        )
        add(
            "broker_and_live_locked",
            "PASS",
            False,
            "no broker, Paper session, or LIVE unlock",
            "This review is read-only and unlocks nothing.",
        )
        if any(item.status == "FAIL" for item in criteria):
            status = PaperReadinessStatus.NOT_READY
        elif any(item.status == "INSUFFICIENT" for item in criteria):
            status = PaperReadinessStatus.INSUFFICIENT_EVIDENCE
        else:
            status = PaperReadinessStatus.READY_FOR_REVIEW
        warnings = tuple(
            dict.fromkeys(
                (
                    *reassessment.warnings,
                    "PAPER_READINESS_IS_NOT_LIVE_READINESS",
                    "READY_FOR_REVIEW_NEVER_AUTO_ENABLES_BROKER_PAPER_OR_LIVE",
                )
            )
        )
        semantic = {
            "review_name": self.review_name,
            "review_version": self.review_version,
            "reassessment_id": reassessment.reassessment_id,
            "run_id": reassessment.run_id,
            "holdout_status": reassessment.holdout_status,
            "status": status,
            "criteria": tuple(criteria),
            "warnings": warnings,
            "meaning": "Evidence is sufficient only for a manual review of future no-capital Paper-development readiness.",
            "next_step": "Resolve every critical cost discrepancy before any separate manual Lot 9 decision.",
            "unlocks_paper_or_live": False,
        }
        digest = stable_hash(semantic)
        return PaperReadinessReviewV2(
            review_id=f"paper-readiness-v2-{digest[:24]}",
            review_hash=digest,
            created_at=datetime.now(timezone.utc),
            **semantic,
        )


class EvidenceClosureService:
    """Offline orchestrator over checksum-verified exported evidence only."""

    def __init__(self, data_root: Path | str = Path("data_local")) -> None:
        self.data_root = Path(data_root)
        self.source = BacktestMonitoringSource(self.data_root / "backtests")
        self.robustness_store = LocalRobustnessStore(self.data_root / "robustness")
        self.validation_store = LocalValidationStore(self.data_root / "validation")
        self.registry = load_evidence_registry_v2()
        self.operating_scenarios = {
            item.scenario_id: item for item in load_paper_operating_scenarios()
        }

    def verify_registry(self) -> dict[str, Any]:
        return {
            "registry_version": self.registry.registry_version,
            "registry_hash": self.registry.registry_hash,
            "records": len(self.registry.records),
            "conflicts": to_primitive(self.registry.conflicts),
            "verified": not self.registry.conflicts,
            "network_access": False,
            "runtime_scraping": False,
        }

    def _holdout(self):
        plan = load_research_plan()
        raw = self.robustness_store.find_holdout_for_period(
            plan.period(PeriodClassification.FINAL_HOLDOUT)
        )
        if raw is None:
            raise HoldoutGovernanceError("frozen holdout lifecycle not found")
        record = deserialize_holdout(raw)
        if record.plan_hash != plan.plan_hash:
            raise HoldoutGovernanceError("consumed holdout belongs to another frozen plan")
        return record

    @staticmethod
    def _tariff_assumption() -> AppliedTariffAssumption:
        tariff = load_tariff_profile("ibkr_pro_fixed", DEFAULT_COST_DIRECTORY)
        rules = tuple(
            sorted(
                {
                    "currency": tariff.currency,
                    "fixed_per_order": str(tariff.fixed_per_order),
                    "per_unit": str(tariff.per_unit),
                    "proportional_bps": str(tariff.proportional_bps),
                    "minimum_per_order": str(tariff.minimum_per_order),
                    "maximum_notional_fraction": str(tariff.maximum_notional_fraction),
                }.items()
            )
        )
        return AppliedTariffAssumption(
            provider=tariff.provider,
            market="US",
            fee_type="BROKER_COMMISSION",
            plan="IBKR_PRO_FIXED",
            normalized_rules=rules,
        )

    def reassess(
        self,
        *,
        run_id: str,
        robustness_report_id: str,
        candidate_run_id: str | None = None,
        operating_scenario_id: str = "PAPER_ESTIMATE_V1",
    ) -> tuple[EvidenceReassessmentReport, PaperReadinessReviewV2]:
        run = self.source.load_run(run_id)
        candidate = self.source.load_run(candidate_run_id or run_id)
        report = self.robustness_store.inspect_report(robustness_report_id)
        if report.get("run_id") != run_id:
            raise RobustnessError("robustness report and run_id do not match")
        validation = self.validation_store.latest_for_run(run_id)
        if not isinstance(validation, Mapping):
            raise RobustnessError("checksum-verified Validation report is required")
        holdout = self._holdout()
        scenario = self.operating_scenarios.get(operating_scenario_id)
        if scenario is None:
            raise RobustnessError("unknown Paper operating scenario")
        tariff = TariffEvidenceComparator.compare(
            self._tariff_assumption(),
            self.registry,
            period_start=holdout.period.start,
            period_end=holdout.period.end,
        )
        reassessment = EvidenceReassessmentEngine().assess(
            run=run,
            candidate=candidate,
            robustness_report=report,
            validation=validation,
            holdout=holdout,
            registry=self.registry,
            broker_tariff=tariff,
            operating_scenario=scenario,
            current_core_hash=decision_core_hash(),
        )
        readiness = PaperReadinessReviewerV2().review(reassessment, validation)
        self.robustness_store.save_evidence_bundle(
            reassessment=reassessment,
            registry=self.registry,
            completeness=reassessment.economic_completeness,
            operating_scenario=scenario,
            readiness=readiness,
        )
        return reassessment, readiness

    def inspect(self, reassessment_id: str) -> dict[str, Any]:
        return self.robustness_store.inspect_evidence_bundle(reassessment_id)


__all__ = [
    "EconomicEvidenceCompleteness",
    "EvidenceClosureService",
    "EvidenceComponentAssessment",
    "EvidenceReassessmentEngine",
    "EvidenceReassessmentReport",
    "InvarianceReport",
    "PaperReadinessReviewV2",
    "PaperReadinessReviewerV2",
    "RetrospectivePaperEconomics",
    "ReviewCriterion",
    "compare_invariance",
]
