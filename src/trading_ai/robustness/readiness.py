"""Read-only Paper readiness review; never an execution unlock."""

from __future__ import annotations

from datetime import datetime, timezone

from trading_ai.core.hashing import stable_hash
from trading_ai.robustness.models import (
    CampaignStatus,
    CostCompleteness,
    DiagnosticAvailability,
    HoldoutStatus,
    PaperReadinessCriterion,
    PaperReadinessReport,
    PaperReadinessStatus,
    RobustnessReport,
    SurvivorshipStatus,
    UncertaintyStatus,
)


class PaperReadinessReviewer:
    review_name = "balanced-paper-readiness-review"
    review_version = "1.0"

    def review(
        self,
        report: RobustnessReport,
        *,
        validation_status: str,
    ) -> PaperReadinessReport:
        criteria: list[PaperReadinessCriterion] = []

        def add(name: str, status: str, observed: object, required: str, reason: str) -> None:
            criteria.append(
                PaperReadinessCriterion(
                    name=name,
                    status=status,
                    observed=str(observed),
                    required=required,
                    reason=reason,
                )
            )

        add(
            "research_validation",
            "PASS" if validation_status == "PASS" else "FAIL",
            validation_status,
            "PASS",
            "Existing hard ValidationGate criteria remain visible and unchanged.",
        )
        holdout_evidence = (
            report.holdout_status is HoldoutStatus.CONSUMED
            and report.campaign_status
            not in {
                CampaignStatus.INSUFFICIENT,
                CampaignStatus.INSUFFICIENT_HOLDOUT_EVIDENCE,
                CampaignStatus.NOT_RUN,
            }
        )
        add(
            "final_holdout_evidence",
            "PASS" if holdout_evidence else "INSUFFICIENT",
            report.holdout_status.value if report.holdout_status else "NOT_RUN",
            "CONSUMED with adequate duration and sample",
            "The final holdout is frozen before first evaluation and cannot be manufactured by combining consumed data.",
        )
        historical_costs = report.cost_robustness.historical_tariff_status == "HISTORICAL_TARIFF_VERIFIED"
        add(
            "historical_cost_provenance",
            "PASS" if historical_costs else "FAIL",
            report.cost_robustness.historical_tariff_status,
            "HISTORICAL_TARIFF_VERIFIED",
            "A current tariff applied retrospectively is not historical proof.",
        )
        complete_cost = report.cost_robustness.completeness in {
            CostCompleteness.COMPLETE_ESTIMATED,
            CostCompleteness.COMPLETE_VERIFIED,
        }
        add(
            "economic_completeness",
            "PASS" if complete_cost else "WARNING",
            report.cost_robustness.completeness.value,
            "COMPLETE_ESTIMATED or COMPLETE_VERIFIED",
            "Operating costs remain separate from variable trading costs.",
        )
        add(
            "sample_uncertainty",
            "PASS" if report.uncertainty.status is UncertaintyStatus.AVAILABLE else "INSUFFICIENT",
            report.uncertainty.status.value,
            "sufficient sample for declared interval",
            "Small samples remain insufficient; no threshold is relaxed.",
        )
        concentration = report.concentration.dominant_symbol_warning
        add(
            "symbol_concentration",
            "WARNING" if concentration else "PASS",
            report.concentration.top1_positive_pnl_share,
            "not dominated above the predeclared warning fraction",
            "Leave-one-out remains diagnostic and never changes the configured universe.",
        )
        add(
            "survivorship",
            "WARNING" if report.survivorship_status is SurvivorshipStatus.SURVIVORSHIP_BIAS_UNRESOLVED else "PASS",
            report.survivorship_status.value,
            "POINT_IN_TIME",
            "The current curated universe is not a point-in-time membership history.",
        )
        loso_complete = all(
            item.availability is DiagnosticAvailability.AVAILABLE
            for item in report.leave_one_symbol_out
        )
        add(
            "leave_one_symbol_out",
            "PASS" if loso_complete else "WARNING",
            "COMPLETE" if loso_complete else "INCOMPLETE",
            "all predeclared symbols evaluated",
            "This is a post-hoc robustness diagnostic, not universe selection.",
        )
        leave_strategy_complete = all(
            item.availability is DiagnosticAvailability.AVAILABLE
            for item in report.leave_one_strategy_out
        )
        add(
            "leave_one_strategy_out",
            "PASS" if leave_strategy_complete else "WARNING",
            "COMPLETE" if leave_strategy_complete else "INCOMPLETE",
            "all predeclared strategy exclusions evaluated",
            "This is a post-hoc robustness diagnostic, not strategy selection.",
        )
        single_strategy_complete = all(
            item.availability is DiagnosticAvailability.AVAILABLE
            for item in report.single_strategy_runs
        )
        add(
            "single_strategy_comparison",
            "PASS" if single_strategy_complete else "WARNING",
            "COMPLETE" if single_strategy_complete else "INCOMPLETE",
            "all predeclared single-strategy runs evaluated",
            "Single-strategy results do not select a winner or reallocate sleeves.",
        )
        hard_fail = any(item.status == "FAIL" for item in criteria)
        insufficient = any(item.status == "INSUFFICIENT" for item in criteria)
        if hard_fail:
            status = PaperReadinessStatus.NOT_READY
        elif insufficient:
            status = PaperReadinessStatus.INSUFFICIENT_EVIDENCE
        else:
            status = PaperReadinessStatus.READY_FOR_REVIEW
        warnings = tuple(
            dict.fromkeys(
                (*report.warnings, "PAPER_READINESS_REVIEW_DOES_NOT_ENABLE_PAPER_OR_LIVE")
            )
        )
        semantic = {
            "review_version": self.review_version,
            "robustness_report_id": report.report_id,
            "validation_status": validation_status,
            "holdout_status": report.holdout_status.value if report.holdout_status else "NOT_RUN",
            "status": status,
            "criteria": tuple(criteria),
            "warnings": warnings,
            "unlocks_paper_or_live": False,
        }
        digest = stable_hash(semantic)
        return PaperReadinessReport(
            review_id=f"paper-readiness-{digest[:24]}",
            review_version=self.review_version,
            review_hash=digest,
            created_at=datetime.now(timezone.utc),
            robustness_report_id=report.report_id,
            validation_status=validation_status,
            holdout_status=(
                report.holdout_status.value if report.holdout_status else "NOT_RUN"
            ),
            status=status,
            criteria=tuple(criteria),
            warnings=warnings,
        )
