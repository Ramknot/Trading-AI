"""Read-only evidence gates. No dependency on execution or broker APIs."""
from dataclasses import dataclass

from trading_ai.brokers.soak.models import GateResult, PaperReadOnlySoakReport, SoakConfig


class PaperReadOnlyReconciliationGate:
    name = "paper-read-only-reconciliation"
    version = "1.0"

    def evaluate(self, *, config: SoakConfig, observed_seconds: float,
                 initial: str, final: str, verified: bool, integrity: bool,
                 failures: tuple[str, ...], warnings: tuple[str, ...],
                 armed: bool = False, submit_calls: int = 0, cancel_calls: int = 0) -> GateResult:
        bad = set(failures)
        if not verified:
            bad.add("ACCOUNT_NOT_VERIFIED")
        if not integrity:
            bad.add("EVIDENCE_INTEGRITY_ERROR")
        if armed or submit_calls or cancel_calls:
            bad.add("READ_ONLY_VIOLATION")
        if initial != "IN_SYNC" or final != "IN_SYNC":
            bad.add("INITIAL_OR_FINAL_RECONCILIATION_NOT_IN_SYNC")
        if bad:
            return GateResult("FAIL", "NO_PASS", tuple(sorted(bad)))
        if warnings:
            return GateResult("WARNING", "NO_PASS", tuple(sorted(set(warnings))))
        if observed_seconds < config.smoke_minimum_seconds:
            return GateResult("INSUFFICIENT_DURATION", "NO_PASS", ("MINIMUM_DURATION_NOT_MET",))
        level = ("READ_ONLY_SOAK_PASS" if observed_seconds >= config.soak_minimum_seconds
                 else "READ_ONLY_SMOKE_PASS")
        return GateResult("PASS", level, ())


@dataclass(frozen=True)
class Lot10ReadinessResult:
    status: str
    reasons: tuple[str, ...]
    auto_arms_execution: bool = False


class Lot10ReadinessGate:
    def evaluate(self, report: PaperReadOnlySoakReport | None, *, lot9_done: bool,
                 connectivity_pass: bool, evidence_integrity: bool,
                 real_broker_evidence: bool) -> Lot10ReadinessResult:
        if report is None:
            return Lot10ReadinessResult("INSUFFICIENT_EVIDENCE", ("SOAK_NOT_RUN",))
        if (not evidence_integrity or report.gate.status == "FAIL" or
            report.paper_execution_armed or not report.live_hard_locked or
            report.critical_drift or not report.account_verified):
            return Lot10ReadinessResult("NOT_READY", ("SAFETY_OR_RECONCILIATION_FAILED",))
        if (not lot9_done or not connectivity_pass or not real_broker_evidence or
            report.gate.status != "PASS" or report.observed_seconds < 3600 or
            report.reconciliation_initial != "IN_SYNC" or report.broker_callback_error_count or
            report.submit_order_calls or report.cancel_order_calls or
            report.gate.evidence_level != "READ_ONLY_SOAK_PASS" or
            report.reconciliation_final != "IN_SYNC"):
            return Lot10ReadinessResult("INSUFFICIENT_EVIDENCE", ("REAL_QUALIFYING_SOAK_REQUIRED",))
        return Lot10ReadinessResult("READY_FOR_HUMAN_REVIEW", ("NO_EXECUTION_AUTHORIZATION",))
