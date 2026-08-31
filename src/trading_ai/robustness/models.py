"""Immutable contracts for Lot 8.2 research governance and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum


ZERO = Decimal("0")
ONE = Decimal("1")


def _text(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _sorted_pairs(value: tuple[tuple[str, str], ...], name: str) -> None:
    if value != tuple(sorted(value)) or len({key for key, _ in value}) != len(value):
        raise ValueError(f"{name} must contain unique deterministically sorted keys")


class PeriodClassification(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    CONSUMED_DIAGNOSTIC = "CONSUMED_DIAGNOSTIC"
    FINAL_HOLDOUT = "FINAL_HOLDOUT"


class HoldoutStatus(str, Enum):
    UNTOUCHED = "UNTOUCHED"
    CONSUMED = "CONSUMED"
    INVALIDATED = "INVALIDATED"


class CampaignStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT"
    INSUFFICIENT_HOLDOUT_EVIDENCE = "INSUFFICIENT_HOLDOUT_EVIDENCE"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class PaperReadinessStatus(str, Enum):
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    NOT_READY = "NOT_READY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SurvivorshipStatus(str, Enum):
    POINT_IN_TIME = "POINT_IN_TIME"
    SURVIVORSHIP_BIAS_UNRESOLVED = "SURVIVORSHIP_BIAS_UNRESOLVED"


class UncertaintyStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    INSUFFICIENT_FOR_RELIABLE_CI = "INSUFFICIENT_FOR_RELIABLE_CI"


class DiagnosticAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_RUN = "NOT_RUN"


class CostCompleteness(str, Enum):
    INCOMPLETE = "INCOMPLETE"
    VARIABLE_COST_COMPLETE = "VARIABLE_COST_COMPLETE"
    OPERATING_COST_COMPLETE = "OPERATING_COST_COMPLETE"
    COMPLETE_ESTIMATED = "COMPLETE_ESTIMATED"
    COMPLETE_VERIFIED = "COMPLETE_VERIFIED"


@dataclass(frozen=True, slots=True)
class ResearchPeriod:
    name: str
    classification: PeriodClassification
    start: datetime
    end: datetime
    label: str

    def __post_init__(self) -> None:
        _text(self.name, "period name")
        _text(self.label, "period label")
        _utc(self.start, "period start")
        _utc(self.end, "period end")
        if self.start >= self.end:
            raise ValueError("research period start must precede end")

    def contains(self, timestamp: datetime) -> bool:
        _utc(timestamp, "timestamp")
        return self.start <= timestamp < self.end

    def overlaps(self, other: ResearchPeriod) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    symbol: str
    dataset_id: str
    checksum: str
    corporate_actions_dataset_id: str | None = None
    corporate_actions_checksum: str | None = None

    def __post_init__(self) -> None:
        _text(self.symbol, "dataset symbol")
        _text(self.dataset_id, "dataset_id")
        _sha(self.checksum, "dataset checksum")
        if (self.corporate_actions_dataset_id is None) != (
            self.corporate_actions_checksum is None
        ):
            raise ValueError("corporate-action ID and checksum must be supplied together")
        if self.corporate_actions_dataset_id is not None:
            _text(self.corporate_actions_dataset_id, "corporate_actions_dataset_id")
            assert self.corporate_actions_checksum is not None
            _sha(self.corporate_actions_checksum, "corporate-actions checksum")


@dataclass(frozen=True, slots=True)
class ResearchBaselineManifest:
    manifest_id: str
    manifest_version: str
    manifest_hash: str
    frozen_at: datetime
    commit_sha: str
    source_hash_sha256: str
    run_id: str
    result_hash: str
    validation_id: str
    validation_status: str
    period: ResearchPeriod
    timeframe: str
    universe_kind: str
    symbols: tuple[str, ...]
    config_hashes: tuple[tuple[str, str], ...]
    datasets: tuple[DatasetFingerprint, ...]
    tariff_profile_id: str
    tariff_status: str
    tariff_period_verified: bool
    tariff_config_hash: str
    closed_trades: int
    max_drawdown: Decimal
    net_return_before_operating: Decimal
    top_contributor: str
    top_contributor_share: Decimal
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "manifest_id", "manifest_version", "run_id", "validation_id",
            "validation_status", "timeframe", "universe_kind", "tariff_profile_id",
            "tariff_status", "top_contributor",
        ):
            _text(getattr(self, name), name)
        _sha(self.manifest_hash, "manifest_hash")
        _sha(self.source_hash_sha256, "source_hash_sha256")
        _sha(self.result_hash, "result_hash")
        _sha(self.tariff_config_hash, "tariff_config_hash")
        _utc(self.frozen_at, "frozen_at")
        if len(self.commit_sha) != 40:
            raise ValueError("baseline commit_sha must identify one Git commit")
        if self.period.classification is not PeriodClassification.CONSUMED_DIAGNOSTIC:
            raise ValueError("the Lot 8.1 OOS baseline must be consumed diagnostic data")
        if self.symbols != tuple(sorted(set(self.symbols))):
            raise ValueError("baseline symbols must be unique and sorted")
        if tuple(item.symbol for item in self.datasets) != self.symbols:
            raise ValueError("baseline datasets must exactly match the frozen universe")
        _sorted_pairs(self.config_hashes, "config_hashes")
        for _, digest in self.config_hashes:
            _sha(digest, "configuration hash")
        if self.closed_trades < 0:
            raise ValueError("closed_trades must be non-negative")
        if not ZERO <= self.max_drawdown <= ONE:
            raise ValueError("max_drawdown must be in [0, 1]")
        if not ZERO <= self.top_contributor_share <= ONE:
            raise ValueError("top contributor share must be in [0, 1]")
        if self.tariff_period_verified:
            raise ValueError("the frozen 2020-2025 tariff was not historically verified")


@dataclass(frozen=True, slots=True)
class RobustnessResearchPlan:
    plan_id: str
    plan_name: str
    plan_version: str
    plan_hash: str
    frozen_at: datetime
    frozen: bool
    baseline_manifest_hash: str
    timeframe: str
    universe_kind: str
    symbols: tuple[str, ...]
    strategies: tuple[str, ...]
    ml_modes: tuple[str, ...]
    cost_profile: str
    validation_profile: str
    periods: tuple[ResearchPeriod, ...]
    frozen_validation_criteria: tuple[tuple[str, str], ...]
    config_hashes: tuple[tuple[str, str], ...]
    planned_analyses: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "plan_id", "plan_name", "plan_version", "timeframe", "universe_kind",
            "cost_profile", "validation_profile",
        ):
            _text(getattr(self, name), name)
        _sha(self.plan_hash, "plan_hash")
        _sha(self.baseline_manifest_hash, "baseline_manifest_hash")
        _utc(self.frozen_at, "frozen_at")
        if not self.frozen:
            raise ValueError("a holdout-eligible research plan must be frozen")
        if self.symbols != tuple(sorted(set(self.symbols))):
            raise ValueError("plan symbols must be unique and sorted")
        if self.strategies != tuple(sorted(set(self.strategies))):
            raise ValueError("plan strategies must be unique and sorted")
        if len({item.name for item in self.periods}) != len(self.periods):
            raise ValueError("plan period names must be unique")
        consumed = self.period(PeriodClassification.CONSUMED_DIAGNOSTIC)
        holdout = self.period(PeriodClassification.FINAL_HOLDOUT)
        if holdout.start < consumed.end:
            raise ValueError("final holdout must begin after consumed diagnostic OOS")
        _sorted_pairs(self.frozen_validation_criteria, "validation criteria")
        _sorted_pairs(self.config_hashes, "plan config hashes")
        for _, digest in self.config_hashes:
            _sha(digest, "plan configuration hash")
        if not self.planned_analyses or len(set(self.planned_analyses)) != len(
            self.planned_analyses
        ):
            raise ValueError("planned analyses must be a non-empty unique declaration")

    def period(self, classification: PeriodClassification) -> ResearchPeriod:
        matches = [item for item in self.periods if item.classification is classification]
        if len(matches) != 1:
            raise ValueError(f"plan must define exactly one {classification.value} period")
        return matches[0]


@dataclass(frozen=True, slots=True)
class HoldoutRecord:
    holdout_id: str
    plan_hash: str
    period: ResearchPeriod
    status: HoldoutStatus
    expected_core_hash: str
    expected_config_hashes: tuple[tuple[str, str], ...]
    record_hash: str
    consumed_at: datetime | None = None
    result_hash: str | None = None
    invalidated_at: datetime | None = None
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.holdout_id, "holdout_id")
        _sha(self.plan_hash, "plan_hash")
        _sha(self.expected_core_hash, "expected_core_hash")
        _sha(self.record_hash, "record_hash")
        _sorted_pairs(self.expected_config_hashes, "expected_config_hashes")
        if self.period.classification is not PeriodClassification.FINAL_HOLDOUT:
            raise ValueError("holdout record must describe FINAL_HOLDOUT")
        if self.status is HoldoutStatus.UNTOUCHED:
            if any((self.consumed_at, self.result_hash, self.invalidated_at, self.invalidation_reason)):
                raise ValueError("untouched holdout cannot contain evaluation metadata")
        elif self.status is HoldoutStatus.CONSUMED:
            if self.consumed_at is None or self.result_hash is None:
                raise ValueError("consumed holdout requires timestamp and result hash")
            _utc(self.consumed_at, "consumed_at")
            _sha(self.result_hash, "holdout result_hash")
        else:
            if self.invalidated_at is None or self.invalidation_reason is None:
                raise ValueError("invalidated holdout requires an explicit reason")
            _utc(self.invalidated_at, "invalidated_at")
            _text(self.invalidation_reason, "invalidation_reason")


@dataclass(frozen=True, slots=True)
class HistoricalCoverageRow:
    symbol: str
    timeframe: str
    dataset_id: str
    checksum: str
    first_timestamp: datetime
    last_timestamp: datetime
    row_count: int
    missing_expected_bars: int
    duplicate_count: int
    invalid_bar_count: int
    corporate_actions_available: bool
    quality_status: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.symbol, "coverage symbol")
        _text(self.timeframe, "coverage timeframe")
        _text(self.dataset_id, "coverage dataset_id")
        _sha(self.checksum, "coverage checksum")
        _utc(self.first_timestamp, "coverage first_timestamp")
        _utc(self.last_timestamp, "coverage last_timestamp")
        if self.first_timestamp > self.last_timestamp:
            raise ValueError("coverage timestamp range is inverted")
        if min(self.row_count, self.missing_expected_bars, self.duplicate_count, self.invalid_bar_count) < 0:
            raise ValueError("coverage counts must be non-negative")


@dataclass(frozen=True, slots=True)
class HistoricalCoverageMatrix:
    rows: tuple[HistoricalCoverageRow, ...]
    common_start: datetime | None
    common_end: datetime | None
    common_history_available: bool
    provider_limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionFunnelRow:
    strategy_name: str
    symbol: str
    candidate_entries: int
    ml_eligible: int
    ml_blocked: int
    activation_eligible: int
    activation_blocked: int
    portfolio_selected: int
    portfolio_deferred: int
    portfolio_rejected: int
    economic_eligible: int
    economic_blocked: int
    economic_incomplete: int
    risk_approved: int
    risk_reduced: int
    risk_rejected: int
    filled_entries: int
    exit_signals: int
    closed_trades: int

    def __post_init__(self) -> None:
        _text(self.strategy_name, "funnel strategy")
        _text(self.symbol, "funnel symbol")
        values = (
            self.candidate_entries, self.ml_eligible, self.ml_blocked,
            self.activation_eligible, self.activation_blocked,
            self.portfolio_selected, self.portfolio_deferred, self.portfolio_rejected,
            self.economic_eligible, self.economic_blocked, self.economic_incomplete,
            self.risk_approved, self.risk_reduced, self.risk_rejected,
            self.filled_entries, self.exit_signals, self.closed_trades,
        )
        if any(value < 0 for value in values):
            raise ValueError("funnel counts must be non-negative")
        if not (
            self.candidate_entries >= self.ml_eligible
            >= self.activation_eligible >= self.portfolio_selected
            >= self.economic_eligible >= self.risk_approved + self.risk_reduced
            >= self.filled_entries
        ):
            raise ValueError("entry decision funnel must be monotone")


@dataclass(frozen=True, slots=True)
class DecisionFunnelReport:
    rows: tuple[DecisionFunnelRow, ...]
    drop_reasons: tuple[tuple[str, int], ...]
    interpretation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    episode_id: str
    peak_timestamp: datetime
    peak_equity: Decimal
    trough_timestamp: datetime
    trough_equity: Decimal
    recovery_timestamp: datetime | None
    drawdown_fraction: Decimal
    duration_seconds: float
    held_symbols_at_trough: tuple[str, ...]
    realized_closure_pnl_by_symbol: tuple[tuple[str, Decimal], ...]
    max_gross_exposure: Decimal | None
    risk_states: tuple[str, ...]
    risk_decision_counts: tuple[tuple[str, int], ...]
    structure_regimes_at_trough: tuple[tuple[str, str], ...]
    volatility_regimes_at_trough: tuple[tuple[str, str], ...]
    variable_costs_during_episode: Decimal
    attribution_scope: str = "REALIZED_CLOSURES_AND_OBSERVED_STATE_ONLY"


@dataclass(frozen=True, slots=True)
class SymbolPnlContribution:
    symbol: str
    closed_trades: int
    gross_pnl: Decimal
    net_pnl: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    average_target_weight: Decimal | None
    share_of_positive_pnl: Decimal | None


@dataclass(frozen=True, slots=True)
class PnlConcentrationReport:
    symbols: tuple[SymbolPnlContribution, ...]
    top_contributor: str | None
    top1_positive_pnl_share: Decimal | None
    top3_positive_pnl_share: Decimal | None
    positive_pnl_hhi: Decimal | None
    dominant_symbol_warning: bool
    definition: str


@dataclass(frozen=True, slots=True)
class TemporalRobustnessRow:
    label: str
    start: datetime
    end: datetime
    availability: DiagnosticAvailability
    gross_return: Decimal | None
    net_return: Decimal | None
    closed_trades: int
    expectancy: Decimal | None
    profit_factor: Decimal | None
    max_drawdown: Decimal | None
    average_exposure: Decimal | None
    turnover: Decimal | None
    variable_costs: Decimal | None


@dataclass(frozen=True, slots=True)
class RegimeRobustnessRow:
    structure_regime: str
    volatility_regime: str
    closed_trades: int
    net_pnl: Decimal
    average_return: Decimal | None
    attribution_scope: str = "ENTRY_REGIME_MATCHED_TO_CLOSED_TRADES"


@dataclass(frozen=True, slots=True)
class CostRobustnessReport:
    completeness: CostCompleteness
    evidence_registry_hash: str
    evidence_verified_at: datetime
    broker_tariff_evidence_kind: str
    exchange_fee_evidence_status: str
    fx_cost_evidence_status: str
    annual_tax_eligibility_records: int
    tariff_profile_id: str | None
    tariff_status: str | None
    historical_tariff_status: str
    total_variable_cost: Decimal | None
    cost_per_closed_trade: Decimal | None
    cost_to_notional: Decimal | None
    cost_to_gross_profit: Decimal | None
    aggregate_estimate_error: Decimal | None
    stress_results: tuple[tuple[Decimal, Decimal | None, Decimal | None], ...]
    operating_scenario: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.evidence_registry_hash, "historical cost evidence registry hash")
        _utc(self.evidence_verified_at, "historical cost evidence verified_at")
        if self.annual_tax_eligibility_records < 0:
            raise ValueError("tax eligibility evidence count must be non-negative")


@dataclass(frozen=True, slots=True)
class StatisticalUncertaintyReport:
    status: UncertaintyStatus
    sample_count: int
    win_rate: Decimal | None
    win_rate_interval_95: tuple[Decimal, Decimal] | None
    expectancy_interval_95: tuple[Decimal, Decimal] | None
    total_return_interval_95: tuple[Decimal, Decimal] | None
    bootstrap_seed: int
    bootstrap_resamples: int
    warning: str | None


@dataclass(frozen=True, slots=True)
class DiagnosticComparisonResult:
    label: str
    diagnostic_type: str
    excluded_item: str
    run_id: str | None
    source_hash_sha256: str | None
    availability: DiagnosticAvailability
    net_return: Decimal | None
    max_drawdown: Decimal | None
    closed_trades: int | None
    expectancy: Decimal | None
    profit_factor: Decimal | None
    config_hashes_unchanged: bool
    post_hoc_only: bool = True
    warning: str = "POST_HOC_ROBUSTNESS_DIAGNOSTIC"

    def __post_init__(self) -> None:
        if self.source_hash_sha256 is not None:
            _sha(self.source_hash_sha256, "diagnostic source_hash_sha256")


@dataclass(frozen=True, slots=True)
class UniverseMembership:
    symbol: str
    valid_from: datetime
    valid_to: datetime | None
    source: str
    status: str

    def __post_init__(self) -> None:
        _text(self.symbol, "membership symbol")
        _text(self.source, "membership source")
        _text(self.status, "membership status")
        _utc(self.valid_from, "membership valid_from")
        if self.valid_to is not None:
            _utc(self.valid_to, "membership valid_to")
            if self.valid_to <= self.valid_from:
                raise ValueError("membership valid_to must follow valid_from")

    def active_at(self, timestamp: datetime) -> bool:
        _utc(timestamp, "membership timestamp")
        return self.valid_from <= timestamp and (
            self.valid_to is None or timestamp < self.valid_to
        )


@dataclass(frozen=True, slots=True)
class PointInTimeUniverse:
    universe_id: str
    memberships: tuple[UniverseMembership, ...]
    source: str
    status: SurvivorshipStatus

    def members_at(self, timestamp: datetime) -> tuple[str, ...]:
        return tuple(sorted({item.symbol for item in self.memberships if item.active_at(timestamp)}))


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    report_id: str
    report_version: str
    report_hash: str
    created_at: datetime
    run_id: str
    baseline_manifest_hash: str
    plan_hash: str
    period_classification: PeriodClassification
    campaign_status: CampaignStatus
    baseline_reproduced: bool
    coverage: HistoricalCoverageMatrix
    decision_funnel: DecisionFunnelReport
    drawdown_episodes: tuple[DrawdownEpisode, ...]
    concentration: PnlConcentrationReport
    temporal_rows: tuple[TemporalRobustnessRow, ...]
    regime_rows: tuple[RegimeRobustnessRow, ...]
    cost_robustness: CostRobustnessReport
    uncertainty: StatisticalUncertaintyReport
    leave_one_symbol_out: tuple[DiagnosticComparisonResult, ...]
    leave_one_strategy_out: tuple[DiagnosticComparisonResult, ...]
    single_strategy_runs: tuple[DiagnosticComparisonResult, ...]
    survivorship_status: SurvivorshipStatus
    holdout_status: HoldoutStatus | None
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.report_id, "report_id")
        _text(self.report_version, "report_version")
        _text(self.run_id, "run_id")
        _sha(self.report_hash, "report_hash")
        _sha(self.baseline_manifest_hash, "baseline_manifest_hash")
        _sha(self.plan_hash, "plan_hash")
        _utc(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class PaperReadinessCriterion:
    name: str
    status: str
    observed: str
    required: str
    reason: str


@dataclass(frozen=True, slots=True)
class PaperReadinessReport:
    review_id: str
    review_version: str
    review_hash: str
    created_at: datetime
    robustness_report_id: str
    validation_status: str
    holdout_status: str
    status: PaperReadinessStatus
    criteria: tuple[PaperReadinessCriterion, ...]
    warnings: tuple[str, ...]
    unlocks_paper_or_live: bool = False

    def __post_init__(self) -> None:
        _text(self.review_id, "review_id")
        _text(self.review_version, "review_version")
        _text(self.robustness_report_id, "robustness_report_id")
        _sha(self.review_hash, "review_hash")
        _utc(self.created_at, "created_at")
        if self.unlocks_paper_or_live:
            raise ValueError("paper-readiness review must never unlock PAPER or LIVE")
