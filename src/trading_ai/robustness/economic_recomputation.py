"""Consumed-holdout economic recomputation and human Paper-readiness review.

This module is intentionally outside the trading decision core.  It consumes
checksum-verified immutable exports and EvidenceRegistryV2, recomputes dated
regulatory economics, and emits a new analytical bundle.  It never mutates the
original run, retrains a model, changes an order, or enables Paper/LIVE.
"""

from __future__ import annotations

import json
import hashlib
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.hashing import stable_hash
from trading_ai.core.versioning import detect_git_commit
from trading_ai.costs.config import (
    DEFAULT_COST_DIRECTORY,
    inspect_cost_config,
    load_instruments,
)
from trading_ai.monitoring.source import BacktestMonitoringData, BacktestMonitoringSource
from trading_ai.robustness.evidence import (
    EconomicEvidenceStatus,
    EvidenceRecord,
    EvidenceRegistryV2,
    EvidenceVerificationStatus,
    PaperOperatingScenario,
    load_evidence_registry_v2,
    load_paper_operating_scenarios,
)
from trading_ai.robustness.exceptions import HoldoutGovernanceError, RobustnessError
from trading_ai.robustness.governance import decision_core_hash
from trading_ai.robustness.models import HoldoutStatus, PaperReadinessStatus
from trading_ai.robustness.service import deserialize_holdout
from trading_ai.robustness.storage import LocalRobustnessStore
from trading_ai.validation import LocalValidationStore, load_validation_config


ZERO = Decimal("0")
ONE_MILLION = Decimal("1000000")
DEFAULT_RECOMPUTATION_CONFIG = (
    PROJECT_ROOT / "config" / "robustness" / "economic_recomputation_v1_1.toml"
)


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
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RobustnessError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise RobustnessError(f"{name} must be a SHA-256 digest")


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RobustnessError(f"{name} must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise RobustnessError(f"{name} must be an object")
    return dict(value)


class DecisionInvarianceStatus(str, Enum):
    STRICTLY_INVARIANT = "STRICTLY_INVARIANT"
    ECONOMIC_DECISION_CHANGED = "ECONOMIC_DECISION_CHANGED"
    RISK_DECISION_CHANGED = "RISK_DECISION_CHANGED"
    ORDER_OR_FILL_CHANGED = "ORDER_OR_FILL_CHANGED"
    DECISION_CORE_CHANGED = "DECISION_CORE_CHANGED"


class RecomputedEconomicStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class HumanReviewStatus(str, Enum):
    AWAITING_HUMAN_REVIEW = "AWAITING_HUMAN_REVIEW"
    HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT = (
        "HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT"
    )
    HUMAN_REVIEW_REJECTED = "HUMAN_REVIEW_REJECTED"


@dataclass(frozen=True, slots=True)
class EconomicRecomputationConfig:
    name: str
    version: str
    engine_name: str
    engine_version: str
    base_currency: str
    evidence_registry_version: str
    fee_type: str
    market: str
    plan: str
    rounding_quantum: Decimal
    rounding_mode: str
    period_label: str
    require_verified_evidence: bool
    config_hash: str

    def __post_init__(self) -> None:
        if self.version != "1.1" or self.engine_version != "1.1":
            raise RobustnessError("Lot 8.4 economic recomputation requires model 1.1")
        if self.rounding_quantum <= ZERO:
            raise RobustnessError("rounding quantum must be positive")
        if self.rounding_mode != "ROUND_HALF_EVEN":
            raise RobustnessError("unsupported regulatory-fee rounding mode")
        if self.period_label != "CONSUMED_HOLDOUT_ECONOMIC_RECOMPUTATION":
            raise HoldoutGovernanceError("economic recomputation cannot freshen the holdout")
        _sha(self.config_hash, "economic recomputation config hash")


def load_economic_recomputation_config(
    path: Path = DEFAULT_RECOMPUTATION_CONFIG,
) -> EconomicRecomputationConfig:
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        semantic = dict(raw)
        return EconomicRecomputationConfig(
            name=str(raw["name"]),
            version=str(raw["version"]),
            engine_name=str(raw["engine_name"]),
            engine_version=str(raw["engine_version"]),
            base_currency=str(raw["base_currency"]),
            evidence_registry_version=str(raw["evidence_registry_version"]),
            fee_type=str(raw["fee_type"]),
            market=str(raw["market"]),
            plan=str(raw["plan"]),
            rounding_quantum=_decimal(raw["rounding_quantum"], "rounding_quantum"),
            rounding_mode=str(raw["rounding_mode"]),
            period_label=str(raw["period_label"]),
            require_verified_evidence=bool(raw["require_verified_evidence"]),
            config_hash=stable_hash(semantic),
        )
    except RobustnessError:
        raise
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RobustnessError(f"invalid economic recomputation config: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RegulatoryFeeResult:
    status: str
    amount: Decimal | None
    evidence_id: str | None
    rate_per_million: Decimal | None
    reason: str


class Section31RuleBook:
    """Point-in-time Section 31 rule selection from verified evidence only."""

    def __init__(
        self, registry: EvidenceRegistryV2, config: EconomicRecomputationConfig
    ) -> None:
        if registry.registry_version != config.evidence_registry_version:
            raise RobustnessError("Evidence Registry version mismatch")
        self.registry = registry
        self.config = config
        self.records = registry.records_for(
            fee_type=config.fee_type, market=config.market, plan=config.plan
        )
        conflicts = tuple(
            item
            for item in registry.conflicts
            if item.fee_type == config.fee_type
            and item.market == config.market
            and item.plan == config.plan
        )
        if conflicts:
            raise RobustnessError("INSUFFICIENT_EVIDENCE: conflicting Section 31 rules")
        if not self.records:
            raise RobustnessError("INSUFFICIENT_EVIDENCE: Section 31 rules are absent")

    @staticmethod
    def _rules(record: EvidenceRecord) -> dict[str, str]:
        return dict(record.normalized_rules)

    def _record_at(self, timestamp: datetime) -> EvidenceRecord | None:
        matches = tuple(
            item
            for item in self.records
            if item.effective_from <= timestamp < item.effective_to
            and item.verification_status is EvidenceVerificationStatus.VERIFIED
        )
        if len(matches) > 1:
            raise RobustnessError("INSUFFICIENT_EVIDENCE: overlapping Section 31 rules")
        return matches[0] if matches else None

    def calculate(
        self,
        *,
        timestamp: datetime,
        side: str,
        market: str,
        notional: Decimal,
        covered_instrument: bool,
        require_period_coverage: bool = True,
    ) -> RegulatoryFeeResult:
        timestamp = _utc(timestamp, "Section 31 timestamp")
        if notional < ZERO:
            raise RobustnessError("Section 31 notional must be non-negative")
        if side != "SELL":
            return RegulatoryFeeResult(
                "NOT_APPLICABLE", ZERO, None, None,
                "Section 31 applies only to covered sales.",
            )
        if market != self.config.market or not covered_instrument:
            return RegulatoryFeeResult(
                "NOT_APPLICABLE", ZERO, None, None,
                "Transaction is outside the verified covered-instrument scope.",
            )
        record = self._record_at(timestamp)
        if record is None:
            if require_period_coverage:
                raise RobustnessError(
                    "INSUFFICIENT_EVIDENCE: no verified Section 31 rule covers the sale"
                )
            return RegulatoryFeeResult(
                "NOT_APPLICABLE", ZERO, None, None,
                "Sale is outside the configured evidence period.",
            )
        rules = self._rules(record)
        if rules.get("applicable_side") != "SELL":
            raise RobustnessError("INSUFFICIENT_EVIDENCE: invalid Section 31 side scope")
        rate = _decimal(rules.get("rate_per_million"), "Section 31 rate")
        amount = (notional * rate / ONE_MILLION).quantize(
            self.config.rounding_quantum, rounding=ROUND_HALF_EVEN
        )
        return RegulatoryFeeResult(
            "KNOWN", amount, record.evidence_id, rate,
            "Verified point-in-time SEC Section 31 rate applied to covered sale notional.",
        )


@dataclass(frozen=True, slots=True)
class AffectedFillCost:
    fill_id: str
    order_id: str
    symbol: str
    timestamp: datetime
    quantity: Decimal
    execution_price: Decimal
    notional: Decimal
    evidence_id: str
    rate_per_million: Decimal
    original_exchange_fees: Decimal
    section31_cost: Decimal
    recomputed_exchange_fees: Decimal
    original_total_variable_cost: Decimal
    recomputed_total_variable_cost: Decimal
    original_ledger_cash_change: Decimal
    recomputed_ledger_cash_change: Decimal

    def __post_init__(self) -> None:
        _utc(self.timestamp, "affected fill timestamp")
        if self.section31_cost <= ZERO:
            raise RobustnessError("affected fill must have a positive Section 31 cost")


@dataclass(frozen=True, slots=True)
class RecomputedTradeOutcome:
    trade_id: str
    symbol: str
    exit_time: datetime
    original_net_pnl: Decimal
    section31_allocation: Decimal
    recomputed_net_pnl: Decimal


@dataclass(frozen=True, slots=True)
class RecomputedEquityPoint:
    timestamp: datetime
    original_cash: Decimal
    recomputed_cash: Decimal
    original_equity: Decimal
    recomputed_equity: Decimal
    cumulative_section31: Decimal


@dataclass(frozen=True, slots=True)
class LayerInvariance:
    layer: str
    original_hash: str
    recomputed_hash: str
    invariant: bool
    reason: str

    def __post_init__(self) -> None:
        _sha(self.original_hash, f"{self.layer} original hash")
        _sha(self.recomputed_hash, f"{self.layer} recomputed hash")


@dataclass(frozen=True, slots=True)
class DecisionInvarianceReportV3:
    status: DecisionInvarianceStatus
    expected_core_hash: str
    current_core_hash: str
    layers: tuple[LayerInvariance, ...]
    changed_layers: tuple[str, ...]
    report_hash: str

    def __post_init__(self) -> None:
        _sha(self.expected_core_hash, "expected core hash")
        _sha(self.current_core_hash, "current core hash")
        _sha(self.report_hash, "decision invariance report hash")
        if self.layers != tuple(sorted(self.layers, key=lambda item: item.layer)):
            raise RobustnessError("decision-invariance layers must be sorted")


def classify_decision_invariance(
    *,
    core_changed: bool,
    economic_changed: bool,
    risk_changed: bool,
    order_or_fill_changed: bool,
) -> DecisionInvarianceStatus:
    if core_changed:
        return DecisionInvarianceStatus.DECISION_CORE_CHANGED
    if order_or_fill_changed:
        return DecisionInvarianceStatus.ORDER_OR_FILL_CHANGED
    if risk_changed:
        return DecisionInvarianceStatus.RISK_DECISION_CHANGED
    if economic_changed:
        return DecisionInvarianceStatus.ECONOMIC_DECISION_CHANGED
    return DecisionInvarianceStatus.STRICTLY_INVARIANT


@dataclass(frozen=True, slots=True)
class RecomputedMetrics:
    initial_cash: Decimal
    closed_trades: int
    affected_fills: int
    original_section31: Decimal
    recomputed_section31: Decimal
    original_variable_costs: Decimal
    recomputed_variable_costs: Decimal
    original_net_pnl_before_operating: Decimal
    recomputed_net_pnl_before_operating: Decimal
    pnl_delta: Decimal
    original_net_return: Decimal
    recomputed_net_return: Decimal
    return_delta: Decimal
    original_max_drawdown: Decimal
    recomputed_max_drawdown: Decimal
    drawdown_delta: Decimal
    original_profit_factor: Decimal
    recomputed_profit_factor: Decimal
    original_expectancy: Decimal
    recomputed_expectancy: Decimal
    original_minimum_cash: Decimal
    recomputed_minimum_cash: Decimal
    minimum_cash_delta: Decimal
    cost_per_trade: Decimal
    cost_to_notional: Decimal
    profit_to_cost_ratio: Decimal | None
    stress_results: tuple[tuple[str, Decimal], ...]


@dataclass(frozen=True, slots=True)
class OperatingEconomicScenarios:
    scenario_id: str
    scenario_hash: str
    period_months: Decimal
    operating_low: Decimal
    operating_central: Decimal
    operating_high: Decimal
    net_before_operating: Decimal
    net_after_low: Decimal
    net_after_central: Decimal
    net_after_high: Decimal
    break_even_fixed_monthly: Decimal
    label: str = "RETROSPECTIVE_ANALYSIS_NOT_A_FORECAST"


@dataclass(frozen=True, slots=True)
class EconomicEvidenceCompletenessV3:
    status: EconomicEvidenceStatus
    component_statuses: tuple[tuple[str, str], ...]
    critical_unresolved: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    completeness_hash: str

    def __post_init__(self) -> None:
        _sha(self.completeness_hash, "economic completeness V3 hash")
        if self.component_statuses != tuple(sorted(self.component_statuses)):
            raise RobustnessError("economic component statuses must be sorted")


@dataclass(frozen=True, slots=True)
class ReadinessCriterionV3:
    name: str
    status: str
    observed: str
    required: str
    reason: str


@dataclass(frozen=True, slots=True)
class EconomicRecomputationReport:
    recomputation_id: str
    recomputation_version: str
    recomputation_hash: str
    created_at: datetime
    period_label: str
    original_run_id: str
    original_result_hash: str
    robustness_report_id: str
    validation_id: str
    validation_config_hash: str
    frozen_validation_thresholds: tuple[tuple[str, str], ...]
    evidence_reassessment_id: str
    holdout_id: str
    holdout_status: str
    baseline_manifest_hash: str
    research_plan_hash: str
    code_sha: str | None
    source_hash: str
    evidence_registry_hash: str
    original_cost_engine_name: str
    original_cost_engine_version: str
    original_cost_config_hash: str
    recomputation_engine_name: str
    recomputation_engine_version: str
    recomputation_config_hash: str
    affected_fills: tuple[AffectedFillCost, ...]
    affected_trades: tuple[RecomputedTradeOutcome, ...]
    recomputed_equity: tuple[RecomputedEquityPoint, ...]
    decision_invariance: DecisionInvarianceReportV3
    metrics: RecomputedMetrics
    completeness: EconomicEvidenceCompletenessV3
    operating: OperatingEconomicScenarios
    assessment_status: RecomputedEconomicStatus
    warnings: tuple[str, ...]
    original_exports_immutable: bool = True
    holdout_remains_consumed: bool = True
    unlocks_paper_or_live: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.recomputation_hash, "recomputation hash"),
            (self.original_result_hash, "original result hash"),
            (self.baseline_manifest_hash, "baseline manifest hash"),
            (self.research_plan_hash, "research plan hash"),
            (self.validation_config_hash, "validation config hash"),
            (self.source_hash, "recomputation source hash"),
            (self.evidence_registry_hash, "evidence registry hash"),
            (self.original_cost_config_hash, "original cost config hash"),
            (self.recomputation_config_hash, "recomputation config hash"),
        ):
            _sha(value, name)
        _utc(self.created_at, "recomputation created_at")
        if self.frozen_validation_thresholds != tuple(
            sorted(self.frozen_validation_thresholds)
        ):
            raise RobustnessError("frozen Validation thresholds must be sorted")
        if self.period_label != "CONSUMED_HOLDOUT_ECONOMIC_RECOMPUTATION":
            raise HoldoutGovernanceError("recomputation period label is not consumed")
        if self.holdout_status != HoldoutStatus.CONSUMED.value:
            raise HoldoutGovernanceError("economic recomputation requires CONSUMED holdout")
        if not self.original_exports_immutable or not self.holdout_remains_consumed:
            raise HoldoutGovernanceError("original exports and consumed lifecycle are immutable")
        if self.unlocks_paper_or_live:
            raise RobustnessError("economic recomputation must never unlock Paper or LIVE")


@dataclass(frozen=True, slots=True)
class PaperReadinessReviewV3:
    readiness_id: str
    review_name: str
    review_version: str
    readiness_hash: str
    created_at: datetime
    recomputation_id: str
    original_run_id: str
    holdout_status: str
    status: PaperReadinessStatus
    criteria: tuple[ReadinessCriterionV3, ...]
    warnings: tuple[str, ...]
    human_review_status: HumanReviewStatus
    meaning: str
    next_step: str
    unlocks_paper_or_live: bool = False

    def __post_init__(self) -> None:
        _sha(self.readiness_hash, "Paper readiness V3 hash")
        _utc(self.created_at, "Paper readiness V3 created_at")
        if self.holdout_status != HoldoutStatus.CONSUMED.value:
            raise HoldoutGovernanceError("Paper Readiness V3 cannot freshen a holdout")
        if self.human_review_status is not HumanReviewStatus.AWAITING_HUMAN_REVIEW:
            raise RobustnessError("generated readiness must await explicit human review")
        if self.unlocks_paper_or_live:
            raise RobustnessError("Paper readiness must never unlock execution")


@dataclass(frozen=True, slots=True)
class HumanReviewRecord:
    review_event_id: str
    readiness_id: str
    readiness_hash: str
    status: HumanReviewStatus
    reason: str
    reviewer_context: str
    code_sha: str | None
    recorded_at: datetime
    event_hash: str
    authorizes_lot9_development_only: bool
    unlocks_paper_or_live: bool = False

    def __post_init__(self) -> None:
        _sha(self.readiness_hash, "human review readiness hash")
        _sha(self.event_hash, "human review event hash")
        _utc(self.recorded_at, "human review recorded_at")
        if not self.reason.strip():
            raise RobustnessError("human review requires an explicit reason")
        if self.status is HumanReviewStatus.AWAITING_HUMAN_REVIEW:
            if self.authorizes_lot9_development_only:
                raise RobustnessError("awaiting review cannot authorize development")
        elif self.status is HumanReviewStatus.HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT:
            if not self.authorizes_lot9_development_only:
                raise RobustnessError("accepted review authorizes only Lot 9 development")
        elif self.authorizes_lot9_development_only:
            raise RobustnessError("rejected review cannot authorize Lot 9 development")
        if self.unlocks_paper_or_live:
            raise RobustnessError("human readiness review never unlocks Paper or LIVE")


def _cost_summary(run: BacktestMonitoringData) -> dict[str, Any]:
    costs = run.summary.get("costs")
    if not isinstance(costs, Mapping) or not isinstance(costs.get("summary"), Mapping):
        raise RobustnessError("original run has no trusted cost summary")
    return dict(costs["summary"])


def _metrics_summary(run: BacktestMonitoringData) -> dict[str, Any]:
    metrics = run.summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RobustnessError("original run has no trusted metrics")
    return dict(metrics)


def _market_map() -> dict[str, tuple[str, str]]:
    return {
        item.symbol: (item.market, item.currency)
        for item in load_instruments("balanced", DEFAULT_COST_DIRECTORY)
    }


def _component_amount(payload: object, component: str) -> Decimal:
    mapping = _json_mapping(payload, "cost breakdown")
    item = mapping.get(component)
    if not isinstance(item, Mapping) or item.get("amount") is None:
        raise RobustnessError(f"cost breakdown lacks numeric {component}")
    return _decimal(item["amount"], component)


def _recompute_affected_fills(
    run: BacktestMonitoringData,
    rule_book: Section31RuleBook,
    markets: Mapping[str, tuple[str, str]],
) -> tuple[AffectedFillCost, ...]:
    ledger = {
        str(item.get("reference_id")): item
        for item in run.tables.get("ledger", ())
        if item.get("reference_id")
    }
    affected: list[AffectedFillCost] = []
    for fill in run.tables.get("fills", ()):
        symbol = str(fill.get("symbol"))
        market_currency = markets.get(symbol)
        if market_currency is None:
            raise RobustnessError(f"unknown instrument cost metadata for {symbol}")
        market, _currency = market_currency
        timestamp = _utc(fill.get("timestamp"), "fill timestamp")
        quantity = _decimal(fill.get("quantity"), "fill quantity")
        price = _decimal(fill.get("price"), "fill execution price")
        notional = quantity * price
        result = rule_book.calculate(
            timestamp=timestamp,
            side=str(fill.get("side")),
            market=market,
            notional=notional,
            covered_instrument=market == rule_book.config.market,
            require_period_coverage=(
                str(fill.get("side")) == "SELL" and market == rule_book.config.market
            ),
        )
        if result.amount is None or result.amount <= ZERO:
            continue
        fill_id = str(fill.get("fill_id"))
        ledger_item = ledger.get(fill_id)
        if ledger_item is None:
            raise RobustnessError(f"fill {fill_id} has no immutable ledger entry")
        original_exchange = _decimal(fill.get("exchange_fees"), "exchange fees")
        original_total = _decimal(
            fill.get("total_variable_cost"), "total variable cost"
        )
        original_cash_change = _decimal(
            ledger_item.get("cash_change"), "ledger cash change"
        )
        assert result.evidence_id is not None and result.rate_per_million is not None
        affected.append(
            AffectedFillCost(
                fill_id=fill_id,
                order_id=str(fill.get("order_id")),
                symbol=symbol,
                timestamp=timestamp,
                quantity=quantity,
                execution_price=price,
                notional=notional,
                evidence_id=result.evidence_id,
                rate_per_million=result.rate_per_million,
                original_exchange_fees=original_exchange,
                section31_cost=result.amount,
                recomputed_exchange_fees=original_exchange + result.amount,
                original_total_variable_cost=original_total,
                recomputed_total_variable_cost=original_total + result.amount,
                original_ledger_cash_change=original_cash_change,
                recomputed_ledger_cash_change=original_cash_change - result.amount,
            )
        )
    return tuple(sorted(affected, key=lambda item: (item.timestamp, item.fill_id)))


def _recompute_equity(
    run: BacktestMonitoringData,
    affected: tuple[AffectedFillCost, ...],
) -> tuple[RecomputedEquityPoint, ...]:
    events: dict[datetime, Decimal] = {}
    for item in affected:
        events[item.timestamp] = events.get(item.timestamp, ZERO) + item.section31_cost
    event_rows = tuple(sorted(events.items()))
    cursor = 0
    cumulative = ZERO
    result: list[RecomputedEquityPoint] = []
    for row in sorted(
        run.tables.get("equity", ()), key=lambda item: str(item.get("timestamp"))
    ):
        timestamp = _utc(row.get("timestamp"), "equity timestamp")
        while cursor < len(event_rows) and event_rows[cursor][0] <= timestamp:
            cumulative += event_rows[cursor][1]
            cursor += 1
        cash = _decimal(row.get("cash"), "equity cash")
        equity = _decimal(row.get("equity"), "equity value")
        result.append(
            RecomputedEquityPoint(
                timestamp=timestamp,
                original_cash=cash,
                recomputed_cash=cash - cumulative,
                original_equity=equity,
                recomputed_equity=equity - cumulative,
                cumulative_section31=cumulative,
            )
        )
    if not result:
        raise RobustnessError("equity curve is required for economic recomputation")
    return tuple(result)


def _allocate_trade_costs(
    run: BacktestMonitoringData,
    affected: tuple[AffectedFillCost, ...],
    quantum: Decimal,
) -> tuple[tuple[RecomputedTradeOutcome, ...], tuple[Decimal, ...]]:
    trades = tuple(run.tables.get("trades", ()))
    allocations: dict[str, Decimal] = {}
    for fill in affected:
        matches = tuple(
            item
            for item in trades
            if str(item.get("symbol")) == fill.symbol
            and _utc(item.get("exit_time"), "trade exit_time") == fill.timestamp
        )
        if not matches:
            raise RobustnessError(
                f"affected sell fill {fill.fill_id} cannot be reconciled to a closed trade"
            )
        total_quantity = sum(
            (_decimal(item.get("quantity"), "trade quantity") for item in matches), ZERO
        )
        if total_quantity <= ZERO:
            raise RobustnessError("affected trade quantities must be positive")
        assigned = ZERO
        for index, trade in enumerate(matches):
            trade_id = str(trade.get("trade_id"))
            if index == len(matches) - 1:
                allocation = fill.section31_cost - assigned
            else:
                allocation = (
                    fill.section31_cost
                    * _decimal(trade.get("quantity"), "trade quantity")
                    / total_quantity
                ).quantize(quantum, rounding=ROUND_HALF_EVEN)
                assigned += allocation
            allocations[trade_id] = allocations.get(trade_id, ZERO) + allocation
    outcomes: list[RecomputedTradeOutcome] = []
    all_net: list[Decimal] = []
    for trade in trades:
        trade_id = str(trade.get("trade_id"))
        original = _decimal(trade.get("net_pnl"), "trade net PnL")
        allocation = allocations.get(trade_id, ZERO)
        recomputed = original - allocation
        all_net.append(recomputed)
        if allocation > ZERO:
            outcomes.append(
                RecomputedTradeOutcome(
                    trade_id=trade_id,
                    symbol=str(trade.get("symbol")),
                    exit_time=_utc(trade.get("exit_time"), "trade exit_time"),
                    original_net_pnl=original,
                    section31_allocation=allocation,
                    recomputed_net_pnl=recomputed,
                )
            )
    if sum((item.section31_allocation for item in outcomes), ZERO) != sum(
        (item.section31_cost for item in affected), ZERO
    ):
        raise RobustnessError("Section 31 trade allocation does not reconcile")
    return tuple(sorted(outcomes, key=lambda item: item.trade_id)), tuple(all_net)


def _max_drawdown(values: Iterable[Decimal]) -> Decimal:
    peak: Decimal | None = None
    result = ZERO
    for value in values:
        peak = value if peak is None or value > peak else peak
        if peak > ZERO:
            result = max(result, (peak - value) / peak)
    return result


def _profit_factor(values: Iterable[Decimal]) -> Decimal:
    items = tuple(values)
    gains = sum((value for value in items if value > ZERO), ZERO)
    losses = -sum((value for value in items if value < ZERO), ZERO)
    if losses == ZERO:
        raise RobustnessError("profit factor is undefined without losing trades")
    return gains / losses


def _counterfactual_economic_outcomes(
    run: BacktestMonitoringData,
    rule_book: Section31RuleBook,
    markets: Mapping[str, tuple[str, str]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], bool]:
    estimates = {
        str(item.get("estimate_id")): item
        for item in run.tables.get("cost_estimates", ())
    }
    config = inspect_cost_config("balanced")
    original: list[dict[str, object]] = []
    recomputed: list[dict[str, object]] = []
    changed = False
    for decision in sorted(
        run.tables.get("economic_decisions", ()),
        key=lambda item: str(item.get("decision_id")),
    ):
        estimate = estimates.get(str(decision.get("cost_estimate_id")))
        if estimate is None:
            raise RobustnessError("economic decision has no cost estimate")
        symbol = str(estimate.get("symbol"))
        market_currency = markets.get(symbol)
        if market_currency is None:
            raise RobustnessError(f"unknown instrument metadata for {symbol}")
        market, _currency = market_currency
        timestamp = _utc(estimate.get("timestamp"), "cost estimate timestamp")
        side = str(estimate.get("side"))
        notional = _decimal(estimate.get("reference_price"), "estimate price") * _decimal(
            estimate.get("quantity"), "estimate quantity"
        )
        # A BUY estimates a future SELL at information available now.  A SELL
        # is itself the covered leg, but remains a risk-reducing exit.
        covered_side = "SELL" if side in {"BUY", "SELL"} else side
        regulatory = rule_book.calculate(
            timestamp=timestamp,
            side=covered_side,
            market=market,
            notional=notional,
            covered_instrument=market == rule_book.config.market,
            require_period_coverage=(market == rule_book.config.market),
        )
        added = regulatory.amount or ZERO
        old_cost_bps = (
            _decimal(decision.get("estimated_round_trip_cost_bps"), "round-trip bps")
            if decision.get("estimated_round_trip_cost_bps") is not None
            else None
        )
        new_cost_bps = (
            old_cost_bps + added / notional * Decimal("10000")
            if old_cost_bps is not None and notional > ZERO
            else old_cost_bps
        )
        old_status = str(decision.get("status"))
        old_allows = bool(decision.get("allows_new_risk"))
        gross = (
            _decimal(decision.get("expected_gross_edge_bps"), "gross edge")
            if decision.get("expected_gross_edge_bps") is not None
            else None
        )
        if old_status == "NOT_APPLICABLE":
            new_status, new_allows = old_status, old_allows
        elif gross is None or new_cost_bps is None:
            new_status, new_allows = old_status, old_allows
        else:
            net = gross - new_cost_bps
            ratio = gross / new_cost_bps if new_cost_bps > ZERO else None
            if net < config.minimum_net_edge_bps or (
                ratio is not None and ratio < config.minimum_edge_to_cost_ratio
            ):
                new_status, new_allows = "BLOCK", False
            else:
                new_status, new_allows = "PASS", True
        old_semantic = {
            "decision_id": decision.get("decision_id"),
            "order_id": decision.get("order_id"),
            "status": old_status,
            "allows_new_risk": old_allows,
        }
        new_semantic = {
            **old_semantic,
            "status": new_status,
            "allows_new_risk": new_allows,
        }
        original.append(old_semantic)
        recomputed.append(new_semantic)
        changed = changed or old_semantic != new_semantic
    return tuple(original), tuple(recomputed), changed


def _layer(
    layer: str, original: object, recomputed: object, reason: str
) -> LayerInvariance:
    left = stable_hash(original)
    right = stable_hash(recomputed)
    return LayerInvariance(layer, left, right, left == right, reason)


def _decision_invariance(
    run: BacktestMonitoringData,
    rule_book: Section31RuleBook,
    markets: Mapping[str, tuple[str, str]],
    expected_core_hash: str,
) -> DecisionInvarianceReportV3:
    current_core = decision_core_hash()
    old_economic, new_economic, economic_changed = _counterfactual_economic_outcomes(
        run, rule_book, markets
    )
    layers = tuple(
        sorted(
            (
                _layer(
                    "FEATURE",
                    (
                        tuple(
                            (item.get("signal_id"), item.get("features_used"))
                            for item in run.tables.get("signals", ())
                        ),
                        tuple(
                            (item.get("snapshot_id"), item.get("evidence"))
                            for item in run.tables.get("regime_snapshots", ())
                        ),
                    ),
                    (
                        tuple(
                            (item.get("signal_id"), item.get("features_used"))
                            for item in run.tables.get("signals", ())
                        ),
                        tuple(
                            (item.get("snapshot_id"), item.get("evidence"))
                            for item in run.tables.get("regime_snapshots", ())
                        ),
                    ),
                    "Exported point-in-time feature evidence is immutable and not recalculated.",
                ),
                _layer(
                    "REGIME",
                    run.tables.get("regime_snapshots", ()),
                    run.tables.get("regime_snapshots", ()),
                    "No regime snapshot is recalculated or changed.",
                ),
                _layer(
                    "SIGNAL",
                    run.tables.get("signals", ()),
                    run.tables.get("signals", ()),
                    "Strategy signals remain byte-semantically identical.",
                ),
                _layer(
                    "ML",
                    (
                        run.tables.get("ml_predictions", ()),
                        run.tables.get("ml_decisions", ()),
                    ),
                    (
                        run.tables.get("ml_predictions", ()),
                        run.tables.get("ml_decisions", ()),
                    ),
                    "No ML fitting, prediction, or filtering is performed.",
                ),
                _layer(
                    "ACTIVATION",
                    run.tables.get("activation_decisions", ()),
                    run.tables.get("activation_decisions", ()),
                    "Activation decisions are immutable inputs.",
                ),
                _layer(
                    "PORTFOLIO",
                    (
                        run.tables.get("portfolio_opportunities", ()),
                        run.tables.get("portfolio_decisions", ()),
                        run.tables.get("portfolio_targets", ()),
                        run.tables.get("portfolio_sleeves", ()),
                    ),
                    (
                        run.tables.get("portfolio_opportunities", ()),
                        run.tables.get("portfolio_decisions", ()),
                        run.tables.get("portfolio_targets", ()),
                        run.tables.get("portfolio_sleeves", ()),
                    ),
                    "Portfolio opportunities, decisions, and targets are unchanged.",
                ),
                _layer(
                    "ECONOMIC",
                    old_economic,
                    new_economic,
                    "Only eligibility status/allows-new-risk are decision semantics; revised cost bps are reported separately.",
                ),
                _layer(
                    "RISK",
                    run.tables.get("risk_decisions", ()),
                    run.tables.get("risk_decisions", ()),
                    "Section 31 affects covered SELL proceeds; BUY entry cash requirements and approved quantities are unchanged.",
                ),
                _layer(
                    "ORDERS",
                    run.tables.get("orders", ()),
                    run.tables.get("orders", ()),
                    "No order intent or quantity is replayed or changed.",
                ),
                _layer(
                    "FILLS",
                    run.tables.get("fills", ()),
                    run.tables.get("fills", ()),
                    "Original fills remain immutable; only analytical cost overlays change.",
                ),
            ),
            key=lambda item: item.layer,
        )
    )
    changed_layers = tuple(item.layer for item in layers if not item.invariant)
    status = classify_decision_invariance(
        core_changed=current_core != expected_core_hash,
        economic_changed=economic_changed,
        risk_changed="RISK" in changed_layers,
        order_or_fill_changed=bool({"ORDERS", "FILLS"}.intersection(changed_layers)),
    )
    semantic = {
        "status": status,
        "expected_core_hash": expected_core_hash,
        "current_core_hash": current_core,
        "layers": layers,
        "changed_layers": changed_layers,
    }
    return DecisionInvarianceReportV3(
        **semantic, report_hash=stable_hash(semantic)
    )


def _build_metrics(
    run: BacktestMonitoringData,
    affected: tuple[AffectedFillCost, ...],
    all_trade_net: tuple[Decimal, ...],
    equity: tuple[RecomputedEquityPoint, ...],
) -> RecomputedMetrics:
    summary = _cost_summary(run)
    original_metrics = _metrics_summary(run)
    initial = _decimal(run.summary.get("initial_cash"), "initial cash")
    section31 = sum((item.section31_cost for item in affected), ZERO)
    original_variable = _decimal(summary.get("total_variable_cost"), "variable costs")
    recomputed_variable = original_variable + section31
    original_net = _decimal(
        summary.get("net_trading_pnl_before_operating"), "net PnL"
    )
    recomputed_net = original_net - section31
    original_return = _decimal(
        summary.get("net_return_before_operating"), "net return"
    )
    recomputed_return = recomputed_net / initial
    original_drawdown = abs(
        _decimal(original_metrics.get("max_drawdown_pct"), "max drawdown")
    )
    recomputed_drawdown = _max_drawdown(
        item.recomputed_equity for item in equity
    )
    original_pf = _decimal(original_metrics.get("profit_factor"), "profit factor")
    recomputed_pf = _profit_factor(all_trade_net)
    original_expectancy = _decimal(original_metrics.get("expectancy"), "expectancy")
    recomputed_expectancy = sum(all_trade_net, ZERO) / Decimal(len(all_trade_net))
    original_minimum_cash = min(item.original_cash for item in equity)
    recomputed_minimum_cash = min(item.recomputed_cash for item in equity)
    total_notional = sum(
        (
            _decimal(item.get("price"), "fill price")
            * _decimal(item.get("quantity"), "fill quantity")
            for item in run.tables.get("fills", ())
        ),
        ZERO,
    )
    gross = _decimal(summary.get("gross_trading_pnl"), "gross trading PnL")
    stress = tuple(
        (
            str(multiplier),
            gross - recomputed_variable * multiplier,
        )
        for multiplier in (
            Decimal("1.00"),
            Decimal("1.25"),
            Decimal("1.50"),
            Decimal("2.00"),
        )
    )
    return RecomputedMetrics(
        initial_cash=initial,
        closed_trades=len(all_trade_net),
        affected_fills=len(affected),
        original_section31=ZERO,
        recomputed_section31=section31,
        original_variable_costs=original_variable,
        recomputed_variable_costs=recomputed_variable,
        original_net_pnl_before_operating=original_net,
        recomputed_net_pnl_before_operating=recomputed_net,
        pnl_delta=-section31,
        original_net_return=original_return,
        recomputed_net_return=recomputed_return,
        return_delta=recomputed_return - original_return,
        original_max_drawdown=original_drawdown,
        recomputed_max_drawdown=recomputed_drawdown,
        drawdown_delta=recomputed_drawdown - original_drawdown,
        original_profit_factor=original_pf,
        recomputed_profit_factor=recomputed_pf,
        original_expectancy=original_expectancy,
        recomputed_expectancy=recomputed_expectancy,
        original_minimum_cash=original_minimum_cash,
        recomputed_minimum_cash=recomputed_minimum_cash,
        minimum_cash_delta=recomputed_minimum_cash - original_minimum_cash,
        cost_per_trade=recomputed_variable / Decimal(len(all_trade_net)),
        cost_to_notional=recomputed_variable / total_notional,
        profit_to_cost_ratio=(gross / recomputed_variable if recomputed_variable > ZERO else None),
        stress_results=stress,
    )


def _operating_scenarios(
    scenario: PaperOperatingScenario,
    run: BacktestMonitoringData,
    recomputed_net: Decimal,
) -> OperatingEconomicScenarios:
    equity = tuple(
        sorted(run.tables.get("equity", ()), key=lambda item: str(item.get("timestamp")))
    )
    if not equity:
        raise RobustnessError("operating scenario requires an equity period")
    start = _utc(equity[0].get("timestamp"), "operating period start")
    end = _utc(equity[-1].get("timestamp"), "operating period end")
    days = Decimal(str((end - start).total_seconds())) / Decimal("86400")
    months = days * Decimal("12") / Decimal("365.2425")
    low, central, high = scenario.monthly_totals
    if any(value is None for value in (low, central, high)):
        raise RobustnessError("PAPER operating scenario must have explicit ranges")
    assert low is not None and central is not None and high is not None
    period_low, period_central, period_high = (
        low * months,
        central * months,
        high * months,
    )
    return OperatingEconomicScenarios(
        scenario_id=scenario.scenario_id,
        scenario_hash=scenario.scenario_hash,
        period_months=months,
        operating_low=period_low,
        operating_central=period_central,
        operating_high=period_high,
        net_before_operating=recomputed_net,
        net_after_low=recomputed_net - period_low,
        net_after_central=recomputed_net - period_central,
        net_after_high=recomputed_net - period_high,
        break_even_fixed_monthly=(recomputed_net / months if months > ZERO else ZERO),
    )


def _economic_completeness(
    run: BacktestMonitoringData,
    registry: EvidenceRegistryV2,
    affected: tuple[AffectedFillCost, ...],
    markets: Mapping[str, tuple[str, str]],
) -> EconomicEvidenceCompletenessV3:
    fills = tuple(run.tables.get("fills", ()))
    used_symbols = {str(item.get("symbol")) for item in fills}
    currencies = {
        markets[symbol][1]
        for symbol in used_symbols
        if symbol in markets
    }
    markets_used = {
        markets[symbol][0]
        for symbol in used_symbols
        if symbol in markets
    }
    if len(used_symbols) != len(
        {symbol for symbol in used_symbols if symbol in markets}
    ):
        raise RobustnessError("instrument metadata is incomplete for executed fills")

    commission_evidence = registry.records_for(
        fee_type="BROKER_COMMISSION", market="US", plan="IBKR_PRO_FIXED"
    )
    scope_evidence = registry.records_for(
        fee_type="FIXED_PLAN_FEE_SCOPE", market="US", plan="IBKR_PRO_FIXED"
    )
    section_evidence = registry.records_for(
        fee_type="SEC_SECTION_31", market="US", plan="IBKR_PRO_FIXED"
    )
    fill_times = tuple(_utc(item.get("timestamp"), "fill timestamp") for item in fills)

    def verified_period_coverage(rows: tuple[EvidenceRecord, ...]) -> bool:
        verified_rows = tuple(
            item
            for item in rows
            if item.verification_status is EvidenceVerificationStatus.VERIFIED
        )
        return bool(verified_rows) and all(
            sum(item.effective_from <= timestamp < item.effective_to for item in verified_rows)
            == 1
            for timestamp in fill_times
        )

    statuses = {
        "broker_commission": (
            "VERIFIED_POINT_IN_TIME"
            if verified_period_coverage(commission_evidence)
            else "UNAVAILABLE"
        ),
        "ordinary_exchange_clearing": (
            "VERIFIED_INCLUDED_IN_FIXED_PLAN"
            if verified_period_coverage(scope_evidence)
            else "UNAVAILABLE"
        ),
        "section31": "VERIFIED_APPLIED_POINT_IN_TIME",
        "spread": "KNOWN_PRESERVED_FROM_ORIGINAL_EXECUTION",
        "slippage": "KNOWN_PRESERVED_FROM_ORIGINAL_EXECUTION",
        "transaction_tax": (
            "NOT_APPLICABLE_NO_FRENCH_EXECUTIONS"
            if "FR" not in markets_used
            else "UNAVAILABLE"
        ),
        "fx": (
            "NOT_APPLICABLE_ALL_EXECUTED_IN_BASE_CURRENCY"
            if currencies == {"USD"}
            else "UNAVAILABLE"
        ),
        "financing": "NOT_APPLICABLE_BALANCED_NO_MARGIN",
        "operating_costs": "COMPLETE_ESTIMATED_PAPER_ESTIMATE_V1",
    }
    unresolved = tuple(
        sorted(name for name, status in statuses.items() if status == "UNAVAILABLE")
    )
    evidence_ids = tuple(
        sorted(
            {
                *(item.evidence_id for item in commission_evidence),
                *(item.evidence_id for item in scope_evidence),
                *(item.evidence_id for item in section_evidence),
                *(item.evidence_id for item in affected),
            }
        )
    )
    status = (
        EconomicEvidenceStatus.INCOMPLETE
        if unresolved
        else EconomicEvidenceStatus.COMPLETE_ESTIMATED
    )
    semantic = {
        "status": status,
        "component_statuses": tuple(sorted(statuses.items())),
        "critical_unresolved": unresolved,
        "evidence_ids": evidence_ids,
    }
    return EconomicEvidenceCompletenessV3(
        **semantic,
        completeness_hash=stable_hash(semantic),
    )


class EconomicRecomputationEngine:
    """Pure Lot 8.4 evaluator over one immutable consumed backtest export."""

    recomputation_version = "1.0"

    def recompute(
        self,
        *,
        run: BacktestMonitoringData,
        robustness_bundle: Mapping[str, Any],
        validation: Mapping[str, Any],
        evidence_reassessment: Mapping[str, Any],
        holdout: Any,
        registry: EvidenceRegistryV2,
        operating_scenario: PaperOperatingScenario,
        config: EconomicRecomputationConfig,
        created_at: datetime,
    ) -> EconomicRecomputationReport:
        created_at = _utc(created_at, "recomputation created_at")
        if holdout.status is not HoldoutStatus.CONSUMED:
            raise HoldoutGovernanceError("holdout V2 must remain CONSUMED")
        if holdout.result_hash != run.summary.get("result_hash"):
            raise HoldoutGovernanceError(
                "immutable original run does not match consumed holdout result hash"
            )
        if evidence_reassessment.get("holdout_status") != HoldoutStatus.CONSUMED.value:
            raise HoldoutGovernanceError("evidence reassessment did not preserve CONSUMED")
        if evidence_reassessment.get("mode") != "ECONOMIC_RECOMPUTATION_REQUIRED":
            raise RobustnessError(
                "numeric evidence does not require an economic recomputation"
            )
        if evidence_reassessment.get("evidence_registry_hash") != registry.registry_hash:
            raise RobustnessError("Evidence Registry hash differs from reassessment")
        if robustness_bundle.get("run_id") != run.run_id:
            raise RobustnessError("robustness bundle targets a different run")
        if validation.get("run_id") != run.run_id:
            raise RobustnessError("validation report targets a different run")

        baseline_hash = str(robustness_bundle.get("baseline_manifest_hash"))
        plan_hash = str(robustness_bundle.get("plan_hash"))
        _sha(baseline_hash, "baseline manifest hash")
        _sha(plan_hash, "research plan hash")
        research_baseline = robustness_bundle.get("research_baseline")
        robustness_plan = robustness_bundle.get("robustness_plan")
        if not isinstance(research_baseline, Mapping):
            raise RobustnessError("embedded ResearchBaselineManifest is unavailable")
        if not isinstance(robustness_plan, Mapping):
            raise RobustnessError("embedded RobustnessResearchPlan is unavailable")
        if research_baseline.get("manifest_hash") != baseline_hash:
            raise RobustnessError("embedded ResearchBaselineManifest hash mismatch")
        if robustness_plan.get("plan_hash") != plan_hash:
            raise RobustnessError("embedded RobustnessResearchPlan hash mismatch")
        if robustness_plan.get("baseline_manifest_hash") != baseline_hash:
            raise RobustnessError(
                "RobustnessResearchPlan does not reference the frozen baseline"
            )

        original_costs = run.summary.get("costs")
        if not isinstance(original_costs, Mapping):
            raise RobustnessError("original cost provenance is unavailable")
        original_config_hash = str(original_costs.get("config_hash"))
        _sha(original_config_hash, "original cost config hash")
        validation_config, validation_config_hash = load_validation_config()
        if validation.get("config_hash") != validation_config_hash:
            raise RobustnessError("frozen Validation configuration hash mismatch")
        markets = _market_map()
        rule_book = Section31RuleBook(registry, config)
        affected = _recompute_affected_fills(run, rule_book, markets)
        equity = _recompute_equity(run, affected)
        trades, all_trade_net = _allocate_trade_costs(
            run, affected, config.rounding_quantum
        )
        invariance = _decision_invariance(
            run, rule_book, markets, holdout.expected_core_hash
        )
        metrics = _build_metrics(run, affected, all_trade_net, equity)
        completeness = _economic_completeness(run, registry, affected, markets)
        operating = _operating_scenarios(
            operating_scenario, run, metrics.recomputed_net_pnl_before_operating
        )
        warnings = tuple(
            sorted(
                {
                    "CONSUMED_HOLDOUT_NOT_FRESH_OOS",
                    "RETROSPECTIVE_ECONOMIC_RECOMPUTATION",
                    "SURVIVORSHIP_BIAS_UNRESOLVED",
                    "HISTORICAL_CONCENTRATION_WARNING_PRESERVED",
                    "OPERATING_COSTS_ARE_ESTIMATES_NOT_OBSERVED_INVOICES",
                }
            )
        )
        hard_pass = (
            invariance.status is DecisionInvarianceStatus.STRICTLY_INVARIANT
            and completeness.status is not EconomicEvidenceStatus.INCOMPLETE
            and metrics.recomputed_minimum_cash >= ZERO
            and metrics.closed_trades >= validation_config.minimum_closed_trades
            and metrics.recomputed_net_return > validation_config.minimum_net_return
            and metrics.recomputed_expectancy > validation_config.minimum_net_expectancy
            and metrics.recomputed_profit_factor > validation_config.minimum_profit_factor
            and metrics.recomputed_max_drawdown <= validation_config.maximum_drawdown
        )
        assessment = (
            RecomputedEconomicStatus.PASS
            if hard_pass
            else RecomputedEconomicStatus.FAIL
        )
        original_result_hash = str(run.summary.get("result_hash"))
        validation_id = str(validation.get("validation_id"))
        reassessment_id = str(evidence_reassessment.get("reassessment_id"))
        semantic = {
            "recomputation_version": self.recomputation_version,
            "period_label": config.period_label,
            "original_run_id": run.run_id,
            "original_result_hash": original_result_hash,
            "robustness_report_id": str(robustness_bundle.get("report_id")),
            "validation_id": validation_id,
            "validation_config_hash": validation_config_hash,
            "frozen_validation_thresholds": tuple(
                sorted(
                    {
                        "minimum_closed_trades": str(validation_config.minimum_closed_trades),
                        "minimum_net_return": str(validation_config.minimum_net_return),
                        "minimum_net_expectancy": str(validation_config.minimum_net_expectancy),
                        "minimum_profit_factor": str(validation_config.minimum_profit_factor),
                        "maximum_drawdown": str(validation_config.maximum_drawdown),
                    }.items()
                )
            ),
            "evidence_reassessment_id": reassessment_id,
            "holdout_id": holdout.holdout_id,
            "holdout_status": holdout.status,
            "baseline_manifest_hash": baseline_hash,
            "research_plan_hash": plan_hash,
            "code_sha": detect_git_commit(PROJECT_ROOT),
            "source_hash": _recomputation_source_hash(config),
            "evidence_registry_hash": registry.registry_hash,
            "original_cost_engine_name": str(original_costs.get("engine_name")),
            "original_cost_engine_version": str(original_costs.get("engine_version")),
            "original_cost_config_hash": original_config_hash,
            "recomputation_engine_name": config.engine_name,
            "recomputation_engine_version": config.engine_version,
            "recomputation_config_hash": config.config_hash,
            "affected_fills": affected,
            "affected_trades": trades,
            "recomputed_equity": equity,
            "decision_invariance": invariance,
            "metrics": metrics,
            "completeness": completeness,
            "operating": operating,
            "assessment_status": assessment,
            "warnings": warnings,
            "original_exports_immutable": True,
            "holdout_remains_consumed": True,
            "unlocks_paper_or_live": False,
        }
        recomputation_hash = stable_hash(semantic)
        return EconomicRecomputationReport(
            recomputation_id=f"economic-recomputation-{recomputation_hash[:24]}",
            recomputation_hash=recomputation_hash,
            created_at=created_at,
            **semantic,
        )


def _recomputation_source_hash(config: EconomicRecomputationConfig) -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        DEFAULT_RECOMPUTATION_CONFIG,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(config.config_hash.encode("ascii"))
    return digest.hexdigest()


class PaperReadinessReviewerV3:
    """Read-only review using frozen thresholds; it cannot unlock execution."""

    review_name = "balanced-paper-readiness"
    review_version = "3.0"

    def review(
        self,
        report: EconomicRecomputationReport,
        *,
        created_at: datetime,
    ) -> PaperReadinessReviewV3:
        created_at = _utc(created_at, "Paper readiness created_at")
        metrics = report.metrics
        frozen, frozen_hash = load_validation_config()
        if frozen_hash != report.validation_config_hash:
            raise RobustnessError("Paper readiness cannot change frozen Validation thresholds")
        checks = (
            ("holdout_governance", report.holdout_remains_consumed, "CONSUMED", report.holdout_status),
            ("decision_core_and_chain", report.decision_invariance.status is DecisionInvarianceStatus.STRICTLY_INVARIANT, "STRICTLY_INVARIANT", report.decision_invariance.status.value),
            ("closed_trades", metrics.closed_trades >= frozen.minimum_closed_trades, f">= {frozen.minimum_closed_trades}", str(metrics.closed_trades)),
            ("net_return", metrics.recomputed_net_return > frozen.minimum_net_return, f"> {frozen.minimum_net_return}", str(metrics.recomputed_net_return)),
            ("net_expectancy", metrics.recomputed_expectancy > frozen.minimum_net_expectancy, f"> {frozen.minimum_net_expectancy}", str(metrics.recomputed_expectancy)),
            ("profit_factor", metrics.recomputed_profit_factor > frozen.minimum_profit_factor, f"> {frozen.minimum_profit_factor}", str(metrics.recomputed_profit_factor)),
            ("max_drawdown", metrics.recomputed_max_drawdown <= frozen.maximum_drawdown, f"<= {frozen.maximum_drawdown}", str(metrics.recomputed_max_drawdown)),
            ("cash_non_negative", metrics.recomputed_minimum_cash >= ZERO, ">= 0", str(metrics.recomputed_minimum_cash)),
            ("transaction_cost_evidence", report.completeness.status is not EconomicEvidenceStatus.INCOMPLETE, "COMPLETE_VERIFIED|CONSERVATIVE|ESTIMATED", report.completeness.status.value),
            ("section31_integrated", metrics.affected_fills > 0 and metrics.recomputed_section31 > ZERO, "verified point-in-time fee applied", f"{metrics.affected_fills} fills / {metrics.recomputed_section31}"),
            ("operating_scenario", report.operating.scenario_id == "PAPER_ESTIMATE_V1", "PAPER_ESTIMATE_V1 present", report.operating.scenario_id),
            ("execution_locked", not report.unlocks_paper_or_live, "no Paper/LIVE unlock", str(report.unlocks_paper_or_live)),
        )
        criteria = tuple(
            ReadinessCriterionV3(
                name=name,
                status="PASS" if passed else "FAIL",
                observed=observed,
                required=required,
                reason=(
                    "Frozen Lot 8.1/8.4 research requirement satisfied."
                    if passed
                    else "Frozen Paper-readiness requirement is not satisfied."
                ),
            )
            for name, passed, required, observed in checks
        )
        failed = tuple(item.name for item in criteria if item.status == "FAIL")
        if not failed:
            status = PaperReadinessStatus.READY_FOR_REVIEW
            meaning = (
                "Evidence is sufficient for a human review of Lot 9 development only; "
                "this is neither Live readiness nor permission to connect Paper."
            )
        elif report.completeness.status is EconomicEvidenceStatus.INCOMPLETE:
            status = PaperReadinessStatus.INSUFFICIENT_EVIDENCE
            meaning = "Critical economic evidence remains insufficient for review."
        else:
            status = PaperReadinessStatus.NOT_READY
            meaning = "One or more frozen hard Paper-readiness requirements failed."
        warnings = tuple(sorted(set(report.warnings)))
        semantic = {
            "review_name": self.review_name,
            "review_version": self.review_version,
            "recomputation_id": report.recomputation_id,
            "original_run_id": report.original_run_id,
            "holdout_status": report.holdout_status,
            "status": status,
            "criteria": criteria,
            "warnings": warnings,
            "human_review_status": HumanReviewStatus.AWAITING_HUMAN_REVIEW,
            "meaning": meaning,
            "next_step": (
                "Human review is required in a separate audited CLI action; no execution capability is enabled."
            ),
            "unlocks_paper_or_live": False,
        }
        digest = stable_hash(semantic)
        return PaperReadinessReviewV3(
            readiness_id=f"paper-readiness-v3-{digest[:24]}",
            readiness_hash=digest,
            created_at=created_at,
            **semantic,
        )


def make_initial_human_review(
    readiness: PaperReadinessReviewV3,
    *,
    recorded_at: datetime,
) -> HumanReviewRecord:
    recorded_at = _utc(recorded_at, "human review recorded_at")
    semantic = {
        "readiness_id": readiness.readiness_id,
        "readiness_hash": readiness.readiness_hash,
        "status": HumanReviewStatus.AWAITING_HUMAN_REVIEW,
        "reason": "No human readiness decision has been recorded.",
        "reviewer_context": "LOCAL_CLI_AUDIT",
        "code_sha": detect_git_commit(PROJECT_ROOT),
        "authorizes_lot9_development_only": False,
        "unlocks_paper_or_live": False,
    }
    digest = stable_hash(semantic)
    return HumanReviewRecord(
        review_event_id=f"human-review-{digest[:24]}",
        event_hash=digest,
        recorded_at=recorded_at,
        **semantic,
    )


def make_human_review_decision(
    readiness: PaperReadinessReviewV3,
    *,
    status: HumanReviewStatus,
    reason: str,
    recorded_at: datetime,
) -> HumanReviewRecord:
    if status is HumanReviewStatus.AWAITING_HUMAN_REVIEW:
        raise RobustnessError("an explicit human decision must accept or reject")
    if not reason.strip():
        raise RobustnessError("human review requires --reason")
    if (
        status is HumanReviewStatus.HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT
        and readiness.status is not PaperReadinessStatus.READY_FOR_REVIEW
    ):
        raise RobustnessError("only READY_FOR_REVIEW may be accepted for Lot 9 development")
    semantic = {
        "readiness_id": readiness.readiness_id,
        "readiness_hash": readiness.readiness_hash,
        "status": status,
        "reason": reason.strip(),
        "reviewer_context": "LOCAL_CLI_AUDIT",
        "code_sha": detect_git_commit(PROJECT_ROOT),
        "authorizes_lot9_development_only": (
            status is HumanReviewStatus.HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT
        ),
        "unlocks_paper_or_live": False,
    }
    digest = stable_hash(semantic)
    return HumanReviewRecord(
        review_event_id=f"human-review-{digest[:24]}",
        event_hash=digest,
        recorded_at=_utc(recorded_at, "human review recorded_at"),
        **semantic,
    )


class EconomicRecomputationService:
    """Checksum-verifying application facade for recompute/inspect/review CLI."""

    def __init__(self, data_root: Path | str = Path("data_local")) -> None:
        self.data_root = Path(data_root)
        self.monitoring = BacktestMonitoringSource(self.data_root / "backtests")
        self.store = LocalRobustnessStore(self.data_root / "robustness")
        self.validation_store = LocalValidationStore(self.data_root / "validation")

    def recompute(
        self,
        *,
        run_id: str,
        robustness_report_id: str | None = None,
        evidence_reassessment_id: str | None = None,
        operating_scenario_id: str = "PAPER_ESTIMATE_V1",
    ) -> tuple[EconomicRecomputationReport, PaperReadinessReviewV3, HumanReviewRecord]:
        run = self.monitoring.load_run(run_id)
        robustness = (
            self.store.inspect_report_bundle(robustness_report_id)
            if robustness_report_id
            else self.store.latest_report_bundle_for_run(run_id)
        )
        if robustness is None:
            raise RobustnessError("no checksum-verified robustness report for run")
        evidence = (
            self.store.inspect_evidence_bundle(evidence_reassessment_id)
            if evidence_reassessment_id
            else self.store.latest_evidence_for_run(run_id)
        )
        if evidence is None:
            raise RobustnessError("no checksum-verified evidence reassessment for run")
        validation = self.validation_store.inspect(str(evidence.get("validation_id")))
        holdout = deserialize_holdout(
            self.store.inspect_holdout(str(evidence.get("holdout_id")))
        )
        registry = load_evidence_registry_v2()
        scenarios = {
            item.scenario_id: item for item in load_paper_operating_scenarios()
        }
        try:
            operating = scenarios[operating_scenario_id]
        except KeyError as exc:
            raise RobustnessError("unknown PAPER operating scenario") from exc
        config = load_economic_recomputation_config()
        now = datetime.now(timezone.utc)
        original_fingerprint = run.source_fingerprint
        report = EconomicRecomputationEngine().recompute(
            run=run,
            robustness_bundle=robustness,
            validation=validation,
            evidence_reassessment=evidence,
            holdout=holdout,
            registry=registry,
            operating_scenario=operating,
            config=config,
            created_at=now,
        )
        readiness = PaperReadinessReviewerV3().review(report, created_at=now)
        human = make_initial_human_review(readiness, recorded_at=now)
        if self.monitoring.load_run(run_id).source_fingerprint != original_fingerprint:
            raise RobustnessError("original backtest changed during recomputation")
        self.store.save_recomputation_bundle(
            report=report,
            readiness=readiness,
            human_review=human,
            evidence_registry=registry,
        )
        return report, readiness, human

    def inspect(self, recomputation_id: str) -> dict[str, Any]:
        return self.store.inspect_recomputation_bundle(recomputation_id)

    def latest_for_run(self, run_id: str) -> dict[str, Any] | None:
        return self.store.latest_recomputation_for_run(run_id)

    def human_review_status(self, readiness_id: str) -> dict[str, Any]:
        latest = self.store.latest_human_review(readiness_id)
        if latest is not None:
            return latest
        return dict(
            self.store.find_recomputation_by_readiness(readiness_id)["human_review"]
        )

    def record_human_review(
        self,
        *,
        readiness_id: str,
        decision: str,
        reason: str,
    ) -> HumanReviewRecord:
        bundle = self.store.find_recomputation_by_readiness(readiness_id)
        readiness = deserialize_paper_readiness_v3(bundle["paper_readiness_v3"])
        status = {
            "accept-for-lot9-development": HumanReviewStatus.HUMAN_REVIEW_ACCEPTED_FOR_LOT9_DEVELOPMENT,
            "reject": HumanReviewStatus.HUMAN_REVIEW_REJECTED,
        }.get(decision)
        if status is None:
            raise RobustnessError("human review decision must accept or reject")
        record = make_human_review_decision(
            readiness,
            status=status,
            reason=reason,
            recorded_at=datetime.now(timezone.utc),
        )
        self.store.save_human_review(record)
        return record


def deserialize_paper_readiness_v3(payload: Mapping[str, Any]) -> PaperReadinessReviewV3:
    return PaperReadinessReviewV3(
        readiness_id=str(payload["readiness_id"]),
        review_name=str(payload["review_name"]),
        review_version=str(payload["review_version"]),
        readiness_hash=str(payload["readiness_hash"]),
        created_at=_utc(payload["created_at"], "readiness created_at"),
        recomputation_id=str(payload["recomputation_id"]),
        original_run_id=str(payload["original_run_id"]),
        holdout_status=str(payload["holdout_status"]),
        status=PaperReadinessStatus(str(payload["status"])),
        criteria=tuple(
            ReadinessCriterionV3(
                name=str(item["name"]),
                status=str(item["status"]),
                observed=str(item["observed"]),
                required=str(item["required"]),
                reason=str(item["reason"]),
            )
            for item in payload.get("criteria", ())
        ),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
        human_review_status=HumanReviewStatus(str(payload["human_review_status"])),
        meaning=str(payload["meaning"]),
        next_step=str(payload["next_step"]),
        unlocks_paper_or_live=bool(payload.get("unlocks_paper_or_live", False)),
    )


__all__ = [
    "AffectedFillCost",
    "DecisionInvarianceReportV3",
    "DecisionInvarianceStatus",
    "EconomicEvidenceCompletenessV3",
    "EconomicRecomputationConfig",
    "EconomicRecomputationEngine",
    "EconomicRecomputationReport",
    "EconomicRecomputationService",
    "HumanReviewRecord",
    "HumanReviewStatus",
    "PaperReadinessReviewV3",
    "PaperReadinessReviewerV3",
    "RecomputedEconomicStatus",
    "RegulatoryFeeResult",
    "Section31RuleBook",
    "classify_decision_invariance",
    "deserialize_paper_readiness_v3",
    "load_economic_recomputation_config",
    "make_human_review_decision",
    "make_initial_human_review",
]
