"""Expected-edge provenance and a monotone economic eligibility gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from statistics import mean

from trading_ai.core.hashing import stable_hash
from trading_ai.costs.config import BalancedCostConfig
from trading_ai.costs.models import (
    BPS,
    CostCoverage,
    EconomicDecision,
    EconomicDecisionStatus,
    EdgeStatus,
    ExpectedEdgeEstimate,
    PreTradeCostEstimate,
)


@dataclass(frozen=True, slots=True)
class HistoricalEdgeObservation:
    timestamp: datetime
    strategy_name: str
    timeframe: str
    gross_return_bps: Decimal
    period: str

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("edge observation timestamp must be timezone-aware")
        if self.period not in {"TRAIN", "VALIDATION"}:
            raise ValueError("HistoricalEdgeEstimator must never consume FINAL TEST")


class HistoricalEdgeEstimator:
    """Research-only mean edge from prior TRAIN/VALIDATION observations."""

    def __init__(self, minimum_samples: int = 30) -> None:
        if minimum_samples < 2:
            raise ValueError("minimum edge samples must be at least two")
        self.minimum_samples = minimum_samples

    def estimate(
        self,
        observations: tuple[HistoricalEdgeObservation, ...],
        *,
        as_of: datetime,
        strategy_name: str,
        timeframe: str,
        horizon_bars: int,
    ) -> ExpectedEdgeEstimate:
        eligible = tuple(
            item for item in observations
            if item.timestamp < as_of
            and item.strategy_name == strategy_name
            and item.timeframe == timeframe
        )
        if len(eligible) < self.minimum_samples:
            return ExpectedEdgeEstimate.unavailable(
                timestamp=as_of,
                strategy_name=strategy_name,
                timeframe=timeframe,
                reason="insufficient prior TRAIN/VALIDATION samples",
            )
        digest = stable_hash((eligible, horizon_bars, self.minimum_samples))
        return ExpectedEdgeEstimate(
            edge_id=f"historical-edge-{digest[:24]}",
            timestamp=as_of,
            strategy_name=strategy_name,
            timeframe=timeframe,
            status=EdgeStatus.AVAILABLE,
            expected_gross_edge_bps=Decimal(str(mean(item.gross_return_bps for item in eligible))),
            horizon_bars=horizon_bars,
            sample_count=len(eligible),
            validation_start=min(item.timestamp for item in eligible),
            validation_end=max(item.timestamp for item in eligible),
            source="prior TRAIN/VALIDATION signal outcomes only",
            provenance_hash=digest,
        )


class EconomicGate:
    """Cost-versus-edge check that cannot authorize execution or override Risk."""

    gate_name = "balanced-economic-gate"
    gate_version = "1.0"

    def __init__(self, config: BalancedCostConfig, config_hash: str) -> None:
        self.config = config
        self.config_hash = config_hash

    def evaluate(
        self,
        *,
        estimate: PreTradeCostEstimate,
        edge: ExpectedEdgeEstimate,
        signal_id: str | None,
        is_risk_reducing_exit: bool,
    ) -> EconomicDecision:
        if edge.timestamp != estimate.timestamp:
            raise ValueError("edge and cost estimate must use the exact decision timestamp")
        gross = edge.expected_gross_edge_bps
        round_trip_amount = estimate.round_trip_costs.amount_if_complete
        notional = estimate.reference_price * estimate.quantity
        cost_bps = (
            round_trip_amount / notional * BPS
            if round_trip_amount is not None and notional > 0 else None
        )
        net = gross - cost_bps if gross is not None and cost_bps is not None else None
        ratio = (
            gross / cost_bps
            if gross is not None and cost_bps is not None and cost_bps > 0 else None
        )
        if is_risk_reducing_exit:
            status = EconomicDecisionStatus.NOT_APPLICABLE
            codes = ("RISK_REDUCING_EXIT",)
            reasons = ("Economic filtering never blocks a risk-reducing exit.",)
            allows = True
        elif estimate.round_trip_costs.coverage is CostCoverage.INCOMPLETE:
            status = EconomicDecisionStatus.INCOMPLETE
            codes = ("COST_COVERAGE_INCOMPLETE",)
            reasons = ("Critical round-trip costs are unavailable; new risk fails closed.",)
            allows = False
        elif edge.status is EdgeStatus.UNAVAILABLE:
            status = EconomicDecisionStatus.INCOMPLETE
            codes = ("EXPECTED_EDGE_UNAVAILABLE",)
            reasons = ("No validated expected-edge estimate exists; result remains research-incomplete.",)
            allows = self.config.missing_edge_policy == "INCOMPLETE_ALLOW_RESEARCH"
        elif net is None:
            status = EconomicDecisionStatus.INCOMPLETE
            codes = ("ECONOMIC_INPUT_INCOMPLETE",)
            reasons = ("Net edge could not be calculated without guessing.",)
            allows = False
        elif net < self.config.minimum_net_edge_bps:
            status = EconomicDecisionStatus.BLOCK
            codes = ("ECONOMICALLY_UNVIABLE", "MINIMUM_NET_EDGE")
            reasons = ("Expected gross edge does not clear round-trip costs and minimum net edge.",)
            allows = False
        elif ratio is not None and ratio < self.config.minimum_edge_to_cost_ratio:
            status = EconomicDecisionStatus.BLOCK
            codes = ("ECONOMICALLY_UNVIABLE", "EDGE_TO_COST_RATIO")
            reasons = ("Expected edge-to-cost ratio is below the configured research floor.",)
            allows = False
        else:
            status = EconomicDecisionStatus.PASS
            codes = ("ECONOMIC_GATE_PASS",)
            reasons = ("Validated edge estimate clears configured cost thresholds.",)
            allows = True
        digest = stable_hash((estimate.estimate_id, edge.edge_id, status, net, ratio, self.config_hash))
        return EconomicDecision(
            decision_id=f"economic-decision-{digest[:24]}",
            timestamp=estimate.timestamp,
            order_id=estimate.order_id,
            signal_id=signal_id,
            cost_estimate_id=estimate.estimate_id,
            expected_edge_id=edge.edge_id,
            status=status,
            expected_gross_edge_bps=gross,
            estimated_round_trip_cost_bps=cost_bps,
            expected_net_edge_bps=net,
            edge_to_cost_ratio=ratio,
            reason_codes=codes,
            human_reasons=reasons,
            gate_name=self.gate_name,
            gate_version=self.gate_version,
            config_hash=self.config_hash,
            allows_new_risk=allows,
        )
