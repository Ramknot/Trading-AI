"""Lot 8.2 frozen research governance and robustness public API."""

from trading_ai.robustness.config import (
    RobustnessConfig,
    load_research_baseline_manifest,
    load_research_plan,
    load_robustness_config,
)
from trading_ai.robustness.diagnostics import (
    ConcentrationAnalyzer,
    DecisionFunnelAnalyzer,
    DrawdownAnalyzer,
    HistoricalCoverageAnalyzer,
    RobustnessAnalyzer,
    StatisticalUncertaintyAnalyzer,
    TemporalAnalyzer,
)
from trading_ai.robustness.cost_evidence import (
    HistoricalCostEvidenceRegistry,
    load_historical_cost_evidence,
)
from trading_ai.robustness.exceptions import *  # noqa: F403
from trading_ai.robustness.governance import (
    BaselineReproducer,
    HoldoutAccessPolicy,
    HoldoutConsumer,
    consume_holdout,
    decision_core_hash,
    make_untouched_holdout,
    observed_decision_config_hashes,
)
from trading_ai.robustness.models import *  # noqa: F403
from trading_ai.robustness.readiness import PaperReadinessReviewer
from trading_ai.robustness.evidence import (
    EvidenceRegistryV2,
    TariffEvidenceComparator,
    load_evidence_registry_v2,
    load_paper_operating_scenarios,
)
from trading_ai.robustness.reassessment import (
    EvidenceClosureService,
    EvidenceReassessmentEngine,
    PaperReadinessReviewerV2,
)
from trading_ai.robustness.service import RobustnessService
from trading_ai.robustness.storage import (
    LocalRobustnessStore,
    ROBUSTNESS_EXPORT_SCHEMA_VERSION,
)

__all__ = [
    "BaselineReproducer",
    "ConcentrationAnalyzer",
    "DecisionFunnelAnalyzer",
    "DrawdownAnalyzer",
    "EvidenceClosureService",
    "EvidenceRegistryV2",
    "EvidenceReassessmentEngine",
    "HistoricalCoverageAnalyzer",
    "HistoricalCostEvidenceRegistry",
    "HoldoutAccessPolicy",
    "HoldoutConsumer",
    "LocalRobustnessStore",
    "PaperReadinessReviewer",
    "PaperReadinessReviewerV2",
    "ROBUSTNESS_EXPORT_SCHEMA_VERSION",
    "RobustnessAnalyzer",
    "RobustnessConfig",
    "RobustnessService",
    "StatisticalUncertaintyAnalyzer",
    "TemporalAnalyzer",
    "consume_holdout",
    "decision_core_hash",
    "load_research_baseline_manifest",
    "load_historical_cost_evidence",
    "load_evidence_registry_v2",
    "load_paper_operating_scenarios",
    "load_research_plan",
    "load_robustness_config",
    "make_untouched_holdout",
    "observed_decision_config_hashes",
    "TariffEvidenceComparator",
]
