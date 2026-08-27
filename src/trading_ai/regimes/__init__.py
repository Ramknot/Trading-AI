"""Deterministic two-axis regime detection and strategy eligibility."""

from trading_ai.regimes.base import ActivationPolicy, RegimeDetector
from trading_ai.regimes.config import (
    BalancedRegimeConfig,
    BalancedStrategyPolicyConfig,
    StrategyPolicyRule,
    VolatilityPolicyOverlay,
    inspect_regime_config,
    inspect_strategy_policy_config,
    load_balanced_regime_config,
    load_balanced_strategy_policy_config,
    regime_config_hash,
    strategy_policy_config_hash,
)
from trading_ai.regimes.detector import BalancedRegimeDetector
from trading_ai.regimes.exceptions import (
    RegimeConfigurationError,
    RegimeError,
    RegimeInputError,
)
from trading_ai.regimes.models import (
    ActivationDecision,
    ActivationStatus,
    RegimeReport,
    RegimeSnapshot,
    RegimeTransition,
    StructureRegime,
    VolatilityRegime,
)
from trading_ai.regimes.policy import BalancedStrategyActivationPolicy
from trading_ai.regimes.reporting import build_regime_report

__all__ = [
    "ActivationDecision",
    "ActivationPolicy",
    "ActivationStatus",
    "BalancedRegimeConfig",
    "BalancedRegimeDetector",
    "BalancedStrategyActivationPolicy",
    "BalancedStrategyPolicyConfig",
    "RegimeConfigurationError",
    "RegimeDetector",
    "RegimeError",
    "RegimeInputError",
    "RegimeReport",
    "RegimeSnapshot",
    "RegimeTransition",
    "StrategyPolicyRule",
    "StructureRegime",
    "VolatilityPolicyOverlay",
    "VolatilityRegime",
    "build_regime_report",
    "inspect_regime_config",
    "inspect_strategy_policy_config",
    "load_balanced_regime_config",
    "load_balanced_strategy_policy_config",
    "regime_config_hash",
    "strategy_policy_config_hash",
]
