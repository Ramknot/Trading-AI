"""Strategy abstractions; strategies cannot authorize or transmit orders."""

from trading_ai.strategies.base import Strategy
from trading_ai.strategies.baselines import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from trading_ai.strategies.config import (
    BreakoutConfig,
    MeanReversionConfig,
    MomentumConfig,
    TrendConfig,
)
from trading_ai.strategies.registry import BASELINE_STRATEGIES, StrategyRegistry
from trading_ai.strategies.reporting import (
    StrategyReport,
    compare_reports,
    strategy_report,
)
from trading_ai.strategies.sizing import BaselineSizer

__all__ = [
    "BASELINE_STRATEGIES",
    "BaselineSizer",
    "BreakoutConfig",
    "BreakoutStrategy",
    "MeanReversionConfig",
    "MeanReversionStrategy",
    "MomentumConfig",
    "MomentumStrategy",
    "Strategy",
    "StrategyRegistry",
    "StrategyReport",
    "TrendConfig",
    "TrendFollowingStrategy",
    "compare_reports",
    "strategy_report",
]
