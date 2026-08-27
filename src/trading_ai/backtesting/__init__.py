"""Deterministic, offline historical simulation public API."""

from trading_ai.backtesting.base import Backtester
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.exceptions import (
    BacktestConfigurationError,
    BacktestDataError,
    BacktestError,
    BacktestExecutionError,
    BacktestStorageError,
)
from trading_ai.backtesting.execution import (
    BarExecutionModel,
    CommissionModel,
    ConfigurableCommissionModel,
    ExecutionModel,
)
from trading_ai.backtesting.input import load_cached_dataset, memory_dataset
from trading_ai.backtesting.metrics import MetricsEngine, annualization_factor
from trading_ai.backtesting.models import (
    BacktestConfig,
    BacktestDataset,
    BacktestMetrics,
    BacktestOrder,
    BenchmarkResult,
    CommissionConfig,
    DataQualityPolicy,
    DatasetReference,
    EquityPoint,
    ExecutionPolicy,
    Fill,
    LedgerEntry,
    OrderIntent,
    OrderStatus,
    PricePolicy,
    StrategyContext,
    Trade,
)
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.backtesting.strategy import BacktestStrategy, BuyAndHoldDemoStrategy

__all__ = [
    "BacktestConfig",
    "BacktestConfigurationError",
    "BacktestDataError",
    "BacktestDataset",
    "BacktestEngine",
    "BacktestError",
    "BacktestExecutionError",
    "BacktestMetrics",
    "BacktestOrder",
    "BacktestResultStore",
    "BacktestStorageError",
    "BacktestStrategy",
    "Backtester",
    "BarExecutionModel",
    "BenchmarkResult",
    "BuyAndHoldDemoStrategy",
    "CommissionConfig",
    "CommissionModel",
    "ConfigurableCommissionModel",
    "DataQualityPolicy",
    "DatasetReference",
    "EquityPoint",
    "ExecutionModel",
    "ExecutionPolicy",
    "Fill",
    "LedgerEntry",
    "MetricsEngine",
    "OrderIntent",
    "OrderStatus",
    "PricePolicy",
    "StrategyContext",
    "Trade",
    "annualization_factor",
    "load_cached_dataset",
    "memory_dataset",
]
