"""Local, read-only monitoring contracts and Dashboard foundation."""

from trading_ai.monitoring.base import EventMonitoringSource, MonitoringSource, MonitoringStore
from trading_ai.monitoring.costs import build_cost_snapshot
from trading_ai.monitoring.dashboard import (
    DashboardSettings,
    build_monitoring_service,
    create_dashboard_app,
    serve_dashboard,
)
from trading_ai.monitoring.models import (
    CostComponent,
    CostCoverageStatus,
    CostKnowledge,
    CostSnapshot,
    DecisionTrace,
    DecisionTraceStep,
    HealthComponent,
    HealthSnapshot,
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSnapshot,
    OperatingCostBreakdown,
    SystemStatus,
    TradingCostBreakdown,
)
from trading_ai.monitoring.service import MonitoringService
from trading_ai.monitoring.source import BacktestMonitoringData, BacktestMonitoringSource
from trading_ai.monitoring.store import SQLiteMonitoringStore

__all__ = [
    "BacktestMonitoringData",
    "BacktestMonitoringSource",
    "CostComponent",
    "CostCoverageStatus",
    "CostKnowledge",
    "CostSnapshot",
    "DashboardSettings",
    "DecisionTrace",
    "DecisionTraceStep",
    "EventMonitoringSource",
    "HealthComponent",
    "HealthSnapshot",
    "MonitoringEvent",
    "MonitoringEventType",
    "MonitoringService",
    "MonitoringSnapshot",
    "MonitoringSource",
    "MonitoringStore",
    "OperatingCostBreakdown",
    "SQLiteMonitoringStore",
    "SystemStatus",
    "TradingCostBreakdown",
    "build_cost_snapshot",
    "build_monitoring_service",
    "create_dashboard_app",
    "serve_dashboard",
]
