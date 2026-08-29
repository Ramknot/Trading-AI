"""Domain errors for local monitoring and dashboard reads."""

from trading_ai.core.exceptions import TradingAIError


class MonitoringError(TradingAIError):
    """Base error exposed by observability boundaries."""


class MonitoringNotFoundError(MonitoringError):
    """A requested local run or monitoring record does not exist."""


class MonitoringIntegrityError(MonitoringError):
    """A source failed integrity verification and must not be trusted."""


class MonitoringConfigurationError(MonitoringError):
    """Monitoring or dashboard configuration is unsafe or invalid."""
