"""Fail-closed Balanced Risk Engine exceptions."""

from trading_ai.core.exceptions import TradingAIError


class RiskError(TradingAIError):
    """Base class for broker- and provider-independent risk failures."""


class RiskConfigurationError(RiskError, ValueError):
    """Risk configuration is malformed, disabled, or exceeds its profile."""


class RiskContextError(RiskError, ValueError):
    """A risk decision lacks valid point-in-time inputs."""
