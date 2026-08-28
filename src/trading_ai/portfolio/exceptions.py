"""Portfolio-domain exceptions that do not leak provider or broker failures."""

from trading_ai.core.exceptions import TradingAIError


class PortfolioError(TradingAIError):
    """Base class for Lot 7 portfolio failures."""


class PortfolioConfigurationError(PortfolioError):
    """Raised when portfolio configuration is invalid or exceeds hard limits."""


class PortfolioPlanningError(PortfolioError):
    """Raised when a deterministic plan cannot be built safely."""


class CurrencyConversionError(PortfolioPlanningError):
    """Raised when a required point-in-time FX conversion is unavailable."""
