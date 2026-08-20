"""Project-specific errors with explicit safety semantics."""


class TradingAIError(Exception):
    """Base class for Trading AI errors."""


class ConfigurationError(TradingAIError, ValueError):
    """Raised when configuration is missing, malformed, or unsafe."""


class ProfileDisabledError(ConfigurationError):
    """Raised when a disabled or locked profile is requested."""


class LiveTradingLockedError(ConfigurationError):
    """Raised whenever Lot 0 is asked to start in LIVE."""
