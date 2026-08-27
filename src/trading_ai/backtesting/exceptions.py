"""Backtesting-specific failures with no broker or provider leakage."""


class BacktestError(Exception):
    """Base class for all historical-simulation failures."""


class BacktestConfigurationError(BacktestError, ValueError):
    """Backtest assumptions conflict with the selected trading profile."""


class BacktestDataError(BacktestError):
    """Historical inputs are missing, invalid, or not approved by policy."""


class BacktestExecutionError(BacktestError):
    """A simulated order or fill violates deterministic execution rules."""


class BacktestStorageError(BacktestError):
    """A result export cannot be written or inspected safely."""
