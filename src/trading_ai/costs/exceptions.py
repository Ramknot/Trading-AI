"""Domain exceptions for transaction-cost economics and validation."""


class CostError(Exception):
    """Base exception for deterministic cost-domain failures."""


class CostConfigurationError(CostError):
    """Raised when a cost, tariff, tax, or instrument configuration is invalid."""


class CostCoverageError(CostError):
    """Raised when a critical cost cannot be established without guessing."""


class EconomicValidationError(CostError):
    """Raised when an economic-validation request is structurally invalid."""
