"""Domain exceptions for deterministic regime classification and policy."""


class RegimeError(Exception):
    """Base exception for Lot 5 regime components."""


class RegimeConfigurationError(RegimeError, ValueError):
    """A regime or activation-policy configuration is invalid or locked."""


class RegimeInputError(RegimeError, ValueError):
    """Point-in-time feature or signal inputs are invalid."""
