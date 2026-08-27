"""Feature Engine domain exceptions."""


class FeatureError(Exception):
    """Base class for deterministic feature-calculation failures."""


class FeatureInputError(FeatureError):
    """Raised when normalized input violates the Feature Engine contract."""
