"""Business exceptions for offline ML training, inference, and storage."""


class MLError(Exception):
    """Base class for all ML platform failures."""


class MLConfigurationError(MLError):
    """Raised when an ML configuration is internally inconsistent."""


class MLDataError(MLError):
    """Raised when training or inference data is incomplete or invalid."""


class MLTrainingError(MLError):
    """Raised when a model cannot be trained safely."""


class MLInferenceError(MLError):
    """Raised when a frozen artifact cannot produce a valid score."""


class MLRegistryError(MLError):
    """Raised for missing, incompatible, corrupt, or unsafe registry entries."""


class MLPromotionError(MLRegistryError):
    """Raised when a requested lifecycle transition is not allowed."""
