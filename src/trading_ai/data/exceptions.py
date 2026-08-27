"""Provider-independent Data Engine exception hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_ai.data.models import DataQualityReport


class DataError(Exception):
    """Base class for all Data Engine failures."""


class DataProviderError(DataError):
    """A provider failed without leaking its implementation-specific error."""


class DataProviderTemporaryError(DataProviderError):
    """A transient provider error that may be retried a bounded number of times."""


class DataUnavailableError(DataProviderError):
    """The provider has no data for the requested symbol and interval."""


class DataValidationError(DataError, ValueError):
    """Normalized data failed explicit quality validation."""

    def __init__(
        self, message: str, quality_report: DataQualityReport | None = None
    ) -> None:
        super().__init__(message)
        self.quality_report = quality_report


class DataIntegrityError(DataError):
    """A persisted dataset does not match its recorded SHA-256 checksum."""


class DataStorageError(DataError):
    """A local dataset or manifest could not be stored or read."""


class CacheMissError(DataError):
    """CACHE_ONLY was requested but no exact local dataset exists."""
