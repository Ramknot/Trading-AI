"""Fail-closed broker and Paper-session errors."""


class BrokerError(Exception):
    """Base error for provider-neutral broker infrastructure."""


class BrokerConfigurationError(BrokerError):
    """Raised when local broker configuration is unsafe or incomplete."""


class BrokerUnavailableError(BrokerError):
    """Raised when a broker connection cannot safely service a request."""


class PaperExecutionLockedError(BrokerError):
    """Raised when Lot 9's non-armed Paper boundary rejects transmission."""


class PaperAccountGuardError(BrokerError):
    """Raised when account identity or environment cannot be proven Paper."""


class ReconciliationRequiredError(BrokerError):
    """Raised when broker/local state must be reconciled before proceeding."""


class BrokerStateTransitionError(BrokerError):
    """Raised for an invalid order or session lifecycle transition."""


class BrokerIntegrityError(BrokerError):
    """Raised when a local Paper artifact fails its checksum."""


class ContractResolutionError(BrokerError):
    """Raised when an IBKR contract is absent, ambiguous, or mismatched."""


class IBKRSDKUnavailableError(BrokerError):
    """Raised when the separately licensed official TWS API is unavailable."""
