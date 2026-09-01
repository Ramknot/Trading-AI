"""TWS API compatibility metadata without vendoring IBKR code."""

from __future__ import annotations

from trading_ai.brokers.exceptions import BrokerConfigurationError


IBKR_ADAPTER_NAME = "ibkr-tws-paper"
IBKR_ADAPTER_VERSION = "1.0"
OFFICIAL_TWS_API_STABLE = "10.45"
OFFICIAL_TWS_API_PYTHON = "10.50"
SUPPORTED_TWS_API_VERSIONS = frozenset({"10.45", "10.50"})


def validate_sdk_version(expected: str, observed: str | None) -> None:
    if expected not in SUPPORTED_TWS_API_VERSIONS:
        raise BrokerConfigurationError(
            f"unsupported expected TWS API version {expected}; reviewed versions are 10.45/10.50"
        )
    if observed is None:
        raise BrokerConfigurationError("official TWS API runtime version is unavailable")
    normalized = observed.strip().lstrip("v")
    if not normalized.startswith(expected):
        raise BrokerConfigurationError(
            f"official TWS API version mismatch: expected {expected}, observed {observed}"
        )
