"""Stable normalization of documented IBKR numeric messages."""

from __future__ import annotations

from dataclasses import dataclass

from trading_ai.brokers.models import BrokerErrorSeverity


@dataclass(frozen=True, slots=True)
class NormalizedIBKRError:
    code: int
    stable_code: str
    severity: BrokerErrorSeverity
    connectivity_lost: bool = False
    reconciliation_required: bool = False


_KNOWN: dict[int, NormalizedIBKRError] = {
    100: NormalizedIBKRError(100, "IBKR_PACING_LIMIT", BrokerErrorSeverity.CRITICAL, True, True),
    103: NormalizedIBKRError(103, "IBKR_DUPLICATE_ORDER_ID", BrokerErrorSeverity.REJECT, False, True),
    1100: NormalizedIBKRError(1100, "IBKR_CONNECTIVITY_LOST", BrokerErrorSeverity.CONNECTIVITY, True, True),
    1101: NormalizedIBKRError(1101, "IBKR_CONNECTIVITY_RESTORED_DATA_LOST", BrokerErrorSeverity.WARNING, False, True),
    1102: NormalizedIBKRError(1102, "IBKR_CONNECTIVITY_RESTORED_DATA_MAINTAINED", BrokerErrorSeverity.INFORMATIONAL),
    1300: NormalizedIBKRError(1300, "IBKR_SOCKET_PORT_RESET", BrokerErrorSeverity.CONNECTIVITY, True, True),
    2103: NormalizedIBKRError(2103, "IBKR_MARKET_DATA_FARM_DISCONNECTED", BrokerErrorSeverity.WARNING),
    2104: NormalizedIBKRError(2104, "IBKR_MARKET_DATA_FARM_OK", BrokerErrorSeverity.INFORMATIONAL),
    2105: NormalizedIBKRError(2105, "IBKR_HISTORICAL_DATA_FARM_DISCONNECTED", BrokerErrorSeverity.WARNING),
    2106: NormalizedIBKRError(2106, "IBKR_HISTORICAL_DATA_FARM_OK", BrokerErrorSeverity.INFORMATIONAL),
    2107: NormalizedIBKRError(2107, "IBKR_HISTORICAL_DATA_FARM_INACTIVE", BrokerErrorSeverity.INFORMATIONAL),
    2108: NormalizedIBKRError(2108, "IBKR_MARKET_DATA_FARM_INACTIVE", BrokerErrorSeverity.INFORMATIONAL),
    2110: NormalizedIBKRError(2110, "IBKR_SERVER_CONNECTIVITY_BROKEN", BrokerErrorSeverity.CONNECTIVITY, True, True),
    502: NormalizedIBKRError(502, "IBKR_SOCKET_CONNECT_FAILED", BrokerErrorSeverity.CONNECTIVITY, True),
    503: NormalizedIBKRError(503, "IBKR_TWS_VERSION_UNSUPPORTED", BrokerErrorSeverity.CRITICAL, True),
    504: NormalizedIBKRError(504, "IBKR_NOT_CONNECTED", BrokerErrorSeverity.CONNECTIVITY, True, True),
}


def normalize_ibkr_error(code: int) -> NormalizedIBKRError:
    return _KNOWN.get(
        code,
        NormalizedIBKRError(code, f"IBKR_CODE_{code}", BrokerErrorSeverity.WARNING),
    )
