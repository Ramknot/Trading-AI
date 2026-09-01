"""Optional official TWS API adapter, isolated behind BrokerAdapter."""

from trading_ai.brokers.ibkr.adapter import IBKRPaperAdapter
from trading_ai.brokers.ibkr.contracts import IBKRContractResolver, IBKRContractSpec
from trading_ai.brokers.ibkr.versioning import IBKR_ADAPTER_NAME, IBKR_ADAPTER_VERSION

__all__ = [
    "IBKR_ADAPTER_NAME",
    "IBKR_ADAPTER_VERSION",
    "IBKRContractResolver",
    "IBKRContractSpec",
    "IBKRPaperAdapter",
]
