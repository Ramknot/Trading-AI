"""Paper-only account guard and non-armed execution boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from trading_ai.brokers.base import BrokerAdapter
from trading_ai.brokers.exceptions import PaperAccountGuardError, PaperExecutionLockedError
from trading_ai.brokers.models import (
    BrokerConnectionState,
    BrokerEnvironment,
    DataFreshnessStatus,
    PaperMode,
    PaperSafetySnapshot,
    PaperSessionState,
    ReconciliationStatus,
)
from trading_ai.core.models import (
    ExecutionEnvironment,
    ExecutionReceipt,
    OrderSide,
    RiskApprovedOrder,
    TradingContext,
    TradingProfileName,
)


class PaperAccountGuard:
    def __init__(self, allowed_account_hashes: tuple[str, ...]) -> None:
        self.allowed_account_hashes = frozenset(allowed_account_hashes)

    def verify(self, snapshot: PaperSafetySnapshot) -> None:
        account = snapshot.account
        if account is None:
            raise PaperAccountGuardError("broker account identity is unavailable")
        if account.environment is BrokerEnvironment.LIVE:
            raise PaperAccountGuardError("LIVE account is hard-locked in Lot 9")
        if account.environment is BrokerEnvironment.UNKNOWN:
            raise PaperAccountGuardError("unknown broker environment fails closed")
        if not account.environment_verified:
            raise PaperAccountGuardError("Paper account environment is not independently verified")
        if account.account_hash not in self.allowed_account_hashes:
            raise PaperAccountGuardError("Paper account is not in the local hashed allowlist")


class PaperSubmissionAuthorization(ABC):
    """Lot 10 extension point; Lot 9 ships no allowing implementation."""

    @abstractmethod
    def allows(self, snapshot: PaperSafetySnapshot, order: RiskApprovedOrder) -> bool:
        raise NotImplementedError


class DenyPaperSubmissionAuthorization(PaperSubmissionAuthorization):
    def allows(self, snapshot: PaperSafetySnapshot, order: RiskApprovedOrder) -> bool:
        del snapshot, order
        return False


class PaperExecutionBoundary(BrokerAdapter):
    """Final Paper-only boundary after Risk; defaults permanently to deny in Lot 9."""

    def __init__(
        self,
        broker: BrokerAdapter,
        *,
        safety_snapshot: Callable[[], PaperSafetySnapshot],
        account_guard: PaperAccountGuard,
        authorization: PaperSubmissionAuthorization | None = None,
        after_attempt: Callable[[], None] | None = None,
    ) -> None:
        self._broker = broker
        self._safety_snapshot = safety_snapshot
        self._account_guard = account_guard
        self._authorization = authorization or DenyPaperSubmissionAuthorization()
        self._after_attempt = after_attempt

    def submit_approved(
        self, approved_order: RiskApprovedOrder, context: TradingContext
    ) -> ExecutionReceipt:
        if context.environment is not ExecutionEnvironment.PAPER:
            raise PaperExecutionLockedError("Paper boundary accepts PAPER context only")
        if context.profile is not TradingProfileName.BALANCED:
            raise PaperExecutionLockedError("aggressive profile remains locked")
        snapshot = self._safety_snapshot()
        self._account_guard.verify(snapshot)
        if snapshot.mode is not PaperMode.PAPER_EXECUTION_ARMED:
            raise PaperExecutionLockedError("Paper execution is not armed in Lot 9")
        if snapshot.session_state is not PaperSessionState.READY:
            raise PaperExecutionLockedError("Paper session is not READY")
        if snapshot.connection_state is not BrokerConnectionState.CONNECTED:
            raise PaperExecutionLockedError("broker is not connected")
        if snapshot.reconciliation_status is not ReconciliationStatus.IN_SYNC:
            raise PaperExecutionLockedError("broker reconciliation is not IN_SYNC")
        if snapshot.data_freshness is not DataFreshnessStatus.FRESH:
            raise PaperExecutionLockedError("decision data is stale or unavailable")
        if not snapshot.risk_health_ok:
            raise PaperExecutionLockedError("Risk health is not verified")
        if snapshot.market_session_open is not True:
            raise PaperExecutionLockedError("market session is closed or unavailable")
        if not snapshot.configs_frozen:
            raise PaperExecutionLockedError("Paper session configurations are not frozen")
        if snapshot.emergency_halt.active:
            raise PaperExecutionLockedError("Paper emergency halt is active")
        order = approved_order.order
        if order.side is OrderSide.BUY and (
            order.cost_estimate_id is None or order.economic_decision_id is None
        ):
            raise PaperExecutionLockedError(
                "new risk requires CostEngine and EconomicGate lineage"
            )
        if not self._authorization.allows(snapshot, approved_order):
            raise PaperExecutionLockedError(
                "Lot 9 provides no authorization issuer for Paper order transmission"
            )
        transmitter = getattr(self._broker, "transmit_approved", None)
        if not callable(transmitter):
            raise PaperExecutionLockedError(
                "broker does not implement the guarded Paper transport port"
            )
        try:
            return transmitter(approved_order, context)
        finally:
            if self._after_attempt is not None:
                self._after_attempt()
