"""Observation consistency, not strategy ownership or execution permission."""
from trading_ai.brokers.models import ReconciliationStatus
from trading_ai.brokers.soak.models import ReadOnlySnapshot, SoakConfig, SoakReconciliation
from trading_ai.core.hashing import to_primitive


def _semantic(rows, ignored):
    # Broker receipt timestamps change on refresh; economic fields must not.
    return sorted(({
        k: v for k, v in to_primitive(row).items() if k not in ignored
    } for row in rows), key=lambda row: str(sorted(row.items())))


def reconcile_observations(baseline: ReadOnlySnapshot, current: ReadOnlySnapshot,
                           config: SoakConfig) -> SoakReconciliation:
    reasons = set()
    critical = set()
    external = False
    a, b = baseline.account, current.account
    if a.account.account_hash != b.account.account_hash:
        critical.add("ACCOUNT_CHANGED")
    if a.account.base_currency != b.account.base_currency:
        critical.add("BASE_CURRENCY_CHANGED")
    if current.state.cash != b.cash or current.state.positions != b.positions:
        critical.add("NON_ATOMIC_BROKER_OBSERVATION")
    if b.cash < 0:
        critical.add("NEGATIVE_CASH")
    if abs(a.cash - b.cash) > config.cash_tolerance:
        critical.add("CASH_MISMATCH")
        external = True
    if a.positions != b.positions:
        critical.add("POSITION_MISMATCH")
        external = True
    if abs(a.net_liquidation - b.net_liquidation) > config.equity_tolerance:
        # Equity can move from market prices, interest or FX without a trade.
        # Preserve the difference; do not silently absorb it or infer its cause.
        reasons.add("EQUITY_CHANGED_MARK_TO_MARKET_UNVERIFIED")
    if _semantic(baseline.state.orders, {"created_at", "updated_at", "session_id"}) != _semantic(
        current.state.orders, {"created_at", "updated_at", "session_id"}
    ):
        reasons.add("ORDER_STATE_MISMATCH")
        external = True
    if _semantic(baseline.state.executions, {"received_at"}) != _semantic(
        current.state.executions, {"received_at"}
    ):
        critical.add("EXECUTION_MISMATCH")
        external = True
    if _semantic(baseline.commissions, {"received_at"}) != _semantic(current.commissions, {"received_at"}):
        reasons.add("COMMISSION_CHANGED")
    if external:
        reasons.add("EXTERNAL_BROKER_ACTIVITY")
    reasons.update(critical)
    status = (ReconciliationStatus.CRITICAL_DRIFT if critical else
              ReconciliationStatus.UNKNOWN if current.missing else
              ReconciliationStatus.DRIFT if reasons else ReconciliationStatus.IN_SYNC)
    return SoakReconciliation(current.timestamp, baseline.snapshot_id, current.snapshot_id,
                              status, tuple(sorted(reasons)), external)
