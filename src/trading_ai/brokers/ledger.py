"""Idempotent cash-only long Paper ledger driven by broker executions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_ai.brokers.exceptions import ReconciliationRequiredError
from trading_ai.brokers.models import (
    BrokerCommissionReport,
    BrokerExecution,
    BrokerPosition,
    CommissionKnowledge,
)
from trading_ai.core.models import OrderSide


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class PaperLedgerSnapshot:
    cash: Decimal
    positions: tuple[BrokerPosition, ...]
    applied_execution_ids: tuple[str, ...]
    applied_commission_exec_ids: tuple[str, ...]


class PaperLedger:
    """No buying power, margin, shorting, or inferred zero commission."""

    def __init__(
        self,
        *,
        cash: Decimal,
        base_currency: str,
        positions: tuple[BrokerPosition, ...] = (),
    ) -> None:
        if not cash.is_finite() or cash < ZERO:
            raise ValueError("Paper ledger cash must be finite and non-negative")
        self.cash = cash
        self._initial_cash = cash
        self.base_currency = base_currency
        self._initial_positions = tuple(positions)
        self._positions = {item.symbol: item for item in positions}
        self._executions: dict[str, BrokerExecution] = {}
        self._active_by_root: dict[str, str] = {}
        self._commission_exec_ids: set[str] = set()
        self._commission_reports: dict[str, BrokerCommissionReport] = {}

    def apply_execution(self, execution: BrokerExecution) -> bool:
        if execution.exec_id in self._executions:
            return False
        if execution.correction_of is not None:
            previous = self._executions.get(execution.correction_of)
            if previous is None:
                raise ReconciliationRequiredError("execution correction lacks its original execution")
        self._executions[execution.exec_id] = execution
        self._active_by_root[execution.exec_id.rsplit(".", 1)[0]] = execution.exec_id
        self._rebuild()
        return True

    def _rebuild(self) -> None:
        self.cash = self._initial_cash
        self._positions = {item.symbol: item for item in self._initial_positions}
        active = [self._executions[exec_id] for exec_id in self._active_by_root.values()]
        for execution in sorted(active, key=lambda item: (item.received_at, item.exec_id)):
            self._apply(execution, reverse=False)
        for exec_id in sorted(self._commission_exec_ids):
            report = self._commission_reports[exec_id]
            assert report.amount is not None
            if self.cash - report.amount < ZERO:
                raise ReconciliationRequiredError("broker commission would create negative cash")
            self.cash -= report.amount

    def _apply(self, execution: BrokerExecution, *, reverse: bool) -> None:
        quantity = -execution.quantity if reverse else execution.quantity
        if execution.side is OrderSide.SELL:
            quantity = -quantity
        current = self._positions.get(
            execution.symbol,
            BrokerPosition(execution.symbol, ZERO, ZERO, self.base_currency),
        )
        next_quantity = current.quantity + quantity
        cash_delta = -(quantity * execution.price)
        if next_quantity < ZERO:
            raise ReconciliationRequiredError("broker execution would create a forbidden short")
        if self.cash + cash_delta < ZERO:
            raise ReconciliationRequiredError("broker execution would create negative cash")
        if quantity > ZERO:
            cost_before = current.quantity * current.average_cost
            average = (cost_before + quantity * execution.price) / next_quantity
        else:
            average = current.average_cost if next_quantity > ZERO else ZERO
        self.cash += cash_delta
        if next_quantity == ZERO:
            self._positions.pop(execution.symbol, None)
        else:
            self._positions[execution.symbol] = BrokerPosition(
                execution.symbol, next_quantity, average, current.currency
            )

    def apply_commission(self, report: BrokerCommissionReport) -> bool:
        if report.exec_id in self._commission_exec_ids:
            return False
        if report.exec_id not in self._executions:
            raise ReconciliationRequiredError("commission report has no recorded execution")
        if report.status is CommissionKnowledge.UNAVAILABLE:
            return False
        assert report.amount is not None
        if report.currency != self.base_currency:
            raise ReconciliationRequiredError("commission currency requires explicit FX conversion")
        self._commission_reports[report.exec_id] = report
        self._commission_exec_ids.add(report.exec_id)
        self._rebuild()
        return True

    @property
    def snapshot(self) -> PaperLedgerSnapshot:
        return PaperLedgerSnapshot(
            cash=self.cash,
            positions=tuple(self._positions[key] for key in sorted(self._positions)),
            applied_execution_ids=tuple(sorted(self._executions)),
            applied_commission_exec_ids=tuple(sorted(self._commission_exec_ids)),
        )
