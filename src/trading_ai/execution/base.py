"""Risk-gated execution entry point with defense-in-depth Lot 0 locks."""

from __future__ import annotations

from dataclasses import replace

from trading_ai.brokers.base import BrokerAdapter
from trading_ai.core.models import (
    ExecutionEnvironment,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    PortfolioSnapshot,
    RiskApprovedOrder,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
    TradingProfileName,
)
from trading_ai.risk.base import RiskEngine
from trading_ai.risk.deny_all import DenyAllRiskEngine


class ExecutionEngine:
    """Only public order-submission path, permanently wrapped by RiskEngine.

    Subclasses may not replace ``submit_order``. Broker implementations receive
    only a ``RiskApprovedOrder`` after this method has checked the mandatory
    risk decision and the Lot 0 environment/profile locks.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "submit_order" in cls.__dict__:
            raise TypeError("ExecutionEngine.submit_order is a sealed risk boundary")

    def __init__(
        self,
        broker: BrokerAdapter,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        if broker is None:
            raise TypeError("broker is required")
        self._broker = broker
        self._risk_engine = risk_engine or DenyAllRiskEngine()

    @property
    def risk_engine(self) -> RiskEngine:
        """Expose the active mandatory risk component for diagnostics."""

        return self._risk_engine

    def submit_order(
        self,
        order: OrderRequest,
        portfolio: PortfolioSnapshot,
        context: TradingContext,
    ) -> ExecutionResult:
        """Evaluate risk, enforce Lot 0 locks, then delegate approved orders."""

        decision = self._risk_engine.evaluate(order, portfolio, context)
        self._validate_risk_decision(order, decision)
        if decision.status is RiskDecisionStatus.REJECT:
            return ExecutionResult(
                order_id=order.order_id,
                status=ExecutionStatus.BLOCKED,
                message=decision.reason,
                risk_decision=decision,
            )
        if context.environment is ExecutionEnvironment.LIVE:
            return ExecutionResult(
                order_id=order.order_id,
                status=ExecutionStatus.BLOCKED,
                message="LIVE order transmission is locked in Lot 0",
                risk_decision=decision,
            )
        if context.profile is TradingProfileName.AGGRESSIVE:
            return ExecutionResult(
                order_id=order.order_id,
                status=ExecutionStatus.BLOCKED,
                message="aggressive execution is locked in Lot 0",
                risk_decision=decision,
            )
        submitted_order = order
        if decision.status is RiskDecisionStatus.REDUCE:
            if decision.approved_quantity is None:
                raise ValueError("REDUCE decision omitted approved_quantity")
            submitted_order = replace(order, quantity=decision.approved_quantity)
        approved_order = RiskApprovedOrder(
            order=submitted_order, risk_decision=decision
        )
        receipt = self._broker.submit_approved(approved_order, context)
        return ExecutionResult(
            order_id=order.order_id,
            status=ExecutionStatus.SUBMITTED,
            message="broker accepted risk-approved order",
            risk_decision=decision,
            receipt=receipt,
        )

    @staticmethod
    def _validate_risk_decision(
        order: OrderRequest, decision: RiskDecision
    ) -> None:
        if not isinstance(decision, RiskDecision):
            raise TypeError("RiskEngine must return RiskDecision")
        if decision.order_id != order.order_id:
            raise ValueError("RiskEngine returned a decision for another order")
        if (
            decision.approved_quantity is not None
            and decision.approved_quantity > order.quantity
        ):
            raise ValueError("RiskEngine may never increase requested quantity")
