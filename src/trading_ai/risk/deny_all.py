"""Fail-closed Lot 0 risk implementation."""

from trading_ai.core.models import (
    OrderRequest,
    PortfolioSnapshot,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
)
from trading_ai.risk.base import RiskEngine


class DenyAllRiskEngine(RiskEngine):
    """Reject every order request, in every profile and environment."""

    def evaluate(
        self,
        order: OrderRequest,
        portfolio: PortfolioSnapshot,
        context: TradingContext,
    ) -> RiskDecision:
        del portfolio, context
        return RiskDecision(
            decision_id=f"deny-all:{order.order_id}",
            order_id=order.order_id,
            status=RiskDecisionStatus.REJECT,
            reason="Lot 0 default policy denies all trading requests",
            risk_engine=type(self).__name__,
        )
