"""Fail-closed Lot 0 risk implementation."""

from decimal import Decimal

from trading_ai.core.models import (
    OrderRequest,
    PortfolioSnapshot,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
)
from trading_ai.risk.base import RiskEngine
from trading_ai.risk.models import RiskContext


class DenyAllRiskEngine(RiskEngine):
    """Reject every order request, in every profile and environment."""

    def evaluate(
        self,
        order: OrderRequest,
        portfolio: PortfolioSnapshot,
        context: TradingContext,
    ) -> RiskDecision:
        del context
        return RiskDecision(
            decision_id=f"deny-all:{order.order_id}",
            order_id=order.order_id,
            status=RiskDecisionStatus.REJECT,
            reason="Lot 0 default policy denies all trading requests",
            risk_engine=type(self).__name__,
            timestamp=order.created_at or portfolio.as_of,
            engine_version=self.engine_version,
            requested_quantity=order.quantity,
            approved_quantity=Decimal("0"),
            reason_codes=("DENY_ALL_DEFAULT",),
            human_readable_reasons=(
                "The default fail-safe Risk Engine denies every request.",
            ),
            risk_state="HALTED",
            config_hash=self.config_hash,
            equity=portfolio.total_equity,
            cash=portfolio.cash,
        )

    def evaluate_context(self, context: RiskContext) -> RiskDecision:
        return RiskDecision(
            decision_id=f"deny-all:{context.order.order_id}",
            order_id=context.order.order_id,
            status=RiskDecisionStatus.REJECT,
            reason="Lot 0 default policy denies all trading requests",
            risk_engine=self.engine_name,
            timestamp=context.timestamp,
            engine_version=self.engine_version,
            requested_quantity=context.order.quantity,
            approved_quantity=Decimal("0"),
            reason_codes=("DENY_ALL_DEFAULT",),
            human_readable_reasons=(
                "The default fail-safe Risk Engine denies every request.",
            ),
            risk_state="HALTED",
            config_hash=self.config_hash,
            equity=context.equity,
            cash=context.cash,
        )

    @property
    def engine_version(self) -> str:
        return "1.0"
