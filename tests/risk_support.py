"""Explicitly permissive risk helpers restricted to deterministic unit tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_ai.backtesting.engine import BacktestEngine as ProductionBacktestEngine
from trading_ai.core.models import (
    OrderRequest,
    PortfolioSnapshot,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
)
from trading_ai.risk.base import RiskEngine
from trading_ai.risk.models import RiskContext, RiskState, RiskStateSnapshot


class PermissiveTestRiskEngine(RiskEngine):
    """Approve exact requested sizes solely for isolated Backtester tests."""

    @property
    def engine_name(self) -> str:
        return "permissive-test-risk"

    @property
    def engine_version(self) -> str:
        return "test-only"

    @property
    def config_hash(self) -> str:
        return "f" * 64

    def evaluate(
        self,
        order: OrderRequest,
        portfolio: PortfolioSnapshot,
        context: TradingContext,
    ) -> RiskDecision:
        del context
        return self._approve(order, portfolio)

    def evaluate_context(self, context: RiskContext) -> RiskDecision:
        return self._approve(context.order, context.portfolio)

    def current_state(
        self, timestamp: datetime, equity: Decimal
    ) -> RiskStateSnapshot:
        return RiskStateSnapshot(
            timestamp=timestamp,
            state=RiskState.NORMAL,
            peak_equity=equity,
            day_start_equity=equity,
            current_equity=equity,
            risk_day=timestamp.date(),
            daily_loss_pct=0.0,
            drawdown_pct=0.0,
        )

    def _approve(
        self, order: OrderRequest, portfolio: PortfolioSnapshot
    ) -> RiskDecision:
        return RiskDecision(
            decision_id=f"test-risk:{order.order_id}",
            order_id=order.order_id,
            status=RiskDecisionStatus.APPROVE,
            reason="explicit test-only approval",
            risk_engine=self.engine_name,
            timestamp=order.created_at or portfolio.as_of,
            engine_version=self.engine_version,
            requested_quantity=order.quantity,
            approved_quantity=order.quantity,
            reason_codes=("TEST_ONLY_APPROVAL",),
            human_readable_reasons=("Explicit test-only approval.",),
            risk_state=RiskState.NORMAL.value,
            config_hash=self.config_hash,
            equity=portfolio.total_equity,
            cash=portfolio.cash,
        )


class PermissiveBacktestEngine(ProductionBacktestEngine):
    """Preserve Lot 2 unit assumptions through explicit test-only injection."""

    def __init__(self, *args, risk_engine: RiskEngine | None = None, **kwargs) -> None:
        super().__init__(
            *args,
            risk_engine=risk_engine or PermissiveTestRiskEngine(),
            **kwargs,
        )
