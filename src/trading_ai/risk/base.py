"""Risk engine interface shared by generic execution and offline simulation."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from trading_ai.core.models import (
    OrderRequest,
    PortfolioSnapshot,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
)
from trading_ai.risk.models import (
    RiskContext,
    RiskState,
    RiskStateSnapshot,
    RiskStateTransition,
    RiskSummary,
)
from trading_ai.risk.reporting import summarize_risk


class RiskEngine(ABC):
    """Mandatory authorization gate between order proposals and execution."""

    @abstractmethod
    def evaluate(
        self,
        order: OrderRequest,
        portfolio: PortfolioSnapshot,
        context: TradingContext,
    ) -> RiskDecision:
        """Evaluate the generic execution path, which remains fail closed."""

    def evaluate_context(self, context: RiskContext) -> RiskDecision:
        """Evaluate a rich point-in-time context; default is fail closed."""

        return RiskDecision(
            decision_id=f"risk-context-unavailable:{context.order.order_id}",
            order_id=context.order.order_id,
            status=RiskDecisionStatus.REJECT,
            reason="risk engine does not implement point-in-time evaluation",
            risk_engine=self.engine_name,
            timestamp=context.timestamp,
            engine_version=self.engine_version,
            requested_quantity=context.order.quantity,
            approved_quantity=Decimal("0"),
            reason_codes=("INVALID_RISK_CONTEXT",),
            human_readable_reasons=(
                "Risk engine does not implement point-in-time evaluation.",
            ),
            risk_state=context.risk_state.state.value,
            config_hash=self.config_hash,
            equity=context.equity,
            cash=context.cash,
        )

    @property
    def engine_name(self) -> str:
        return type(self).__name__

    @property
    def engine_version(self) -> str:
        return "0"

    @property
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        return ()

    @property
    def config_hash(self) -> str:
        return "0" * 64

    @property
    def state_transitions(self) -> tuple[RiskStateTransition, ...]:
        return ()

    def reset(self, timestamp: datetime, equity: Decimal) -> None:
        """Start one deterministic simulation run."""

        del timestamp, equity

    def observe(self, timestamp: datetime, equity: Decimal) -> None:
        """Observe current equity without making an order decision."""

        del timestamp, equity

    def current_state(
        self, timestamp: datetime, equity: Decimal
    ) -> RiskStateSnapshot:
        """Return a conservative point-in-time state for rich evaluation."""

        return RiskStateSnapshot(
            timestamp=timestamp,
            state=RiskState.HALTED,
            peak_equity=equity,
            day_start_equity=equity,
            current_equity=equity,
            risk_day=timestamp.date(),
            daily_loss_pct=0.0,
            drawdown_pct=0.0,
            halt_reason="DEFAULT_FAIL_CLOSED",
        )

    def summary(
        self,
        decisions: tuple[RiskDecision, ...],
        completed_at: datetime,
    ) -> RiskSummary:
        return summarize_risk(
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            config_hash=self.config_hash,
            decisions=decisions,
            tracker=None,
            completed_at=completed_at,
        )
