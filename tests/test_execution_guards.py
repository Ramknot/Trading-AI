import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_ai.brokers.base import BrokerAdapter
from trading_ai.core.models import (
    ExecutionEnvironment,
    ExecutionReceipt,
    ExecutionStatus,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
    TradingProfileName,
)
from trading_ai.execution.base import ExecutionEngine
from trading_ai.risk.base import RiskEngine
from trading_ai.risk.deny_all import DenyAllRiskEngine


class RecordingBroker(BrokerAdapter):
    def __init__(self) -> None:
        self.calls = 0
        self.last_quantity = None

    def submit_approved(self, approved_order, context) -> ExecutionReceipt:
        del context
        self.calls += 1
        self.last_quantity = approved_order.order.quantity
        return ExecutionReceipt(
            order_id=approved_order.order.order_id,
            broker_order_id="paper-1",
            accepted_at=datetime.now(timezone.utc),
        )


class AllowingRiskEngine(RiskEngine):
    def evaluate(self, order, portfolio, context) -> RiskDecision:
        del portfolio, context
        return RiskDecision(
            decision_id=f"allow:{order.order_id}",
            order_id=order.order_id,
            status=RiskDecisionStatus.APPROVE,
            reason="test approval",
            risk_engine=type(self).__name__,
        )


class ReducingRiskEngine(RiskEngine):
    def evaluate(self, order, portfolio, context) -> RiskDecision:
        del context
        return RiskDecision(
            decision_id=f"reduce:{order.order_id}",
            order_id=order.order_id,
            status=RiskDecisionStatus.REDUCE,
            reason="test reduction",
            risk_engine=type(self).__name__,
            timestamp=portfolio.as_of,
            requested_quantity=order.quantity,
            approved_quantity=order.quantity / 2,
        )


def test_execution_defaults_to_deny_all(order, portfolio, paper_context) -> None:
    broker = RecordingBroker()
    engine = ExecutionEngine(broker)

    result = engine.submit_order(order, portfolio, paper_context)

    assert isinstance(engine.risk_engine, DenyAllRiskEngine)
    assert result.status is ExecutionStatus.BLOCKED
    assert broker.calls == 0


def test_execution_entry_point_cannot_be_overridden() -> None:
    with pytest.raises(TypeError, match="sealed risk boundary"):

        class UnsafeExecutionEngine(ExecutionEngine):
            def submit_order(self, order, portfolio, context):
                return None


def test_live_never_reaches_broker_even_with_approving_risk(order, portfolio) -> None:
    broker = RecordingBroker()
    engine = ExecutionEngine(broker, AllowingRiskEngine())
    live_context = TradingContext(
        ExecutionEnvironment.LIVE, TradingProfileName.BALANCED
    )

    result = engine.submit_order(order, portfolio, live_context)

    assert result.status is ExecutionStatus.BLOCKED
    assert "LIVE" in result.message
    assert broker.calls == 0


def test_aggressive_never_reaches_broker_even_with_approving_risk(
    order, portfolio
) -> None:
    broker = RecordingBroker()
    engine = ExecutionEngine(broker, AllowingRiskEngine())
    aggressive_context = TradingContext(
        ExecutionEnvironment.PAPER, TradingProfileName.AGGRESSIVE
    )

    result = engine.submit_order(order, portfolio, aggressive_context)

    assert result.status is ExecutionStatus.BLOCKED
    assert "aggressive" in result.message
    assert broker.calls == 0


def test_approved_balanced_paper_uses_guarded_broker_path(
    order, portfolio, paper_context
) -> None:
    broker = RecordingBroker()
    engine = ExecutionEngine(broker, AllowingRiskEngine())

    result = engine.submit_order(order, portfolio, paper_context)

    assert result.status is ExecutionStatus.SUBMITTED
    assert broker.calls == 1
    assert result.receipt is not None


def test_generic_execution_submits_only_the_risk_reduced_quantity(
    order, portfolio, paper_context
) -> None:
    broker = RecordingBroker()
    engine = ExecutionEngine(broker, ReducingRiskEngine())

    result = engine.submit_order(order, portfolio, paper_context)

    assert result.status is ExecutionStatus.SUBMITTED
    assert broker.last_quantity == order.quantity / 2


@pytest.mark.parametrize("component", ["strategies", "ml", "portfolio"])
def test_decision_components_do_not_import_brokers_or_execution(component: str) -> None:
    component_root = Path("src/trading_ai") / component
    forbidden = {"trading_ai.brokers", "trading_ai.execution"}

    for path in component_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name == blocked or name.startswith(f"{blocked}.")
            for name in imported
            for blocked in forbidden
        ), f"{path} bypasses the guarded execution architecture"
