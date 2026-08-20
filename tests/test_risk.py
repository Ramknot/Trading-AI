import pytest

from trading_ai.core.models import (
    ExecutionEnvironment,
    RiskDecisionStatus,
    TradingContext,
    TradingProfileName,
)
from trading_ai.risk.deny_all import DenyAllRiskEngine


@pytest.mark.parametrize("environment", list(ExecutionEnvironment))
@pytest.mark.parametrize("profile", list(TradingProfileName))
def test_deny_all_rejects_every_context(
    environment, profile, order, portfolio
) -> None:
    decision = DenyAllRiskEngine().evaluate(
        order,
        portfolio,
        TradingContext(environment=environment, profile=profile),
    )

    assert decision.status is RiskDecisionStatus.REJECT
    assert decision.order_id == order.order_id
    assert decision.risk_engine == "DenyAllRiskEngine"
    assert "denies all" in decision.reason
