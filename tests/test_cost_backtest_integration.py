from __future__ import annotations

from dataclasses import replace
from datetime import timezone
from decimal import Decimal

import pyarrow.parquet as parquet

from backtest_support import bar, dataset
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.models import BacktestConfig, OrderIntent
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.backtesting.strategy import BacktestStrategy, BuyAndHoldDemoStrategy
from trading_ai.core.config import load_profile, load_runtime_settings
from trading_ai.core.hashing import stable_hash
from trading_ai.core.models import OrderRequest, OrderSide, PortfolioSnapshot
from trading_ai.costs.economics import EconomicGate
from trading_ai.costs.engine import BalancedTransactionCostEngine
from trading_ai.costs.models import CostCoverage, EconomicDecisionStatus
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.config import load_balanced_risk_config
from trading_ai.risk.models import RiskContext


class TwoPendingBuysStrategy(BacktestStrategy):
    name = "two-pending-cost-reservations"

    def __init__(self) -> None:
        self.sent = False

    def reset(self) -> None:
        self.sent = False

    def on_bar(self, context):
        if self.sent:
            return ()
        self.sent = True
        return (
            OrderIntent("AAPL", OrderSide.BUY, Decimal("1")),
            OrderIntent("AAPL", OrderSide.BUY, Decimal("1")),
        )


def _cost_engine():
    profile = load_profile("balanced")
    costs = BalancedTransactionCostEngine.from_profile(profile)
    gate = EconomicGate(costs.bundle.config, costs.config_hash)
    return profile, costs, gate


def _run(next_open: str = "100"):
    profile, costs, gate = _cost_engine()
    context = load_runtime_settings("PAPER", "balanced").context
    return BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(profile),
        cost_engine=costs,
        economic_gate=gate,
        code_version="lot-8.1-test",
    ).run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("10")),
        (
            dataset(
                (
                    bar(0, opening="100", close="100"),
                    bar(1, opening=next_open, high=str(Decimal(next_open) + 2),
                        low=str(Decimal(next_open) - 2), close=str(Decimal(next_open) + 1)),
                )
            ),
        ),
        context,
        BacktestConfig(
            starting_cash=Decimal("10000"),
            benchmark_symbol="AAPL",
            spread_bps=Decimal("5"),
            slippage_bps=Decimal("2"),
        ),
    )


def test_cost_aware_backtest_records_lineage_debits_fees_once_and_keeps_risk_sovereign() -> None:
    result = _run()
    assert len(result.cost_estimates) == len(result.economic_decisions) == 1
    assert len(result.risk_decisions) == len(result.fills) == 1
    assert len(result.cost_actuals) == len(result.cost_reconciliations) == 1
    estimate = result.cost_estimates[0]
    decision = result.economic_decisions[0]
    risk = result.risk_decisions[0]
    fill = result.fills[0]
    assert decision.status is EconomicDecisionStatus.INCOMPLETE
    assert decision.allows_new_risk is True  # explicit research-only missing-edge policy
    assert fill.cost_estimate_id == estimate.estimate_id
    assert fill.economic_decision_id == decision.decision_id
    assert result.orders[0].risk_decision_id == risk.decision_id
    assert fill.commission == Decimal("1.00000000")
    assert fill.spread_cost == Decimal("0.5000")
    assert fill.slippage_cost == Decimal("0.2000")
    # Execution price already embeds spread/slippage. Only separately charged
    # components (commission here) are debited again.
    assert result.equity_curve[-1].cash == Decimal("8998.30000000")
    assert result.cost_summary is not None
    assert result.cost_summary.cost_coverage is CostCoverage.COMPLETE
    assert result.cost_summary.gross_trading_pnl == Decimal("10.00000000")
    assert result.cost_summary.net_trading_pnl_before_operating == Decimal("8.30000000")
    assert result.operating_costs.total_operating_cost.amount is None


def test_pretrade_cost_and_cash_reservation_use_close_t_not_next_open() -> None:
    normal = _run("100")
    gap = _run("150")
    first, second = normal.cost_estimates[0], gap.cost_estimates[0]
    assert first.reference_price == second.reference_price == Decimal("100")
    assert first.entry_costs == second.entry_costs
    assert first.cash_requirement == second.cash_requirement
    assert normal.result_hash != gap.result_hash  # datasets/fills genuinely differ


def test_balanced_risk_cash_cap_includes_fixed_costs_and_buffer() -> None:
    profile = replace(load_profile("balanced"), max_exposure=1.0)
    config, groups = load_balanced_risk_config(profile)
    config = replace(
        config,
        max_portfolio_exposure=Decimal("1"),
        max_single_position_exposure=Decimal("1"),
        max_group_exposure=Decimal("1"),
        max_highly_correlated_exposure=Decimal("1"),
    )
    engine = BalancedRiskEngine(profile, config, groups)
    timestamp = bar(0).timestamp.astimezone(timezone.utc)
    engine.reset(timestamp, Decimal("1000"))
    portfolio = PortfolioSnapshot(timestamp, Decimal("1000"), Decimal("1000"))
    order = OrderRequest(
        order_id="cost-aware-cash",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=timestamp,
        expected_entry_price=Decimal("995"),
        estimated_cash_requirement=Decimal("1002"),
        estimated_unit_cash_requirement=Decimal("1002"),
    )
    decision = engine.evaluate_context(
        RiskContext(
            timestamp=timestamp,
            profile=profile,
            portfolio=portfolio,
            order=order,
            expected_entry_price=Decimal("995"),
            market_prices=(("AAPL", Decimal("995")),),
            risk_state=engine.state_snapshot,
        )
    )
    assert decision.approved_quantity is not None
    assert Decimal("0") < decision.approved_quantity < Decimal("1")
    assert "INSUFFICIENT_CASH" in decision.reason_codes
    assert decision.approved_quantity * Decimal("995") + Decimal("7") <= Decimal("1000")


def test_pending_orders_reserve_notional_costs_and_buffer_without_double_spending() -> None:
    profile = load_profile("balanced")
    base_costs = BalancedTransactionCostEngine.from_profile(profile)
    test_config = replace(
        base_costs.bundle.config,
        cash_buffer_bps=Decimal("0"),
        cash_buffer_absolute=Decimal("500"),
    )
    test_bundle = replace(
        base_costs.bundle,
        config=test_config,
        config_hash=stable_hash(
            (base_costs.bundle.config_hash, test_config.to_parameters())
        ),
    )
    costs = BalancedTransactionCostEngine(test_bundle)
    result = BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(profile),
        cost_engine=costs,
        economic_gate=EconomicGate(costs.bundle.config, costs.config_hash),
        code_version="pending-cost-reservation-test",
    ).run(
        TwoPendingBuysStrategy(),
        (
            dataset(
                (
                    bar(0, opening="100", high="101", low="99", close="100"),
                    bar(1, opening="100", high="101", low="99", close="100"),
                )
            ),
        ),
        load_runtime_settings("PAPER", "balanced").context,
        BacktestConfig(starting_cash=Decimal("1000"), benchmark_symbol="AAPL"),
    )

    assert len(result.cost_estimates) == 2
    assert result.cost_estimates[0].cash_requirement.total_cash_required == Decimal(
        "601.00000000"
    )
    assert result.risk_decisions[0].approved_quantity == Decimal("1")
    assert result.risk_decisions[1].approved_quantity == Decimal("0")
    assert "INSUFFICIENT_CASH" in result.risk_decisions[1].reason_codes
    assert len(result.fills) == 1
    assert result.equity_curve[-1].cash >= Decimal("0")


def test_schema_1_6_exports_costs_validation_placeholder_and_checksums(tmp_path) -> None:
    result = _run()
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)
    summary = store.inspect(result.run_id)
    assert summary["schema_version"] == "1.6"
    assert summary["costs"]["engine_name"] == "balanced-transaction-cost"
    assert summary["validation"]["status"] == "UNAVAILABLE"
    assert parquet.read_table(directory / "cost_estimates.parquet").num_rows == 1
    assert parquet.read_table(directory / "cost_actuals.parquet").num_rows == 1
    assert parquet.read_table(directory / "economic_decisions.parquet").num_rows == 1
    assert parquet.read_table(directory / "cost_reconciliation.parquet").num_rows == 1
    assert store.verify_integrity(result.run_id) is True


def test_cost_aware_backtest_is_deterministic_outside_technical_timestamps() -> None:
    first = _run()
    second = _run()
    assert first.cost_estimates == second.cost_estimates
    assert first.economic_decisions == second.economic_decisions
    assert first.cost_actuals == second.cost_actuals
    assert first.cost_reconciliations == second.cost_reconciliations
    assert first.orders == second.orders
    assert first.fills == second.fills
    assert first.result_hash == second.result_hash
