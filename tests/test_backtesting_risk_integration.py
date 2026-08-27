"""Mandatory risk-gate integration across strategy, order, fill, and export."""

from __future__ import annotations

import ast
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as parquet
import pytest

from backtest_support import START, bar, dataset
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.exceptions import BacktestStorageError
from trading_ai.backtesting.models import BacktestConfig, OrderIntent, OrderStatus
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.backtesting.strategy import BacktestStrategy, BuyAndHoldDemoStrategy
from trading_ai.core.config import PROJECT_ROOT, load_runtime_settings
from trading_ai.core.models import OrderSide, RiskDecisionStatus
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.deny_all import DenyAllRiskEngine
from trading_ai.strategies import (
    BreakoutConfig,
    BreakoutStrategy,
    MomentumConfig,
    MomentumStrategy,
    TrendConfig,
    TrendFollowingStrategy,
)


PROFILE = load_runtime_settings().profile


def _series(closes: list[float], symbol: str = "AAPL"):
    return tuple(
        bar(
            index,
            symbol=symbol,
            opening=str(close),
            high=str(close + 0.5),
            low=str(close - 0.5),
            close=str(close),
            timestamp=START + timedelta(days=index),
        )
        for index, close in enumerate(closes)
    )


def _run(strategy, datasets, paper_context, *, benchmark="AAPL"):
    return BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(PROFILE),
        code_version="lot4-test",
    ).run(
        strategy,
        datasets,
        paper_context,
        BacktestConfig(
            starting_cash=Decimal("10000"),
            primary_timeframe="1d",
            benchmark_symbol=benchmark,
        ),
    )


class SellWithoutPositionStrategy(BacktestStrategy):
    name = "sell-without-position-test"

    def __init__(self) -> None:
        self.done = False

    def reset(self) -> None:
        self.done = False

    def on_bar(self, context):
        if self.done:
            return ()
        self.done = True
        return (
            OrderIntent(
                symbol=context.current_bar.symbol,
                side=OrderSide.SELL,
                quantity=Decimal("1"),
                timeframe=context.current_bar.timeframe,
            ),
        )


class FiveConcurrentBuysStrategy(BacktestStrategy):
    name = "concurrent-reservation-test"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        self.symbols = symbols
        self.done = False

    def reset(self) -> None:
        self.done = False

    def on_bar(self, context):
        if self.done or context.current_bar.symbol != max(self.symbols):
            return ()
        current = {
            item.symbol
            for item in context.history
            if item.timestamp == context.current_time
        }
        if not set(self.symbols).issubset(current):
            return ()
        self.done = True
        return tuple(
            OrderIntent(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal("1000"),
                timeframe="1d",
            )
            for symbol in self.symbols
        )


class DuplicateExitStrategy(BacktestStrategy):
    name = "pending-exit-reservation-test"

    def __init__(self) -> None:
        self.bought = False
        self.sells_sent = False

    def reset(self) -> None:
        self.bought = False
        self.sells_sent = False

    def on_bar(self, context):
        held = next(
            (
                position.quantity
                for position in context.portfolio.positions
                if position.symbol == "AAPL"
            ),
            Decimal("0"),
        )
        if not self.bought:
            self.bought = True
            return (OrderIntent("AAPL", OrderSide.BUY, Decimal("10")),)
        if held > 0 and not self.sells_sent:
            self.sells_sent = True
            return (
                OrderIntent("AAPL", OrderSide.SELL, held),
                OrderIntent("AAPL", OrderSide.SELL, held),
            )
        return ()


class UnderpricedIntentStrategy(BacktestStrategy):
    name = "underpriced-intent-test"

    def __init__(self) -> None:
        self.done = False

    def reset(self) -> None:
        self.done = False

    def on_bar(self, context):
        if self.done:
            return ()
        self.done = True
        return (
            OrderIntent(
                "AAPL",
                OrderSide.BUY,
                Decimal("1000"),
                expected_entry_price=Decimal("1"),
            ),
        )


def test_production_backtester_defaults_to_deny_all_and_never_fills() -> None:
    from trading_ai.core.models import ExecutionEnvironment, TradingContext, TradingProfileName

    context = TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED)
    result = BacktestEngine(code_version="default-deny-test").run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("1")),
        (dataset(_series([100, 101, 102])),),
        context,
        BacktestConfig(starting_cash=Decimal("10000")),
    )

    assert result.risk_engine_name == "DenyAllRiskEngine"
    assert isinstance(BacktestEngine().risk_engine, DenyAllRiskEngine)
    assert result.orders[0].status is OrderStatus.REJECTED
    assert result.risk_decisions[0].status is RiskDecisionStatus.REJECT
    assert result.fills == ()


def test_reduced_decision_quantity_is_the_only_quantity_sent_to_fill(
    paper_context,
) -> None:
    result = _run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("1000")),
        (dataset(_series([100, 101, 102])),),
        paper_context,
    )

    decision = result.risk_decisions[0]
    assert decision.status is RiskDecisionStatus.REDUCE
    assert decision.requested_quantity == Decimal("1000")
    assert result.orders[0].quantity == decision.approved_quantity
    assert result.fills[0].quantity == decision.approved_quantity
    assert result.orders[0].risk_decision_id == decision.decision_id


def test_rejected_risk_order_never_reaches_execution(paper_context) -> None:
    result = _run(
        SellWithoutPositionStrategy(),
        (dataset(_series([100, 101, 102])),),
        paper_context,
    )
    assert result.risk_summary.rejected_orders == 1
    assert result.orders[0].status is OrderStatus.REJECTED
    assert result.fills == ()


def test_strategy_cannot_understate_observed_price_to_evade_risk_limits(
    paper_context,
) -> None:
    result = _run(
        UnderpricedIntentStrategy(),
        (dataset(_series([100, 101, 102])),),
        paper_context,
    )
    assert result.risk_decisions[0].approved_quantity == Decimal("15")
    assert result.orders[0].quantity == Decimal("15")


def test_pending_buy_reservations_prevent_same_timestamp_exposure_bypass(
    paper_context,
) -> None:
    symbols = ("AAPL", "AIR.PA", "AMZN", "META", "QQQ")
    datasets = tuple(
        dataset(_series([100, 101, 102], symbol)) for symbol in symbols
    )
    result = _run(
        FiveConcurrentBuysStrategy(symbols),
        datasets,
        paper_context,
        benchmark="AAPL",
    )

    assert result.risk_summary.reduced_orders == 4
    assert result.risk_summary.rejected_orders == 1
    assert len(result.fills) == 4
    assert result.risk_summary.max_portfolio_exposure <= 0.600001
    assert all(
        decision.approved_quantity <= decision.requested_quantity
        for decision in result.risk_decisions
    )


def test_pending_exit_reservations_prevent_two_sells_of_same_position(
    paper_context,
) -> None:
    result = _run(
        DuplicateExitStrategy(),
        (dataset(_series([100, 101, 102, 103])),),
        paper_context,
    )
    sell_decisions = result.risk_decisions[1:]

    assert [item.status for item in sell_decisions] == [
        RiskDecisionStatus.APPROVE,
        RiskDecisionStatus.REJECT,
    ]
    assert [fill.side for fill in result.fills] == [OrderSide.BUY, OrderSide.SELL]


def _baseline_cases():
    return (
        (
            "trend",
            TrendFollowingStrategy(
                ("AAPL",),
                "1d",
                TrendConfig(
                    fast_window=2,
                    slow_window=3,
                    slope_lookback=1,
                    allocation_fraction=Decimal("0.5"),
                ),
            ),
            (dataset(_series([100, 101, 102, 103, 104, 99, 98, 97])),),
        ),
        (
            "breakout",
            BreakoutStrategy(
                ("AAPL",),
                "1d",
                BreakoutConfig(
                    entry_window=3,
                    exit_window=2,
                    allocation_fraction=Decimal("0.5"),
                ),
            ),
            (dataset(_series([100, 100, 100, 102, 103, 98, 97, 97])),),
        ),
        (
            "momentum",
            MomentumStrategy(
                ("AAPL", "META", "MSFT"),
                "1d",
                MomentumConfig(
                    lookback=2,
                    top_k=1,
                    rebalance_every=1,
                    allocation_fraction=Decimal("0.5"),
                ),
            ),
            (
                dataset(_series([100, 110, 120, 105, 100, 95], "AAPL")),
                dataset(_series([100, 99, 98, 97, 96, 95], "META")),
                dataset(_series([100, 105, 110, 125, 135, 140], "MSFT")),
            ),
        ),
    )


@pytest.mark.parametrize(("name", "strategy", "datasets"), _baseline_cases())
def test_lot3_baselines_use_balanced_risk_with_full_lineage(
    name, strategy, datasets, paper_context
) -> None:
    result = _run(strategy, datasets, paper_context)

    assert result.strategy_name == name
    assert result.risk_engine_name == "balanced-risk"
    assert result.risk_engine_version == "1.0"
    assert result.risk_config_hash == result.risk_summary.risk_config_hash
    assert result.orders
    assert len(result.orders) == len(result.risk_decisions)
    assert all(order.risk_decision_id for order in result.orders)
    assert all(
        order.risk_decision_id == decision.decision_id
        for order, decision in zip(result.orders, result.risk_decisions)
    )
    assert all(
        decision.approved_quantity <= decision.requested_quantity
        for decision in result.risk_decisions
    )


def test_same_risk_inputs_produce_identical_decisions_and_result_hash(
    paper_context,
) -> None:
    inputs = (dataset(_series([100, 101, 102, 103, 104, 99, 98, 97])),)
    config = TrendConfig(fast_window=2, slow_window=3, slope_lookback=1)
    first = _run(TrendFollowingStrategy(("AAPL",), "1d", config), inputs, paper_context)
    second = _run(TrendFollowingStrategy(("AAPL",), "1d", config), inputs, paper_context)

    assert first.risk_decisions == second.risk_decisions
    assert first.risk_state_transitions == second.risk_state_transitions
    assert first.orders == second.orders
    assert first.fills == second.fills
    assert first.result_hash == second.result_hash


def test_risk_export_is_hashed_and_tamper_evident(tmp_path, paper_context) -> None:
    result = _run(
        BuyAndHoldDemoStrategy("AAPL", Decimal("1000")),
        (dataset(_series([100, 101, 102])),),
        paper_context,
    )
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)

    assert parquet.read_table(directory / "risk_decisions.parquet").num_rows == 1
    assert (directory / "risk_states.parquet").is_file()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "1.3"
    assert summary["risk"]["config_hash"] == result.risk_config_hash
    assert store.verify_integrity(result.run_id) is True

    (directory / "risk_decisions.parquet").write_bytes(b"tampered")
    with pytest.raises(BacktestStorageError, match="SHA-256 mismatch"):
        store.verify_integrity(result.run_id)


def test_risk_modules_import_no_provider_broker_or_network_client() -> None:
    risk_root = PROJECT_ROOT / "src" / "trading_ai" / "risk"
    forbidden = (
        "yfinance",
        "requests",
        "trading_ai.data",
        "trading_ai.brokers",
        "ibkr",
    )
    for path in risk_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.lower())
        assert not any(
            module.startswith(prefix)
            for module in imported
            for prefix in forbidden
        ), f"forbidden risk dependency in {path.name}: {imported}"


def test_risk_package_contains_no_signal_or_strategy_dependency() -> None:
    risk_root = Path(PROJECT_ROOT) / "src" / "trading_ai" / "risk"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in risk_root.glob("*.py")
    )
    assert "StrategySignal" not in source
    assert "ENTER_LONG" not in source
    assert "EXIT_LONG" not in source
    assert "trading_ai.strategies" not in source
