from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from backtest_support import bar, dataset
from ml_support import ConstantAdapter, model_artifact
from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.models import BacktestConfig
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.core.config import load_runtime_settings
from trading_ai.portfolio import BalancedPortfolioEngine, compare_single_to_multi
from trading_ai.ml.features import MLFeatureBuilder
from trading_ai.ml.inference import InferenceEngine, SignalMLScorer
from trading_ai.ml.models import MLMode
from trading_ai.regimes.detector import BalancedRegimeDetector
from trading_ai.regimes.policy import BalancedStrategyActivationPolicy
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.deny_all import DenyAllRiskEngine
from trading_ai.strategies.baselines import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from trading_ai.strategies.config import MomentumConfig


def _rising(symbol: str, *, slope: int, count: int = 130):
    values = []
    for index in range(count):
        opening = Decimal("100") + Decimal(index * slope)
        values.append(
            bar(
                index,
                symbol=symbol,
                opening=opening,
                high=opening + Decimal("2"),
                low=opening - Decimal("1"),
                close=opening + Decimal("1"),
            )
        )
    return dataset(values)


def _oscillating(symbol: str, *, count: int = 150):
    pattern = (-3, -1, 0, 1, 3, 1, 0, -1)
    values = []
    for index in range(count):
        close = Decimal("100") + Decimal(pattern[index % len(pattern)])
        values.append(
            bar(
                index,
                symbol=symbol,
                opening=close,
                high=close + Decimal("1"),
                low=close - Decimal("1"),
                close=close,
            )
        )
    return dataset(values)


def _engine(profile):
    return BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(profile),
        regime_detector=BalancedRegimeDetector.from_profile(profile),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(profile),
        portfolio_engine=BalancedPortfolioEngine.from_profile(profile),
        code_version="lot7-test",
    )


def _scorer(strategy: str, probability: float, mode: MLMode) -> SignalMLScorer:
    adapter = ConstantAdapter(
        probability,
        MLFeatureBuilder.feature_names(strategy),
    )
    artifact = model_artifact(
        adapter,
        model_id=f"model-{strategy}-{probability}",
        strategy_name=strategy,
    )
    return SignalMLScorer(
        mode=mode,
        inference_engine=InferenceEngine(artifact, adapter),
        threshold=0.55,
    )


def _strategies():
    return (
        TrendFollowingStrategy(("AAPL", "MSFT"), "1d"),
        MomentumStrategy(("AAPL", "MSFT"), "1d", MomentumConfig(top_k=2)),
    )


def test_two_strategies_share_one_portfolio_and_every_order_keeps_full_lineage() -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    strategies = _strategies()
    result = _engine(settings.profile).run(
        strategies,
        (_rising("AAPL", slope=1), _rising("MSFT", slope=2)),
        settings.context,
        BacktestConfig(
            starting_cash=Decimal("100000"),
            primary_timeframe="1d",
            benchmark_symbol="AAPL",
        ),
    )

    assert result.strategy_name == "multi-strategy-portfolio"
    assert result.portfolio_engine_name == "balanced-portfolio"
    assert result.portfolio_engine_version == "1.0"
    assert result.portfolio_opportunities
    assert result.portfolio_decisions
    assert result.portfolio_plans
    assert all(order.portfolio_plan_id for order in result.orders)
    assert all(order.portfolio_decision_id for order in result.orders)
    assert all(order.risk_decision_id for order in result.orders)
    assert all(
        decision.approved_quantity <= decision.requested_quantity
        for decision in result.risk_decisions
    )


def test_multi_strategy_score_only_does_not_change_targets_and_filter_blocks_before_portfolio() -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    datasets = (_rising("AAPL", slope=1), _rising("MSFT", slope=2))
    config = BacktestConfig(primary_timeframe="1d", benchmark_symbol="AAPL")
    quant = _engine(settings.profile).run(
        _strategies(), datasets, settings.context, config
    )
    scored = BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(settings.profile),
        regime_detector=BalancedRegimeDetector.from_profile(settings.profile),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(settings.profile),
        portfolio_engine=BalancedPortfolioEngine.from_profile(settings.profile),
        ml_scorers={
            "trend": _scorer("trend", 0.01, MLMode.SCORE_ONLY),
            "momentum": _scorer("momentum", 0.01, MLMode.SCORE_ONLY),
        },
        code_version="lot7-test",
    ).run(_strategies(), datasets, settings.context, config)
    assert scored.ml_mode == "SCORE_ONLY"
    def economic_targets(result):
        return tuple(
            (
                item.timestamp,
                item.symbol,
                item.target_weight,
                item.current_weight,
                item.delta_weight,
                tuple(
                    (contribution.strategy_name, contribution.weight)
                    for contribution in item.contributors
                ),
            )
            for item in result.portfolio_targets
        )

    assert economic_targets(scored) == economic_targets(quant)
    assert tuple(order.quantity for order in scored.orders) == tuple(
        order.quantity for order in quant.orders
    )

    filtered = BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(settings.profile),
        regime_detector=BalancedRegimeDetector.from_profile(settings.profile),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(settings.profile),
        portfolio_engine=BalancedPortfolioEngine.from_profile(settings.profile),
        ml_scorers={
            "trend": _scorer("trend", 0.01, MLMode.FILTER),
            "momentum": _scorer("momentum", 0.99, MLMode.FILTER),
        },
        code_version="lot7-test",
    ).run(_strategies(), datasets, settings.context, config)
    assert filtered.ml_mode == "FILTER"
    assert all(item.strategy_name == "momentum" for item in filtered.portfolio_opportunities)
    assert all(
        item.signal_id.startswith("signal-momentum")
        for item in filtered.portfolio_opportunities
    )


def test_portfolio_never_bypasses_deny_all_even_after_ml_and_policy_pass() -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    result = BacktestEngine(
        risk_engine=DenyAllRiskEngine(),
        regime_detector=BalancedRegimeDetector.from_profile(settings.profile),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(settings.profile),
        portfolio_engine=BalancedPortfolioEngine.from_profile(settings.profile),
        ml_scorers={
            "trend": _scorer("trend", 0.99, MLMode.FILTER),
            "momentum": _scorer("momentum", 0.99, MLMode.FILTER),
        },
        code_version="lot7-test",
    ).run(
        _strategies(),
        (_rising("AAPL", slope=1), _rising("MSFT", slope=2)),
        settings.context,
        BacktestConfig(primary_timeframe="1d", benchmark_symbol="AAPL"),
    )
    assert result.portfolio_plans
    assert result.risk_decisions
    assert all(item.status.value == "REJECT" for item in result.risk_decisions)
    assert result.fills == ()


def test_multi_strategy_run_is_deterministic_under_strategy_order_permutation() -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    datasets = (_rising("AAPL", slope=1), _rising("MSFT", slope=2))
    first = _engine(settings.profile).run(
        (
            TrendFollowingStrategy(("AAPL", "MSFT"), "1d"),
            MomentumStrategy(("AAPL", "MSFT"), "1d", MomentumConfig(top_k=2)),
        ),
        datasets,
        settings.context,
        BacktestConfig(primary_timeframe="1d", benchmark_symbol="AAPL"),
    )
    second = _engine(settings.profile).run(
        (
            MomentumStrategy(("AAPL", "MSFT"), "1d", MomentumConfig(top_k=2)),
            TrendFollowingStrategy(("AAPL", "MSFT"), "1d"),
        ),
        datasets,
        settings.context,
        BacktestConfig(primary_timeframe="1d", benchmark_symbol="AAPL"),
    )
    assert first.portfolio_opportunities == second.portfolio_opportunities
    assert first.portfolio_plans == second.portfolio_plans
    assert first.orders == second.orders
    assert first.fills == second.fills
    assert first.metrics == second.metrics
    assert first.result_hash == second.result_hash


def test_research_comparison_requires_identical_single_and_multi_assumptions() -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    datasets = (_rising("AAPL", slope=1), _rising("MSFT", slope=2))
    config = BacktestConfig(primary_timeframe="1d", benchmark_symbol="AAPL")
    singles = tuple(
        BacktestEngine(
            risk_engine=BalancedRiskEngine.from_profile(settings.profile),
            regime_detector=BalancedRegimeDetector.from_profile(settings.profile),
            activation_policy=BalancedStrategyActivationPolicy.from_profile(
                settings.profile
            ),
            code_version="lot7-test",
        ).run(strategy, datasets, settings.context, config)
        for strategy in _strategies()
    )
    multi = _engine(settings.profile).run(
        _strategies(), datasets, settings.context, config
    )
    comparison = compare_single_to_multi(singles, multi)

    assert comparison.assumptions_verified is True
    assert comparison.automatic_winner_selection is False
    assert tuple(
        item.strategy_name for item in comparison.single_strategy_runs
    ) == ("momentum", "trend")
    assert comparison.multi_strategy_run.strategy_name == "multi-strategy-portfolio"
    mismatched = replace(
        singles[0],
        config=replace(config, spread_bps=Decimal("1")),
    )
    with pytest.raises(ValueError, match="identical datasets"):
        compare_single_to_multi((mismatched, singles[1]), multi)


def test_all_four_production_baselines_run_in_one_shared_portfolio() -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    strategies = (
        TrendFollowingStrategy(("AAPL", "MSFT", "SPY"), "1d"),
        MomentumStrategy(
            ("AAPL", "MSFT", "SPY"),
            "1d",
            MomentumConfig(top_k=2),
        ),
        BreakoutStrategy(("AAPL", "MSFT", "SPY"), "1d"),
        MeanReversionStrategy(("AAPL", "MSFT", "SPY"), "1d"),
    )
    result = _engine(settings.profile).run(
        strategies,
        (
            _rising("AAPL", slope=1, count=150),
            _rising("MSFT", slope=2, count=150),
            _oscillating("SPY"),
        ),
        settings.context,
        BacktestConfig(primary_timeframe="1d", benchmark_symbol="SPY"),
    )

    parameters = dict(result.strategy_parameters)
    assert result.strategy_name == "multi-strategy-portfolio"
    assert {
        name.split(".")[1]
        for name in parameters
        if name.startswith("strategy.")
    } == {"trend", "momentum", "breakout", "mean-reversion"}
    assert result.portfolio_metrics is not None
    assert result.portfolio_metrics.max_unique_positions <= 5
    assert result.portfolio_metrics.max_gross_exposure <= 0.60
    assert all(order.portfolio_plan_id for order in result.orders)
    assert all(order.risk_decision_id for order in result.orders)


def test_future_next_open_cannot_change_the_portfolio_plan_at_signal_time() -> None:
    settings = load_runtime_settings("PAPER", "balanced")
    original_datasets = (_rising("AAPL", slope=1), _rising("MSFT", slope=2))
    config = BacktestConfig(primary_timeframe="1d", benchmark_symbol="AAPL")
    original = _engine(settings.profile).run(
        _strategies(), original_datasets, settings.context, config
    )
    first_plan = original.portfolio_plans[0]
    proposal_symbol = first_plan.orders_to_create[0].symbol
    changed_datasets = []
    for item in original_datasets:
        changed_bars = []
        future_changed = False
        for market_bar in item.bars:
            if (
                item.reference.symbol == proposal_symbol
                and market_bar.timestamp > first_plan.timestamp
                and not future_changed
            ):
                market_bar = replace(
                    market_bar,
                    open=Decimal("10000"),
                    high=Decimal("10001"),
                )
                future_changed = True
            changed_bars.append(market_bar)
        changed_datasets.append(dataset(changed_bars))

    changed = _engine(settings.profile).run(
        _strategies(), tuple(changed_datasets), settings.context, config
    )
    assert changed.portfolio_plans[0] == first_plan


def test_portfolio_export_1_5_is_hashed_and_contains_all_lineage_tables(
    tmp_path,
) -> None:
    import pyarrow.parquet as parquet

    settings = load_runtime_settings("PAPER", "balanced")
    result = _engine(settings.profile).run(
        (
            TrendFollowingStrategy(("AAPL", "MSFT"), "1d"),
            MomentumStrategy(("AAPL", "MSFT"), "1d", MomentumConfig(top_k=2)),
        ),
        (_rising("AAPL", slope=1), _rising("MSFT", slope=2)),
        settings.context,
        BacktestConfig(primary_timeframe="1d", benchmark_symbol="AAPL"),
    )
    store = BacktestResultStore(tmp_path / "backtests")
    directory = store.export(result)
    inspected = store.inspect(result.run_id)

    assert inspected["schema_version"] == "1.6"
    assert inspected["portfolio"]["engine_name"] == "balanced-portfolio"
    assert store.verify_integrity(result.run_id) is True
    for name, expected_rows in (
        ("portfolio_opportunities.parquet", len(result.portfolio_opportunities)),
        ("portfolio_decisions.parquet", len(result.portfolio_decisions)),
        ("portfolio_targets.parquet", len(result.portfolio_targets)),
        ("portfolio_sleeves.parquet", len(result.portfolio_sleeves)),
    ):
        assert parquet.read_table(directory / name).num_rows == expected_rows
