from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

from trading_ai.core.config import inspect_profile, load_profile
from trading_ai.core.models import OrderSide, PortfolioSnapshot, Position
from trading_ai.data.engine import DataEngine
from trading_ai.data.models import CacheMode
from trading_ai.data.storage import ParquetDataStore
from trading_ai.features.models import ReturnObservation, ReturnSeries
from trading_ai.cli import build_parser, main
from trading_ai.portfolio import (
    BalancedPortfolioEngine,
    CurrencyConverter,
    PortfolioAction,
    PortfolioContext,
    PortfolioDecisionBatch,
    PortfolioDecisionStatus,
    PortfolioOpportunity,
    PendingPortfolioOrder,
    inspect_portfolio_config,
    load_asset_currencies,
    load_balanced_portfolio_config,
    portfolio_config_hash,
)
from trading_ai.portfolio.exceptions import (
    CurrencyConversionError,
    PortfolioConfigurationError,
)
from trading_ai.risk.config import load_balanced_risk_config


NOW = datetime(2024, 6, 3, 20, tzinfo=timezone.utc)


class FixedFxRateBook(CurrencyConverter):
    def __init__(self, rates: dict[tuple[str, str], Decimal]) -> None:
        self.rates = rates

    def has_rate(self, from_currency, to_currency, timestamp) -> bool:
        del timestamp
        return from_currency == to_currency or (from_currency, to_currency) in self.rates

    def convert(self, amount, from_currency, to_currency, timestamp):
        del timestamp
        if from_currency == to_currency:
            return amount
        try:
            return amount * self.rates[(from_currency, to_currency)]
        except KeyError as exc:
            raise CurrencyConversionError("missing test FX") from exc


def engine(*, fx: CurrencyConverter | None = None) -> BalancedPortfolioEngine:
    profile = load_profile("balanced")
    risk, groups = load_balanced_risk_config(profile)
    config, currencies = load_balanced_portfolio_config(profile, risk)
    result = BalancedPortfolioEngine(
        config,
        currencies,
        asset_groups=groups.symbol_mapping,
        currency_converter=fx,
    )
    result.reset(NOW, Decimal("100000"))
    return result


def constrained_engine(*, max_positions: int) -> BalancedPortfolioEngine:
    profile = load_profile("balanced")
    risk, groups = load_balanced_risk_config(profile)
    config, currencies = load_balanced_portfolio_config(profile, risk)
    result = BalancedPortfolioEngine(
        replace(
            config,
            max_unique_positions=max_positions,
            max_entry_turnover_per_cycle=Decimal("1"),
        ),
        currencies,
        asset_groups=groups.symbol_mapping,
    )
    result.reset(NOW, Decimal("100000"))
    return result


def opportunity(
    strategy: str,
    symbol: str,
    *,
    action: PortfolioAction = PortfolioAction.ENTER_LONG,
    strength: float = 0.7,
    activation: str = "1",
    sequence: int = 1,
) -> PortfolioOpportunity:
    signal_id = f"signal-{strategy}-{symbol}-{sequence}"
    return PortfolioOpportunity(
        opportunity_id=f"opp-{strategy}-{symbol}-{sequence}",
        timestamp=NOW,
        symbol=symbol,
        strategy_name=strategy,
        strategy_version="1.0",
        signal_id=signal_id,
        action=action,
        signal_strength=strength,
        ml_mode="DISABLED",
        ml_prediction_id=None,
        ml_decision_id=None,
        activation_decision_id=f"activation-{signal_id}",
        activation_multiplier=Decimal(activation),
        regime_snapshot_id=f"regime-{symbol}",
        current_sleeve_weight=Decimal("0"),
        reason="synthetic point-in-time candidate",
    )


def context(
    allocator: BalancedPortfolioEngine,
    opportunities: tuple[PortfolioOpportunity, ...],
    *,
    positions: tuple[Position, ...] = (),
    prices: dict[str, Decimal] | None = None,
    pending_orders: tuple[PendingPortfolioOrder, ...] = (),
    return_series: tuple[ReturnSeries, ...] = (),
) -> PortfolioContext:
    prices = prices or {item.symbol: Decimal("100") for item in opportunities}
    equity = Decimal("100000")
    market_value = sum(
        (item.quantity * prices[item.symbol] for item in positions), Decimal("0")
    )
    return PortfolioContext(
        timestamp=NOW,
        portfolio=PortfolioSnapshot(
            NOW,
            equity - market_value,
            equity,
            positions,
        ),
        pending_orders=pending_orders,
        sleeve_state=allocator.sleeve_state,
        opportunities=opportunities,
        market_prices=tuple(sorted(prices.items())),
        return_series=return_series,
        asset_groups=allocator.asset_groups,
        asset_currencies=allocator.currencies.currencies,
        portfolio_config=allocator.config_parameters,
    )


def plan(
    allocator: BalancedPortfolioEngine,
    opportunities: tuple[PortfolioOpportunity, ...],
    **kwargs,
):
    return allocator.plan(
        PortfolioDecisionBatch(NOW, opportunities),
        context(allocator, opportunities, **kwargs),
    )


def test_balanced_portfolio_config_is_bounded_and_hash_is_deterministic() -> None:
    profile = load_profile("balanced")
    risk, _ = load_balanced_risk_config(profile)
    config, currencies = load_balanced_portfolio_config(profile, risk)
    assert config.max_target_exposure == Decimal("0.6")
    assert config.max_target_per_symbol == Decimal("0.15")
    assert config.max_unique_positions == 5
    assert config.sleeve_budget_total == Decimal("0.60")
    assert portfolio_config_hash(config, currencies) == portfolio_config_hash(config, currencies)


def test_aggressive_portfolio_is_disabled_and_cannot_be_loaded() -> None:
    assert inspect_portfolio_config("aggressive").enabled is False
    risk, _ = load_balanced_risk_config(load_profile("balanced"))
    with pytest.raises(PortfolioConfigurationError, match="aggressive"):
        load_balanced_portfolio_config(inspect_profile("aggressive"), risk)


def test_config_above_risk_limits_fails_closed() -> None:
    profile = load_profile("balanced")
    risk, _ = load_balanced_risk_config(profile)
    config = inspect_portfolio_config("balanced")
    currencies = load_asset_currencies()
    with pytest.raises(PortfolioConfigurationError):
        load_balanced_portfolio_config(profile, replace(risk, max_positions=4))
    with pytest.raises(PortfolioConfigurationError):
        load_balanced_portfolio_config(
            profile,
            replace(risk, max_portfolio_exposure=Decimal("0.59")),
        )
    with pytest.raises(PortfolioConfigurationError):
        load_balanced_portfolio_config(
            profile,
            replace(risk, max_single_position_exposure=Decimal("0.14")),
        )
    with pytest.raises(PortfolioConfigurationError):
        replace(config, strategy_sleeves=tuple(
            replace(item, budget_weight=Decimal("0.20"))
            for item in config.strategy_sleeves
        ))
    with pytest.raises(ValueError):
        replace(
            config.strategy_sleeves[0],
            budget_weight=Decimal("-0.01"),
        )
    with pytest.raises(PortfolioConfigurationError):
        replace(config, strategy_sleeves=config.strategy_sleeves[:-1])
    assert currencies.currency_for("MC.PA") == "EUR"


def test_equal_weight_within_sleeve_and_unused_budget_stays_cash() -> None:
    allocator = engine()
    opportunities = tuple(opportunity("trend", symbol, sequence=index) for index, symbol in enumerate(("AAPL", "MSFT", "NVDA"), 1))
    result = plan(allocator, opportunities)
    assert [target.target_weight for target in result.plan.targets] == [Decimal("0.05")] * 3
    assert result.plan.target_exposure_after == Decimal("0.15")
    assert result.plan.target_cash_fraction == Decimal("0.85")


def test_batch_order_invariance_and_stable_intra_strategy_tie_break() -> None:
    values = (
        opportunity("trend", "MSFT", strength=0.7, sequence=1),
        opportunity("trend", "AAPL", strength=0.7, sequence=2),
    )
    first = plan(engine(), values)
    second = plan(engine(), tuple(reversed(values)))
    assert first.ranked_opportunities == second.ranked_opportunities
    assert first.plan.targets == second.plan.targets
    assert first.plan.orders_to_create == second.plan.orders_to_create
    assert first.plan.plan_id == second.plan.plan_id


def test_same_symbol_contributions_aggregate_to_one_capped_target() -> None:
    allocator = engine()
    values = (
        opportunity("trend", "AAPL"),
        opportunity("momentum", "AAPL"),
    )
    result = plan(allocator, values)
    assert len(result.plan.targets) == 1
    target = result.plan.targets[0]
    assert target.symbol == "AAPL"
    assert target.target_weight == Decimal("0.15")
    assert {item.strategy_name for item in target.contributors} == {"trend", "momentum"}
    assert len(result.plan.orders_to_create) == 1


def test_one_sleeve_exit_preserves_other_and_all_exits_target_zero() -> None:
    allocator = engine()
    entered = plan(
        allocator,
        (opportunity("trend", "AAPL"), opportunity("momentum", "AAPL")),
    )
    assert entered.plan.targets[0].target_weight == Decimal("0.15")
    exit_one = opportunity("trend", "AAPL", action=PortfolioAction.EXIT_LONG, sequence=2)
    reduced = plan(
        allocator,
        (exit_one,),
        positions=(Position("AAPL", Decimal("150"), Decimal("100")),),
        prices={"AAPL": Decimal("100")},
    )
    assert reduced.plan.targets[0].target_weight == Decimal("0.075")
    assert {item.strategy_name for item in reduced.plan.targets[0].contributors} == {"momentum"}
    exit_two = opportunity("momentum", "AAPL", action=PortfolioAction.EXIT_LONG, sequence=3)
    closed = plan(
        allocator,
        (exit_two,),
        positions=(Position("AAPL", Decimal("150"), Decimal("100")),),
        prices={"AAPL": Decimal("100")},
    )
    assert closed.plan.targets[0].target_weight == Decimal("0")
    assert closed.plan.orders_to_create[0].side is OrderSide.SELL


def test_same_cycle_entry_and_exit_are_netted_to_one_symbol_target() -> None:
    allocator = engine()
    plan(allocator, (opportunity("trend", "AAPL"),))
    result = plan(
        allocator,
        (
            opportunity(
                "trend",
                "AAPL",
                action=PortfolioAction.EXIT_LONG,
                sequence=2,
            ),
            opportunity("momentum", "AAPL", sequence=3),
        ),
        positions=(Position("AAPL", Decimal("150"), Decimal("100")),),
        prices={"AAPL": Decimal("100")},
    )
    assert len(result.plan.targets) == 1
    assert result.plan.targets[0].target_weight == Decimal("0.15")
    assert result.plan.orders_to_create == ()
    assert {
        item.status for item in result.decisions
    } == {PortfolioDecisionStatus.EXIT, PortfolioDecisionStatus.NO_CHANGE}


def test_turnover_defers_new_entries_but_not_exit() -> None:
    allocator = engine()
    values = tuple(
        opportunity(strategy, symbol)
        for strategy, symbol in (
            ("trend", "AAPL"),
            ("momentum", "MSFT"),
            ("breakout", "NVDA"),
            ("mean-reversion", "SPY"),
        )
    )
    result = plan(allocator, values)
    assert sum(item.status is PortfolioDecisionStatus.SELECT for item in result.decisions) == 1
    assert sum(item.status is PortfolioDecisionStatus.DEFER for item in result.decisions) == 3
    selected = next(item for item in result.decisions if item.status is PortfolioDecisionStatus.SELECT)
    selected_opportunity = next(item for item in result.ranked_opportunities if item.opportunity_id == selected.opportunity_id)
    exiting = opportunity(selected_opportunity.strategy_name, selected_opportunity.symbol, action=PortfolioAction.EXIT_LONG, sequence=9)
    exit_result = plan(
        allocator,
        (exiting,),
        positions=(Position(selected_opportunity.symbol, Decimal("150"), Decimal("100")),),
        prices={selected_opportunity.symbol: Decimal("100")},
    )
    assert exit_result.decisions[0].status is PortfolioDecisionStatus.EXIT
    assert exit_result.plan.orders_to_create[0].side is OrderSide.SELL


def test_no_trade_band_uses_current_close_and_not_a_future_open() -> None:
    allocator = engine()
    entered = plan(allocator, (opportunity("trend", "AAPL"),))
    assert entered.plan.orders_to_create[0].quantity == Decimal("150.000000")
    # A future open is intentionally absent from PortfolioContext; changing any
    # unobserved future value cannot alter this close(t)-sized plan.
    unchanged = entered.plan.orders_to_create[0].quantity
    assert unchanged == Decimal("150.000000")


def test_mixed_currency_is_fail_closed_without_fx_and_works_with_explicit_fx() -> None:
    no_fx = plan(engine(), (opportunity("trend", "MC.PA"),))
    assert no_fx.decisions[0].status is PortfolioDecisionStatus.REJECT
    assert "FX_RATE_UNAVAILABLE" in no_fx.decisions[0].reason_codes
    with_fx = plan(
        engine(fx=FixedFxRateBook({("EUR", "USD"): Decimal("1.10")})),
        (opportunity("trend", "MC.PA"),),
    )
    assert with_fx.decisions[0].status is PortfolioDecisionStatus.SELECT
    assert with_fx.plan.orders_to_create[0].quantity == Decimal("136.363636")


def test_score_probability_is_not_an_allocation_input() -> None:
    low = replace(
        opportunity("trend", "AAPL"),
        ml_mode="FILTER",
        ml_prediction_id="pred-low",
        ml_decision_id="decision-low",
    )
    high = replace(
        low,
        opportunity_id="opp-high",
        signal_id="signal-high",
        ml_prediction_id="pred-high",
        ml_decision_id="decision-high",
    )
    assert plan(engine(), (low,)).plan.targets[0].target_weight == plan(engine(), (high,)).plan.targets[0].target_weight


def test_unknown_currency_never_defaults_to_usd() -> None:
    allocator = engine()
    unknown = opportunity("trend", "UNKNOWN")
    custom = context(
        allocator,
        (unknown,),
        prices={"UNKNOWN": Decimal("100")},
    )
    result = allocator.plan(PortfolioDecisionBatch(NOW, (unknown,)), custom)
    assert result.decisions[0].status is PortfolioDecisionStatus.REJECT
    assert "UNKNOWN_CURRENCY" in result.decisions[0].reason_codes


def test_no_trade_band_and_pending_orders_never_duplicate_or_cross() -> None:
    allocator = engine()
    first = plan(allocator, (opportunity("trend", "AAPL"),))
    assert first.plan.orders_to_create
    repeated = opportunity("trend", "AAPL", sequence=2)
    almost_target = Position("AAPL", Decimal("149"), Decimal("100"))
    band = plan(
        allocator,
        (repeated,),
        positions=(almost_target,),
        prices={"AAPL": Decimal("100")},
    )
    assert band.plan.orders_to_create == ()
    assert "NO_TRADE_BAND" in band.decisions[0].reason_codes

    pending_buy = PendingPortfolioOrder(
        "pending-buy", "AAPL", OrderSide.BUY, Decimal("150"), NOW
    )
    same_direction = allocator.plan(
        PortfolioDecisionBatch(NOW, (repeated,)),
        context(
            allocator,
            (repeated,),
            prices={"AAPL": Decimal("100")},
            pending_orders=(pending_buy,),
        ),
    )
    assert same_direction.plan.orders_to_create == ()

    pending_sell = PendingPortfolioOrder(
        "pending-sell", "AAPL", OrderSide.SELL, Decimal("10"), NOW
    )
    conflict = allocator.plan(
        PortfolioDecisionBatch(NOW, (repeated,)),
        context(
            allocator,
            (repeated,),
            prices={"AAPL": Decimal("100")},
            pending_orders=(pending_sell,),
        ),
    )
    assert conflict.plan.orders_to_create == ()
    assert conflict.plan.orders_to_defer == ("AAPL",)


def test_cross_currency_exit_remains_possible_without_fx() -> None:
    allocator = engine()
    exiting = opportunity(
        "trend", "MC.PA", action=PortfolioAction.EXIT_LONG, sequence=8
    )
    result = plan(
        allocator,
        (exiting,),
        positions=(Position("MC.PA", Decimal("10"), Decimal("100")),),
        prices={"MC.PA": Decimal("100")},
    )
    assert result.plan.orders_to_create[0].side is OrderSide.SELL
    assert result.plan.orders_to_create[0].quantity == Decimal("10")


def _returns(symbol: str, values: tuple[float, ...]) -> ReturnSeries:
    return ReturnSeries(
        symbol,
        "1d",
        tuple(
            ReturnObservation(NOW - timedelta(days=len(values) - index), value)
            for index, value in enumerate(values)
        ),
    )


def test_soft_correlation_prefers_less_correlated_candidate_at_equal_rank() -> None:
    allocator = constrained_engine(max_positions=2)
    plan(allocator, (opportunity("trend", "AAPL"),))
    candidates = (
        opportunity("trend", "MSFT", sequence=2),
        opportunity("trend", "NVDA", sequence=3),
    )
    base = tuple(float(index) / 100 for index in range(1, 21))
    series = (
        _returns("AAPL", base),
        _returns("MSFT", base),
        _returns("NVDA", tuple(-value for value in base)),
    )
    result = allocator.plan(
        PortfolioDecisionBatch(NOW, candidates),
        context(
            allocator,
            candidates,
            prices={"AAPL": Decimal("100"), "MSFT": Decimal("100"), "NVDA": Decimal("100")},
            return_series=series,
        ),
    )
    selected_ids = {
        item.opportunity_id
        for item in result.decisions
        if item.status is PortfolioDecisionStatus.SELECT
    }
    assert "opp-trend-NVDA-3" in selected_ids
    assert "opp-trend-MSFT-2" not in selected_ids


def test_group_diversification_and_unknown_correlation_are_conservative() -> None:
    allocator = constrained_engine(max_positions=2)
    plan(allocator, (opportunity("trend", "AAPL"),))
    candidates = (
        opportunity("trend", "MSFT", sequence=2),
        opportunity("trend", "SPY", sequence=3),
    )
    result = allocator.plan(
        PortfolioDecisionBatch(NOW, candidates),
        context(
            allocator,
            candidates,
            prices={"AAPL": Decimal("100"), "MSFT": Decimal("100"), "SPY": Decimal("100")},
        ),
    )
    selected = next(
        item for item in result.decisions if item.status is PortfolioDecisionStatus.SELECT
    )
    assert selected.signal_id == "signal-trend-SPY-3"
    assert "CORRELATION_UNKNOWN_DEPRIORITIZED" in selected.reason_codes


def test_unknown_group_is_deprioritized_against_a_known_group() -> None:
    profile = load_profile("balanced")
    risk, groups = load_balanced_risk_config(profile)
    config, currencies = load_balanced_portfolio_config(profile, risk)
    allocator = BalancedPortfolioEngine(
        replace(
            config,
            max_unique_positions=1,
            max_entry_turnover_per_cycle=Decimal("1"),
        ),
        currencies,
        asset_groups=tuple(
            item for item in groups.symbol_mapping if item[0] != "ASML"
        ),
    )
    allocator.reset(NOW, Decimal("100000"))
    values = (
        opportunity("trend", "ASML", sequence=1),
        opportunity("trend", "MSFT", sequence=2),
    )
    result = plan(
        allocator,
        values,
        prices={"ASML": Decimal("100"), "MSFT": Decimal("100")},
    )
    selected = next(
        item for item in result.decisions if item.status is PortfolioDecisionStatus.SELECT
    )
    assert selected.signal_id == "signal-trend-MSFT-2"


def test_portfolio_cli_inspects_balanced_and_keeps_aggressive_locked(capsys) -> None:
    assert main(["portfolio", "inspect", "--profile", "balanced", "--json"]) == 0
    balanced = json.loads(capsys.readouterr().out)
    assert balanced["engine"] == "balanced-portfolio"
    assert balanced["version"] == "1.0"
    assert balanced["limits"]["max_target_exposure"] == "0.6"
    assert sum(Decimal(value) for value in balanced["sleeves"].values()) == Decimal("0.60")
    assert balanced["unused_budget_policy"] == "CASH"
    assert balanced["risk_authority"] == "BalancedRiskEngine"

    assert main(["portfolio", "inspect", "--profile", "aggressive", "--json"]) == 0
    aggressive = json.loads(capsys.readouterr().out)
    assert aggressive["enabled"] is False
    assert aggressive["locked"] is True


def test_backtest_cli_accepts_repeated_strategies_and_explicit_model_mapping() -> None:
    parsed = build_parser().parse_args(
        [
            "backtest", "run",
            "--strategy", "trend",
            "--strategy", "momentum",
            "--symbol", "AAPL",
            "--symbol", "MSFT",
            "--symbol", "NVDA",
            "--timeframe", "1d",
            "--start", "2024-01-01",
            "--end", "2025-01-01",
            "--ml-mode", "filter",
            "--ml-model-id", "trend=model-trend",
            "--ml-model-id", "momentum=model-momentum",
        ]
    )
    assert parsed.strategy == ["trend", "momentum"]
    assert parsed.ml_model_id == [
        "trend=model-trend",
        "momentum=model-momentum",
    ]


def test_backtest_cli_runs_repeated_strategies_in_one_offline_portfolio(
    tmp_path,
    fake_data_provider,
    market_start,
    market_end,
    capsys,
) -> None:
    data_root = tmp_path / "data_local"
    fake_data_provider.datasets[("MSFT", "1d")] = tuple(
        replace(item, symbol="MSFT")
        for item in fake_data_provider.datasets[("AAPL", "1d")]
    )
    fake_data_provider.metadata_by_symbol["MSFT"] = replace(
        fake_data_provider.metadata_by_symbol["AAPL"],
        symbol="MSFT",
    )
    data_engine = DataEngine(fake_data_provider, ParquetDataStore(data_root))
    for symbol in ("AAPL", "MSFT"):
        data_engine.fetch(
            profile_name="balanced",
            symbol=symbol,
            timeframe="1d",
            start=market_start,
            end=market_end,
            cache_mode=CacheMode.REFRESH,
        )

    assert main(
        [
            "backtest", "run",
            "--strategy", "trend",
            "--strategy", "momentum",
            "--symbol", "AAPL",
            "--symbol", "MSFT",
            "--timeframe", "1d",
            "--top-k", "2",
            "--start", "2024-07-01",
            "--end", "2024-07-03",
            "--data-root", str(data_root),
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "multi-strategy-portfolio"
    assert payload["portfolio"]["engine"] == "balanced-portfolio"
    assert payload["risk"]["engine"] == "balanced-risk"


def test_portfolio_package_is_provider_broker_and_heavy_optimizer_free() -> None:
    forbidden_imports = (
        "yfinance",
        "requests",
        "BrokerAdapter",
        "IBKR",
        "tensorflow",
        "torch",
        "cvxpy",
        "sklearn",
    )
    forbidden_construction = (
        "Markowitz",
        "Black-Litterman",
        "Kelly",
        "risk parity",
        "winner gets more",
        "double after loss",
    )
    for path in Path("src/trading_ai/portfolio").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in (*forbidden_imports, *forbidden_construction):
            assert token.lower() not in source.lower(), f"{token} leaked into {path}"


def test_physical_positions_are_counted_when_enforcing_portfolio_slots() -> None:
    allocator = constrained_engine(max_positions=1)
    candidate = opportunity("trend", "MSFT")
    result = plan(
        allocator,
        (candidate,),
        positions=(Position("AAPL", Decimal("10"), Decimal("100")),),
        prices={"AAPL": Decimal("100"), "MSFT": Decimal("100")},
    )
    assert result.decisions[0].status is PortfolioDecisionStatus.DEFER
    assert "MAX_UNIQUE_POSITIONS" in result.decisions[0].reason_codes
