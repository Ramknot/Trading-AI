"""Chronological, offline, provider-neutral historical simulation engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from itertools import groupby
from typing import Any

from trading_ai.backtesting.base import Backtester
from trading_ai.backtesting.benchmark import BuyAndHoldBenchmark
from trading_ai.backtesting.exceptions import (
    BacktestConfigurationError,
    BacktestDataError,
)
from trading_ai.backtesting.execution import BarExecutionModel, ExecutionModel
from trading_ai.backtesting.metrics import MetricsEngine
from trading_ai.backtesting.models import (
    BacktestConfig,
    BacktestDataset,
    BacktestOrder,
    BacktestStatus,
    DataQualityPolicy,
    EquityPoint,
    Fill,
    OrderIntent,
    OrderStatus,
    StrategyContext,
    StrategySignal,
    StrategySignalAction,
)
from trading_ai.backtesting.portfolio import PortfolioLedger
from trading_ai.backtesting.reproducibility import (
    detect_git_commit,
    source_tree_hash,
    stable_hash,
    stable_result_hash,
)
from trading_ai.backtesting.strategy import BacktestStrategy
from trading_ai.backtesting.trades import reconstruct_trades
from trading_ai.core.config import PROJECT_ROOT, load_runtime_settings
from trading_ai.core.models import (
    BacktestResult,
    MarketBar,
    OrderRequest,
    OrderSide,
    PortfolioSnapshot,
    Position,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
)
from trading_ai.data.models import (
    CorporateAction,
    DataKind,
    Dividend,
    QualityStatus,
    StockSplit,
)
from trading_ai.features import FeatureEngine, FeatureRequest
from trading_ai.ml.base import MLScorer
from trading_ai.ml.decisions import MLFilterDecision, MLPrediction
from trading_ai.ml.models import MLFilterStatus, MLMode
from trading_ai.portfolio.base import PortfolioEngine
from trading_ai.portfolio.models import (
    PendingPortfolioOrder,
    PortfolioAction,
    PortfolioContext,
    PortfolioDecision,
    PortfolioDecisionBatch,
    PortfolioOpportunity,
    RebalancePlan,
    StrategySleeveState,
)
from trading_ai.portfolio.reporting import build_portfolio_metrics
from trading_ai.risk.base import RiskEngine
from trading_ai.risk.deny_all import DenyAllRiskEngine
from trading_ai.risk.models import RiskContext
from trading_ai.regimes.base import ActivationPolicy, RegimeDetector
from trading_ai.regimes.models import (
    ActivationDecision,
    ActivationStatus,
    RegimeSnapshot,
)
from trading_ai.regimes.reporting import build_regime_report


ExecutionModelFactory = Callable[[BacktestConfig], ExecutionModel]


def _event_key(bar: MarketBar) -> tuple[datetime, str, str]:
    return bar.timestamp, bar.symbol, bar.timeframe


def _action_key(action: CorporateAction) -> tuple[datetime, str, str, str, str]:
    return (
        action.timestamp,
        action.symbol,
        action.action_type.value,
        str(action.value),
        action.source,
    )


class BacktestEngine(Backtester):
    """Fail-closed event loop whose strategy boundary exposes no future bars.

    An intent emitted at a bar close is created at that bar timestamp. The
    execution model can only fill it on a strictly later eligible bar, so the
    default MARKET convention is next-bar open.
    """

    def __init__(
        self,
        *,
        execution_model_factory: ExecutionModelFactory | None = None,
        metrics_engine: MetricsEngine | None = None,
        risk_engine: RiskEngine | None = None,
        feature_engine: FeatureEngine | None = None,
        regime_detector: RegimeDetector | None = None,
        activation_policy: ActivationPolicy | None = None,
        ml_scorer: MLScorer | None = None,
        ml_scorers: Mapping[str, MLScorer] | None = None,
        portfolio_engine: PortfolioEngine | None = None,
        code_version: str | None = None,
    ) -> None:
        if (regime_detector is None) != (activation_policy is None):
            raise BacktestConfigurationError(
                "regime_detector and activation_policy must be injected together"
            )
        if (
            ml_scorer is not None
            and ml_scorer.mode is not MLMode.DISABLED
            and regime_detector is None
        ):
            raise BacktestConfigurationError(
                "active ML scoring requires regime detector and activation policy"
            )
        if ml_scorer is not None and ml_scorers:
            raise BacktestConfigurationError(
                "inject ml_scorer or strategy-keyed ml_scorers, not both"
            )
        if any(
            scorer.mode is not MLMode.DISABLED for scorer in (ml_scorers or {}).values()
        ) and regime_detector is None:
            raise BacktestConfigurationError(
                "active strategy-keyed ML requires regime detector and policy"
            )
        self._execution_model_factory = (
            execution_model_factory or BarExecutionModel
        )
        self._metrics_engine = metrics_engine or MetricsEngine()
        self._risk_engine = risk_engine or DenyAllRiskEngine()
        self._feature_engine = feature_engine or FeatureEngine()
        self._regime_detector = regime_detector
        self._activation_policy = activation_policy
        self._ml_scorer = ml_scorer
        self._ml_scorers = dict(ml_scorers or {})
        self._portfolio_engine = portfolio_engine
        self._code_version = code_version

    @property
    def risk_engine(self) -> RiskEngine:
        """Expose the mandatory gate for diagnostics without allowing bypass."""

        return self._risk_engine

    def run(
        self,
        strategy: BacktestStrategy | Sequence[BacktestStrategy],
        datasets: Sequence[BacktestDataset],
        context: TradingContext,
        config: BacktestConfig,
    ) -> BacktestResult:
        settings = load_runtime_settings(context.environment, context.profile)
        profile = settings.profile
        strategies = (
            (strategy,)
            if isinstance(strategy, BacktestStrategy)
            else tuple(strategy)
        )
        if not strategies or any(not isinstance(item, BacktestStrategy) for item in strategies):
            raise BacktestConfigurationError("run requires BacktestStrategy instances")
        if len({item.name for item in strategies}) != len(strategies):
            raise BacktestConfigurationError("multi-strategy names must be unique")
        if len(strategies) > 1 and self._portfolio_engine is None:
            raise BacktestConfigurationError(
                "multiple strategies require an explicitly injected PortfolioEngine"
            )
        if self._portfolio_engine is not None and (
            self._regime_detector is None or self._activation_policy is None
        ):
            raise BacktestConfigurationError(
                "portfolio runs require regime detector and activation policy"
            )
        primary_strategy = strategies[0]
        active_modes = {
            scorer.mode
            for item in strategies
            if (scorer := self._scorer_for(item.name)) is not None
            and scorer.mode is not MLMode.DISABLED
        }
        if len(active_modes) > 1:
            raise BacktestConfigurationError(
                "all active multi-strategy ML scorers must use the same mode"
            )
        if len(strategies) > 1 and active_modes and any(
            self._scorer_for(item.name) is None
            or self._scorer_for(item.name).mode is MLMode.DISABLED  # type: ignore[union-attr]
            for item in strategies
        ):
            raise BacktestConfigurationError(
                "active multi-strategy ML requires an explicit scorer for every strategy"
            )
        for item in strategies:
            scorer = self._scorer_for(item.name)
            if scorer is None or scorer.mode is MLMode.DISABLED:
                continue
            artifact = scorer.artifact
            if artifact is None:
                raise BacktestConfigurationError("active ML mode requires model artifact")
            if (
                artifact.strategy_name != item.name
                or artifact.strategy_version != item.version
                or artifact.timeframe != config.primary_timeframe
            ):
                raise BacktestConfigurationError(
                    "ML model strategy/version/timeframe is incompatible with the run"
                )
        if config.allow_short and not profile.allow_short:
            raise BacktestConfigurationError(
                "backtest allow_short exceeds the active profile; Balanced is long-only"
            )
        if config.allow_short:
            raise BacktestConfigurationError(
                "short simulation is not implemented in Balanced V1"
            )
        normalized_datasets = tuple(datasets)
        if not normalized_datasets:
            raise BacktestDataError("at least one validated dataset is required")
        warnings = self._validate_datasets(
            normalized_datasets,
            profile.asset_universe,
            profile.timeframes,
            config,
        )
        bars = self._merge_bars(normalized_datasets)
        actions = self._merge_actions(normalized_datasets)
        available_series = {
            (dataset.reference.symbol, dataset.reference.timeframe)
            for dataset in normalized_datasets
        }
        if not any(
            timeframe == config.primary_timeframe
            for _, timeframe in available_series
        ):
            raise BacktestDataError(
                f"no dataset uses primary timeframe {config.primary_timeframe}"
            )
        secondary_timeframes = sorted(
            timeframe
            for _, timeframe in available_series
            if timeframe != config.primary_timeframe
        )
        if secondary_timeframes:
            raise BacktestConfigurationError(
                "Lot 2 runs one primary timeframe at a time; secondary datasets "
                "are reserved for a future extension: "
                f"{', '.join(sorted(set(secondary_timeframes)))}"
            )
        if (
            config.benchmark_symbol is not None
            and (config.benchmark_symbol, config.primary_timeframe)
            not in available_series
        ):
            raise BacktestDataError(
                f"benchmark dataset is missing for {config.benchmark_symbol} "
                f"{config.primary_timeframe}"
            )

        execution_model = self._execution_model_factory(config)
        if not isinstance(execution_model, ExecutionModel):
            raise BacktestConfigurationError(
                "execution_model_factory must return an ExecutionModel"
            )
        ledger = PortfolioLedger(config.starting_cash, allow_short=False)
        orders: list[BacktestOrder] = []
        order_indexes: dict[str, int] = {}
        pending_order_ids: list[str] = []
        fills: list[Fill] = []
        risk_decisions: list[RiskDecision] = []
        regime_snapshots: list[RegimeSnapshot] = []
        activation_decisions: list[ActivationDecision] = []
        ml_predictions: list[MLPrediction] = []
        ml_decisions: list[MLFilterDecision] = []
        portfolio_opportunities: list[PortfolioOpportunity] = []
        portfolio_decisions: list[PortfolioDecision] = []
        portfolio_plans: list[RebalancePlan] = []
        portfolio_sleeves: list[StrategySleeveState] = []
        pending_buy_risk: dict[str, tuple[str, Decimal, Decimal]] = {}
        pending_sell_risk: dict[str, tuple[str, Decimal]] = {}
        history: list[MarketBar] = []
        last_prices: dict[str, Decimal] = {}
        equity_curve: list[EquityPoint] = []
        action_index = 0
        for item in strategies:
            item.reset()
        self._feature_engine.clear_cache()
        self._risk_engine.reset(bars[0].timestamp, config.starting_cash)
        if self._portfolio_engine is not None:
            self._portfolio_engine.reset(bars[0].timestamp, config.starting_cash)
        if self._regime_detector is not None:
            self._regime_detector.reset()

        for timestamp, grouped in groupby(bars, key=lambda bar: bar.timestamp):
            current_bars = tuple(grouped)
            regimes_at_timestamp: dict[tuple[str, str], RegimeSnapshot] = {}
            portfolio_cycle_intents: list[
                tuple[BacktestStrategy, OrderIntent, MarketBar]
            ] = []
            while (
                action_index < len(actions)
                and actions[action_index].timestamp <= timestamp
            ):
                action = actions[action_index]
                if isinstance(action, Dividend):
                    ledger.apply_dividend(action)
                elif isinstance(action, StockSplit):
                    ledger.apply_split(action, last_prices)
                action_index += 1

            for bar in current_bars:
                last_prices[bar.symbol] = bar.close
                for order_id in tuple(pending_order_ids):
                    order = orders[order_indexes[order_id]]
                    if (
                        order.symbol != bar.symbol
                        or order.timeframe != bar.timeframe
                        or order.created_at >= bar.timestamp
                    ):
                        continue
                    order = replace(
                        order, eligible_bar_count=order.eligible_bar_count + 1
                    )
                    self._replace_order(order, orders, order_indexes)
                    fill = execution_model.try_fill(
                        order, bar, f"fill-{len(fills) + 1:06d}"
                    )
                    if fill is not None:
                        reason = ledger.validate_fill(fill)
                        if reason is None:
                            ledger.apply_fill(fill)
                            fills.append(fill)
                            order = replace(
                                order,
                                status=OrderStatus.FILLED,
                                completed_at=bar.timestamp,
                            )
                        else:
                            order = replace(
                                order,
                                status=OrderStatus.REJECTED,
                                status_reason=reason,
                                completed_at=bar.timestamp,
                            )
                        self._replace_order(order, orders, order_indexes)
                        pending_order_ids.remove(order_id)
                        pending_buy_risk.pop(order_id, None)
                        pending_sell_risk.pop(order_id, None)
                        continue
                    if (
                        config.order_expiration_bars is not None
                        and order.eligible_bar_count
                        >= config.order_expiration_bars
                    ):
                        order = replace(
                            order,
                            status=OrderStatus.EXPIRED,
                            status_reason="configured eligible-bar expiry reached",
                            completed_at=bar.timestamp,
                        )
                        self._replace_order(order, orders, order_indexes)
                        pending_order_ids.remove(order_id)
                        pending_buy_risk.pop(order_id, None)
                        pending_sell_risk.pop(order_id, None)

                history.append(bar)
                current_portfolio = ledger.snapshot(bar.timestamp, last_prices)
                self._risk_engine.observe(
                    bar.timestamp, current_portfolio.total_equity
                )
                current_regime = None
                if self._regime_detector is not None:
                    series_history = tuple(
                        item
                        for item in history
                        if item.symbol == bar.symbol
                        and item.timeframe == bar.timeframe
                    )
                    regime_features = self._feature_engine.compute(
                        series_history,
                        self._regime_detector.feature_request,
                        as_of=bar.timestamp,
                    )
                    current_regime = self._regime_detector.evaluate(regime_features)
                    regime_snapshots.append(current_regime)
                    regimes_at_timestamp[(bar.symbol, bar.timeframe)] = current_regime
                strategy_context = StrategyContext(
                    current_time=bar.timestamp,
                    current_bar=bar,
                    history=tuple(history),
                    portfolio=current_portfolio,
                    trading_context=context,
                    profile=profile,
                    current_regime=current_regime,
                    regime_history=tuple(regime_snapshots),
                )
                if self._portfolio_engine is not None:
                    for item_strategy in strategies:
                        item_context = replace(
                            strategy_context,
                            portfolio=self._strategy_portfolio_snapshot(
                                current_portfolio,
                                item_strategy.name,
                                self._portfolio_engine.sleeve_state,
                            ),
                        )
                        item_intents = tuple(item_strategy.on_bar(item_context))
                        if any(not isinstance(intent, OrderIntent) for intent in item_intents):
                            raise BacktestConfigurationError(
                                "strategy.on_bar must return only OrderIntent values"
                            )
                        portfolio_cycle_intents.extend(
                            (item_strategy, intent, bar) for intent in item_intents
                        )
                    continue
                strategy = primary_strategy
                intents = tuple(strategy.on_bar(strategy_context))
                for intent in intents:
                    if not isinstance(intent, OrderIntent):
                        raise BacktestConfigurationError(
                            "strategy.on_bar must return only OrderIntent values"
                        )
                    timeframe = intent.timeframe or bar.timeframe
                    if intent.symbol not in profile.asset_universe:
                        raise BacktestConfigurationError(
                            f"strategy requested symbol {intent.symbol!r} outside profile universe"
                        )
                    if (intent.symbol, timeframe) not in available_series:
                        raise BacktestDataError(
                            f"no supplied dataset for order series {intent.symbol} {timeframe}"
                        )
                    signal = None
                    regime = None
                    if (
                        self._activation_policy is not None
                        or (
                            self._ml_scorer is not None
                            and self._ml_scorer.mode is not MLMode.DISABLED
                        )
                    ):
                        signal = self._signal_for_intent(strategy, intent)
                        regime = regimes_at_timestamp.get((intent.symbol, timeframe))
                        if regime is None:
                            raise BacktestConfigurationError(
                                "guarded intent requires an exact current regime snapshot"
                            )
                    ml_decision_id = None
                    if (
                        self._ml_scorer is not None
                        and self._ml_scorer.mode is not MLMode.DISABLED
                    ):
                        assert signal is not None and regime is not None
                        ml_series_history = tuple(
                            item
                            for item in history
                            if item.symbol == intent.symbol
                            and item.timeframe == timeframe
                            and item.timestamp <= bar.timestamp
                        )
                        if (
                            not ml_series_history
                            or ml_series_history[-1].timestamp != bar.timestamp
                        ):
                            raise BacktestDataError(
                                "ML-scored intent requires an exact current source bar"
                            )
                        ml_features = self._feature_engine.compute(
                            ml_series_history,
                            self._ml_scorer.feature_request,
                            as_of=bar.timestamp,
                        )
                        prediction, ml_decision = self._ml_scorer.evaluate(
                            signal=signal,
                            features=ml_features,
                            regime=regime,
                        )
                        self._validate_ml_decision(signal, prediction, ml_decision)
                        if prediction is not None:
                            ml_predictions.append(prediction)
                        ml_decisions.append(ml_decision)
                        strategy.on_ml_decision(ml_decision)
                        ml_decision_id = ml_decision.decision_id
                        if (
                            signal.action is StrategySignalAction.ENTER_LONG
                            and self._ml_scorer.mode is MLMode.FILTER
                            and ml_decision.status is not MLFilterStatus.PASS
                        ):
                            continue
                    activation_decision_id = None
                    if self._activation_policy is not None:
                        assert signal is not None and regime is not None
                        activation = self._activation_policy.evaluate(
                            strategy_name=strategy.name,
                            strategy_version=strategy.version,
                            signal=signal,
                            regime=regime,
                            proposed_quantity=intent.quantity,
                        )
                        self._validate_activation_decision(intent, signal, activation)
                        activation_decisions.append(activation)
                        strategy.on_activation_decision(activation)
                        activation_decision_id = activation.decision_id
                        if activation.status is ActivationStatus.BLOCK:
                            continue
                        intent = replace(
                            intent,
                            quantity=activation.adjusted_quantity,
                        )
                    order_id = f"order-{len(orders) + 1:06d}"
                    observed_price = last_prices.get(intent.symbol)
                    if observed_price is None:
                        raise BacktestDataError(
                            f"no observed price exists for risk evaluation of {intent.symbol}"
                        )
                    if intent.side is OrderSide.BUY:
                        expected_entry_price = max(
                            value
                            for value in (
                                observed_price,
                                intent.expected_entry_price,
                                intent.limit_price,
                            )
                            if value is not None
                        )
                    else:
                        expected_entry_price = observed_price
                    order_request = OrderRequest(
                        order_id=order_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        quantity=intent.quantity,
                        order_type=intent.order_type,
                        limit_price=intent.limit_price,
                        strategy_decision_id=intent.signal_id,
                        ml_decision_id=ml_decision_id,
                        activation_decision_id=activation_decision_id,
                        created_at=bar.timestamp,
                        expected_entry_price=expected_entry_price,
                        invalidation_price=intent.invalidation_price,
                        risk_distance=intent.risk_distance,
                    )
                    risk_portfolio = self._portfolio_with_pending_buys(
                        ledger.snapshot(bar.timestamp, last_prices),
                        pending_buy_risk,
                    )
                    risk_state = self._risk_engine.current_state(
                        bar.timestamp, risk_portfolio.total_equity
                    )
                    series_history = tuple(
                        item
                        for item in history
                        if item.symbol == intent.symbol
                        and item.timeframe == timeframe
                    )
                    feature_snapshot = None
                    if (
                        series_history
                        and series_history[-1].timestamp == bar.timestamp
                    ):
                        feature_snapshot = self._feature_engine.compute(
                            series_history,
                            FeatureRequest(),
                            as_of=bar.timestamp,
                        )
                    return_series = self._feature_engine.return_series(
                        history,
                        symbols=tuple(
                            sorted(
                                symbol
                                for symbol, item_timeframe in available_series
                                if item_timeframe == timeframe
                            )
                        ),
                        timeframe=timeframe,
                        as_of=bar.timestamp,
                    )
                    risk_context = RiskContext(
                        timestamp=bar.timestamp,
                        profile=profile,
                        portfolio=risk_portfolio,
                        order=order_request,
                        expected_entry_price=expected_entry_price,
                        market_prices=tuple(sorted(last_prices.items())),
                        risk_state=risk_state,
                        timeframe=timeframe,
                        feature_snapshot=feature_snapshot,
                        return_series=return_series,
                        pending_sell_quantities=self._pending_sell_quantities(
                            pending_sell_risk
                        ),
                    )
                    risk_decision = self._risk_engine.evaluate_context(risk_context)
                    self._validate_risk_decision(order_request, risk_decision)
                    risk_decisions.append(risk_decision)
                    if risk_decision.status is RiskDecisionStatus.REJECT:
                        order = BacktestOrder(
                            order_id=order_id,
                            symbol=intent.symbol,
                            timeframe=timeframe,
                            side=intent.side,
                            quantity=intent.quantity,
                            order_type=intent.order_type,
                            created_at=bar.timestamp,
                            limit_price=intent.limit_price,
                            signal_id=intent.signal_id,
                            ml_decision_id=ml_decision_id,
                            activation_decision_id=activation_decision_id,
                            risk_decision_id=risk_decision.decision_id,
                            status=OrderStatus.REJECTED,
                            status_reason=risk_decision.reason,
                            completed_at=bar.timestamp,
                        )
                        order_indexes[order.order_id] = len(orders)
                        orders.append(order)
                        continue
                    approved_quantity = risk_decision.approved_quantity
                    if approved_quantity is None or approved_quantity <= Decimal("0"):
                        raise BacktestConfigurationError(
                            "approved risk decision requires a positive approved_quantity"
                        )
                    order = BacktestOrder(
                        order_id=order_id,
                        symbol=intent.symbol,
                        timeframe=timeframe,
                        side=intent.side,
                        quantity=approved_quantity,
                        order_type=intent.order_type,
                        created_at=bar.timestamp,
                        limit_price=intent.limit_price,
                        signal_id=intent.signal_id,
                        ml_decision_id=ml_decision_id,
                        activation_decision_id=activation_decision_id,
                        risk_decision_id=risk_decision.decision_id,
                    )
                    order_indexes[order.order_id] = len(orders)
                    orders.append(order)
                    pending_order_ids.append(order.order_id)
                    if order.side is OrderSide.BUY:
                        pending_buy_risk[order.order_id] = (
                            order.symbol,
                            order.quantity,
                            expected_entry_price,
                        )
                    else:
                        pending_sell_risk[order.order_id] = (
                            order.symbol,
                            order.quantity,
                        )
            if self._portfolio_engine is not None and portfolio_cycle_intents:
                cycle_opportunities: list[PortfolioOpportunity] = []
                for item_strategy, intent, source_bar in portfolio_cycle_intents:
                    timeframe = intent.timeframe or source_bar.timeframe
                    if intent.symbol not in profile.asset_universe:
                        raise BacktestConfigurationError(
                            f"strategy requested symbol {intent.symbol!r} outside profile universe"
                        )
                    if (intent.symbol, timeframe) not in available_series:
                        raise BacktestDataError(
                            f"no supplied dataset for order series {intent.symbol} {timeframe}"
                        )
                    signal = self._signal_for_intent(item_strategy, intent)
                    regime = regimes_at_timestamp.get((intent.symbol, timeframe))
                    if regime is None:
                        raise BacktestConfigurationError(
                            "portfolio opportunity requires an exact current regime snapshot"
                        )
                    scorer = self._scorer_for(item_strategy.name)
                    ml_decision_id = None
                    ml_prediction_id = None
                    ml_mode = MLMode.DISABLED
                    if scorer is not None:
                        ml_mode = scorer.mode
                    if scorer is not None and scorer.mode is not MLMode.DISABLED:
                        ml_series_history = tuple(
                            item
                            for item in history
                            if item.symbol == intent.symbol
                            and item.timeframe == timeframe
                            and item.timestamp <= timestamp
                        )
                        if (
                            not ml_series_history
                            or ml_series_history[-1].timestamp != timestamp
                        ):
                            raise BacktestDataError(
                                "ML-scored portfolio opportunity requires an exact current bar"
                            )
                        ml_features = self._feature_engine.compute(
                            ml_series_history,
                            scorer.feature_request,
                            as_of=timestamp,
                        )
                        prediction, ml_decision = scorer.evaluate(
                            signal=signal,
                            features=ml_features,
                            regime=regime,
                        )
                        self._validate_ml_decision(signal, prediction, ml_decision)
                        if prediction is not None:
                            ml_predictions.append(prediction)
                            ml_prediction_id = prediction.prediction_id
                        ml_decisions.append(ml_decision)
                        item_strategy.on_ml_decision(ml_decision)
                        ml_decision_id = ml_decision.decision_id
                        if (
                            signal.action is StrategySignalAction.ENTER_LONG
                            and scorer.mode is MLMode.FILTER
                            and ml_decision.status is not MLFilterStatus.PASS
                        ):
                            continue
                    assert self._activation_policy is not None
                    activation = self._activation_policy.evaluate(
                        strategy_name=item_strategy.name,
                        strategy_version=item_strategy.version,
                        signal=signal,
                        regime=regime,
                        proposed_quantity=intent.quantity,
                    )
                    self._validate_activation_decision(intent, signal, activation)
                    activation_decisions.append(activation)
                    item_strategy.on_activation_decision(activation)
                    if activation.status is ActivationStatus.BLOCK:
                        continue
                    current_sleeve = next(
                        (
                            state.target_weight_contribution
                            for state in self._portfolio_engine.sleeve_state
                            if state.strategy_name == item_strategy.name
                            and state.symbol == intent.symbol
                        ),
                        Decimal("0"),
                    )
                    action = (
                        PortfolioAction.ENTER_LONG
                        if signal.action is StrategySignalAction.ENTER_LONG
                        else PortfolioAction.EXIT_LONG
                    )
                    opportunity_payload = {
                        "timestamp": timestamp,
                        "signal_id": signal.signal_id,
                        "ml_decision_id": ml_decision_id,
                        "activation_decision_id": activation.decision_id,
                        "portfolio_config_hash": self._portfolio_engine.config_hash,
                    }
                    opportunity = PortfolioOpportunity(
                        opportunity_id=(
                            "portfolio-opportunity-"
                            + stable_hash(opportunity_payload)[:24]
                        ),
                        timestamp=timestamp,
                        symbol=intent.symbol,
                        strategy_name=item_strategy.name,
                        strategy_version=item_strategy.version,
                        signal_id=signal.signal_id,
                        action=action,
                        signal_strength=signal.strength,
                        ml_mode=ml_mode.value,
                        ml_prediction_id=ml_prediction_id,
                        ml_decision_id=ml_decision_id,
                        activation_decision_id=activation.decision_id,
                        activation_multiplier=activation.allocation_multiplier,
                        regime_snapshot_id=regime.snapshot_id,
                        current_sleeve_weight=current_sleeve,
                        reason=signal.reason,
                        timeframe=timeframe,
                    )
                    cycle_opportunities.append(opportunity)

                if cycle_opportunities:
                    normalized_opportunities = tuple(
                        sorted(
                            cycle_opportunities,
                            key=lambda item: (
                                item.strategy_name,
                                item.symbol,
                                item.signal_id,
                                item.opportunity_id,
                            ),
                        )
                    )
                    current_portfolio = ledger.snapshot(timestamp, last_prices)
                    pending_context = tuple(
                        PendingPortfolioOrder(
                            order_id=order_id,
                            symbol=orders[order_indexes[order_id]].symbol,
                            side=orders[order_indexes[order_id]].side,
                            quantity=orders[order_indexes[order_id]].quantity,
                            created_at=orders[order_indexes[order_id]].created_at,
                        )
                        for order_id in sorted(pending_order_ids)
                    )
                    return_series = self._feature_engine.return_series(
                        history,
                        symbols=tuple(
                            sorted(
                                symbol
                                for symbol, item_timeframe in available_series
                                if item_timeframe == config.primary_timeframe
                            )
                        ),
                        timeframe=config.primary_timeframe,
                        as_of=timestamp,
                    )
                    asset_groups = tuple(
                        sorted(getattr(self._portfolio_engine, "asset_groups", ()))
                    )
                    asset_currencies = tuple(
                        sorted(
                            getattr(
                                getattr(self._portfolio_engine, "currencies", None),
                                "currencies",
                                (),
                            )
                        )
                    )
                    portfolio_context = PortfolioContext(
                        timestamp=timestamp,
                        portfolio=current_portfolio,
                        pending_orders=pending_context,
                        sleeve_state=self._portfolio_engine.sleeve_state,
                        opportunities=normalized_opportunities,
                        market_prices=tuple(sorted(last_prices.items())),
                        return_series=return_series,
                        asset_groups=asset_groups,
                        asset_currencies=asset_currencies,
                        portfolio_config=self._portfolio_engine.config_parameters,
                    )
                    plan_result = self._portfolio_engine.plan(
                        PortfolioDecisionBatch(timestamp, normalized_opportunities),
                        portfolio_context,
                    )
                    portfolio_opportunities.extend(plan_result.ranked_opportunities)
                    portfolio_decisions.extend(plan_result.decisions)
                    portfolio_plans.append(plan_result.plan)
                    portfolio_sleeves.extend(plan_result.sleeve_state)
                    strategy_by_name = {item.name: item for item in strategies}
                    for decision in plan_result.decisions:
                        opportunity = next(
                            item
                            for item in plan_result.ranked_opportunities
                            if item.opportunity_id == decision.opportunity_id
                        )
                        strategy_by_name[opportunity.strategy_name].on_portfolio_decision(
                            decision
                        )

                    for proposal in plan_result.plan.orders_to_create:
                        order_id = f"order-{len(orders) + 1:06d}"
                        observed_price = last_prices.get(proposal.symbol)
                        if observed_price is None:
                            raise BacktestDataError(
                                f"no observed close exists for portfolio sizing of {proposal.symbol}"
                            )
                        order_request = OrderRequest(
                            order_id=order_id,
                            symbol=proposal.symbol,
                            side=proposal.side,
                            quantity=proposal.quantity,
                            strategy_decision_id=proposal.signal_id,
                            ml_decision_id=proposal.ml_decision_id,
                            activation_decision_id=proposal.activation_decision_id,
                            portfolio_plan_id=proposal.portfolio_plan_id,
                            portfolio_decision_id=proposal.portfolio_decision_id,
                            portfolio_opportunity_ids=proposal.opportunity_ids,
                            created_at=timestamp,
                            expected_entry_price=observed_price,
                        )
                        risk_portfolio = self._portfolio_with_pending_buys(
                            ledger.snapshot(timestamp, last_prices),
                            pending_buy_risk,
                        )
                        risk_state = self._risk_engine.current_state(
                            timestamp, risk_portfolio.total_equity
                        )
                        series_history = tuple(
                            item
                            for item in history
                            if item.symbol == proposal.symbol
                            and item.timeframe == proposal.timeframe
                            and item.timestamp <= timestamp
                        )
                        feature_snapshot = None
                        if series_history and series_history[-1].timestamp == timestamp:
                            feature_snapshot = self._feature_engine.compute(
                                series_history,
                                FeatureRequest(),
                                as_of=timestamp,
                            )
                        risk_context = RiskContext(
                            timestamp=timestamp,
                            profile=profile,
                            portfolio=risk_portfolio,
                            order=order_request,
                            expected_entry_price=observed_price,
                            market_prices=tuple(sorted(last_prices.items())),
                            risk_state=risk_state,
                            timeframe=proposal.timeframe,
                            feature_snapshot=feature_snapshot,
                            return_series=return_series,
                            pending_sell_quantities=self._pending_sell_quantities(
                                pending_sell_risk
                            ),
                        )
                        risk_decision = self._risk_engine.evaluate_context(risk_context)
                        self._validate_risk_decision(order_request, risk_decision)
                        risk_decisions.append(risk_decision)
                        common_order = {
                            "order_id": order_id,
                            "symbol": proposal.symbol,
                            "timeframe": proposal.timeframe,
                            "side": proposal.side,
                            "order_type": order_request.order_type,
                            "created_at": timestamp,
                            "signal_id": proposal.signal_id,
                            "ml_decision_id": proposal.ml_decision_id,
                            "activation_decision_id": proposal.activation_decision_id,
                            "portfolio_plan_id": proposal.portfolio_plan_id,
                            "portfolio_decision_id": proposal.portfolio_decision_id,
                            "portfolio_opportunity_ids": proposal.opportunity_ids,
                            "risk_decision_id": risk_decision.decision_id,
                        }
                        if risk_decision.status is RiskDecisionStatus.REJECT:
                            order = BacktestOrder(
                                quantity=proposal.quantity,
                                status=OrderStatus.REJECTED,
                                status_reason=risk_decision.reason,
                                completed_at=timestamp,
                                **common_order,
                            )
                            order_indexes[order.order_id] = len(orders)
                            orders.append(order)
                            continue
                        approved = risk_decision.approved_quantity
                        if approved is None or approved <= Decimal("0"):
                            raise BacktestConfigurationError(
                                "approved portfolio risk decision requires positive quantity"
                            )
                        order = BacktestOrder(quantity=approved, **common_order)
                        order_indexes[order.order_id] = len(orders)
                        orders.append(order)
                        pending_order_ids.append(order.order_id)
                        if order.side is OrderSide.BUY:
                            pending_buy_risk[order.order_id] = (
                                order.symbol,
                                order.quantity,
                                observed_price,
                            )
                        else:
                            pending_sell_risk[order.order_id] = (
                                order.symbol,
                                order.quantity,
                            )

            equity_point = ledger.equity_point(timestamp, last_prices)
            equity_curve.append(equity_point)
            self._risk_engine.observe(timestamp, equity_point.equity)

        if action_index < len(actions):
            warnings.append(
                f"{len(actions) - action_index} corporate action(s) after the final market event were not applied"
            )
        if pending_order_ids:
            warnings.append(
                f"{len(pending_order_ids)} order(s) remained PENDING at dataset end"
            )

        fill_tuple = tuple(fills)
        applied_actions = tuple(
            action for action in actions if action.timestamp <= bars[-1].timestamp
        )
        trades = reconstruct_trades(fill_tuple, applied_actions)
        curve_tuple = tuple(equity_curve)
        risk_decision_tuple = tuple(risk_decisions)
        risk_summary = self._risk_engine.summary(
            risk_decision_tuple, bars[-1].timestamp
        )
        metrics = self._metrics_engine.calculate(
            initial_capital=config.starting_cash,
            curve=curve_tuple,
            fills=fill_tuple,
            trades=trades,
            timeframe=config.primary_timeframe,
            risk_free_rate=config.risk_free_rate,
            dividend_income=ledger.dividend_income,
        )
        benchmark = None
        if config.benchmark_symbol is not None:
            benchmark_bars = tuple(
                bar
                for bar in bars
                if bar.symbol == config.benchmark_symbol
                and bar.timeframe == config.primary_timeframe
            )
            benchmark_actions = tuple(
                action
                for action in actions
                if action.symbol == config.benchmark_symbol
            )
            benchmark = BuyAndHoldBenchmark().run(
                symbol=config.benchmark_symbol,
                bars=benchmark_bars,
                corporate_actions=benchmark_actions,
                initial_capital=config.starting_cash,
                strategy_total_return=metrics.total_return,
            )

        code_version = self._code_version
        if code_version is None:
            code_version = detect_git_commit(PROJECT_ROOT)
        source_hash = source_tree_hash(PROJECT_ROOT)
        references = tuple(
            sorted(
                (dataset.reference for dataset in normalized_datasets),
                key=lambda reference: (
                    reference.symbol,
                    reference.timeframe,
                    reference.dataset_id,
                ),
            )
        )
        if len(strategies) == 1:
            run_strategy_name = primary_strategy.name
            run_strategy_version = primary_strategy.version
            strategy_parameters = tuple(sorted(primary_strategy.parameters))
        else:
            run_strategy_name = "multi-strategy-portfolio"
            run_strategy_version = "1.0"
            strategy_parameters = tuple(
                sorted(
                    (
                        f"strategy.{item.name}.{name}",
                        value,
                    )
                    for item in strategies
                    for name, value in item.parameters
                )
            )
        strategy_signals = tuple(
            sorted(
                (signal for item in strategies for signal in item.signals),
                key=lambda signal: (
                    signal.timestamp,
                    signal.strategy_name,
                    signal.symbol,
                    signal.signal_id,
                ),
            )
        )
        if any(not isinstance(signal, StrategySignal) for signal in strategy_signals):
            raise BacktestConfigurationError(
                "strategy.signals must contain only StrategySignal values"
            )
        if any(
            signal.strategy_name not in {item.name for item in strategies}
            or signal.strategy_version
            != next(
                item.version
                for item in strategies
                if item.name == signal.strategy_name
            )
            or not bars[0].timestamp <= signal.timestamp <= bars[-1].timestamp
            for signal in strategy_signals
        ):
            raise BacktestConfigurationError(
                "strategy signal identity or timestamp does not match the run"
            )
        regime_snapshot_tuple = tuple(regime_snapshots)
        activation_decision_tuple = tuple(
            sorted(
                activation_decisions,
                key=lambda item: (
                    item.timestamp,
                    item.strategy_name,
                    item.symbol,
                    item.signal_id,
                    item.decision_id,
                ),
            )
        )
        regime_transition_tuple = (
            self._regime_detector.transitions
            if self._regime_detector is not None
            else ()
        )
        regime_report = (
            build_regime_report(
                regime_snapshot_tuple,
                regime_transition_tuple,
                activation_decision_tuple,
            )
            if self._regime_detector is not None
            else None
        )
        regime_detector_name = (
            self._regime_detector.detector_name
            if self._regime_detector is not None
            else "unavailable"
        )
        regime_detector_version = (
            self._regime_detector.detector_version
            if self._regime_detector is not None
            else "0"
        )
        regime_config = (
            self._regime_detector.config_parameters
            if self._regime_detector is not None
            else ()
        )
        regime_config_hash = (
            self._regime_detector.config_hash
            if self._regime_detector is not None
            else "0" * 64
        )
        strategy_policy_name = (
            self._activation_policy.policy_name
            if self._activation_policy is not None
            else "unavailable"
        )
        strategy_policy_version = (
            self._activation_policy.policy_version
            if self._activation_policy is not None
            else "0"
        )
        strategy_policy_config = (
            self._activation_policy.config_parameters
            if self._activation_policy is not None
            else ()
        )
        strategy_policy_config_hash = (
            self._activation_policy.config_hash
            if self._activation_policy is not None
            else "0" * 64
        )
        active_scorers = tuple(
            scorer
            for item in strategies
            if (scorer := self._scorer_for(item.name)) is not None
            and scorer.mode is not MLMode.DISABLED
        )
        ml_mode = active_scorers[0].mode if active_scorers else MLMode.DISABLED
        ml_artifacts = tuple(
            scorer.artifact for scorer in active_scorers if scorer.artifact is not None
        )
        ml_artifact = ml_artifacts[0] if len(ml_artifacts) == 1 else None
        if len(ml_artifacts) > 1:
            ml_model_id = "ml-multi-" + stable_hash(
                tuple((item.model_id, item.artifact_checksum) for item in ml_artifacts)
            )[:24]
            ml_model_family = "MULTI"
            ml_model_version = "1.0"
            ml_model_status = (
                ml_artifacts[0].status.value
                if len({item.status for item in ml_artifacts}) == 1
                else "MIXED"
            )
            ml_model_artifact_hash = stable_hash(
                tuple((item.model_id, item.artifact_checksum) for item in ml_artifacts)
            )
            ml_base_feature_schema_version = (
                ml_artifacts[0].feature_schema_version
                if len({item.feature_schema_version for item in ml_artifacts}) == 1
                else "mixed"
            )
            ml_feature_schema_version = (
                ml_artifacts[0].ml_feature_schema_version
                if len({item.ml_feature_schema_version for item in ml_artifacts}) == 1
                else "mixed"
            )
            ml_feature_names = tuple(
                sorted({name for item in ml_artifacts for name in item.feature_names})
            )
            ml_label_config = tuple(
                sorted(
                    (f"{item.model_id}.{name}", value)
                    for item in ml_artifacts
                    for name, value in item.label_config
                )
            )
            ml_split_config = tuple(
                sorted(
                    (f"{item.model_id}.{name}", value)
                    for item in ml_artifacts
                    for name, value in item.split_config
                )
            )
            ml_model_config = tuple(
                sorted(
                    (f"{item.model_id}.{name}", value)
                    for item in ml_artifacts
                    for name, value in item.model_config
                )
            )
            ml_training_period = None
            ml_validation_period = None
            ml_test_period = None
        else:
            ml_model_id = ml_artifact.model_id if ml_artifact else None
            ml_model_family = ml_artifact.model_family.value if ml_artifact else None
            ml_model_version = ml_artifact.model_version if ml_artifact else None
            ml_model_status = ml_artifact.status.value if ml_artifact else None
            ml_model_artifact_hash = (
                ml_artifact.artifact_checksum if ml_artifact else None
            )
            ml_base_feature_schema_version = (
                ml_artifact.feature_schema_version if ml_artifact else None
            )
            ml_feature_schema_version = (
                ml_artifact.ml_feature_schema_version if ml_artifact else None
            )
            ml_feature_names = ml_artifact.feature_names if ml_artifact else ()
            ml_label_config = ml_artifact.label_config if ml_artifact else ()
            ml_split_config = ml_artifact.split_config if ml_artifact else ()
            ml_model_config = ml_artifact.model_config if ml_artifact else ()
            ml_training_period = ml_artifact.training_period if ml_artifact else None
            ml_validation_period = ml_artifact.validation_period if ml_artifact else None
            ml_test_period = ml_artifact.test_period if ml_artifact else None
        ml_threshold = None
        if ml_mode is MLMode.FILTER:
            thresholds = {scorer.threshold for scorer in active_scorers}
            ml_threshold = next(iter(thresholds)) if len(thresholds) == 1 else None
        ml_prediction_tuple = tuple(
            sorted(
                ml_predictions,
                key=lambda item: (
                    item.timestamp,
                    item.strategy_name,
                    item.symbol,
                    item.prediction_id,
                ),
            )
        )
        ml_decision_tuple = tuple(
            sorted(
                ml_decisions,
                key=lambda item: (item.timestamp, item.symbol, item.signal_id, item.decision_id),
            )
        )
        portfolio_opportunity_tuple = tuple(portfolio_opportunities)
        portfolio_decision_tuple = tuple(portfolio_decisions)
        portfolio_plan_tuple = tuple(portfolio_plans)
        portfolio_target_tuple = tuple(
            target for plan in portfolio_plan_tuple for target in plan.targets
        )
        portfolio_sleeve_tuple = tuple(portfolio_sleeves)
        portfolio_metrics = (
            build_portfolio_metrics(
                portfolio_plan_tuple,
                portfolio_decision_tuple,
                portfolio_sleeve_tuple,
                fill_tuple,
                curve_tuple,
                self._portfolio_engine.config,
            )
            if self._portfolio_engine is not None
            else None
        )
        run_identity = {
            "strategy_name": run_strategy_name,
            "strategy_version": run_strategy_version,
            "strategy_parameters": strategy_parameters,
            "datasets": references,
            "config": config,
            "context": context,
            "code_version": code_version,
            "source_hash_sha256": source_hash,
            "risk_engine_name": self._risk_engine.engine_name,
            "risk_engine_version": self._risk_engine.engine_version,
            "risk_config": self._risk_engine.config_parameters,
            "risk_config_hash": self._risk_engine.config_hash,
            "regime_detector_name": regime_detector_name,
            "regime_detector_version": regime_detector_version,
            "regime_config": regime_config,
            "regime_config_hash": regime_config_hash,
            "strategy_policy_name": strategy_policy_name,
            "strategy_policy_version": strategy_policy_version,
            "strategy_policy_config": strategy_policy_config,
            "strategy_policy_config_hash": strategy_policy_config_hash,
            "ml_mode": ml_mode.value,
            "ml_model_id": ml_model_id,
            "ml_model_artifact_hash": ml_model_artifact_hash,
            "ml_model_config": ml_model_config,
            "ml_split_config": ml_split_config,
            "ml_threshold": ml_threshold,
            "ml_models": tuple(
                (item.model_id, item.artifact_checksum) for item in ml_artifacts
            ),
            "portfolio_engine_name": (
                self._portfolio_engine.engine_name
                if self._portfolio_engine is not None
                else "unavailable"
            ),
            "portfolio_engine_version": (
                self._portfolio_engine.engine_version
                if self._portfolio_engine is not None
                else "0"
            ),
            "portfolio_config": (
                self._portfolio_engine.config_parameters
                if self._portfolio_engine is not None
                else ()
            ),
            "portfolio_config_hash": (
                self._portfolio_engine.config_hash
                if self._portfolio_engine is not None
                else "0" * 64
            ),
            "portfolio_opportunities": portfolio_opportunity_tuple,
            "portfolio_decisions": portfolio_decision_tuple,
            "portfolio_plans": portfolio_plan_tuple,
            "portfolio_targets": portfolio_target_tuple,
            "portfolio_sleeves": portfolio_sleeve_tuple,
        }
        run_id = f"bt-{stable_hash(run_identity)[:24]}"
        created_at = datetime.now(timezone.utc)
        result_values: dict[str, Any] = {
            "run_id": run_id,
            "status": BacktestStatus.COMPLETED.value,
            "started_at": bars[0].timestamp,
            "completed_at": bars[-1].timestamp,
            "created_at": created_at,
            "strategy_name": run_strategy_name,
            "strategy_version": run_strategy_version,
            "strategy_parameters": strategy_parameters,
            "dataset_references": references,
            "config": config,
            "initial_cash": config.starting_cash,
            "final_equity": curve_tuple[-1].equity,
            "metrics": metrics,
            "equity_curve": curve_tuple,
            "orders": tuple(orders),
            "fills": fill_tuple,
            "trades": trades,
            "signals": strategy_signals,
            "ledger_entries": ledger.entries,
            "warnings": tuple(dict.fromkeys(warnings)),
            "benchmark": benchmark,
            "code_version": code_version,
            "source_hash_sha256": source_hash,
            "risk_engine_name": self._risk_engine.engine_name,
            "risk_engine_version": self._risk_engine.engine_version,
            "risk_config": self._risk_engine.config_parameters,
            "risk_config_hash": self._risk_engine.config_hash,
            "risk_decisions": risk_decision_tuple,
            "risk_state_transitions": self._risk_engine.state_transitions,
            "risk_summary": risk_summary,
            "regime_detector_name": regime_detector_name,
            "regime_detector_version": regime_detector_version,
            "regime_config": regime_config,
            "regime_config_hash": regime_config_hash,
            "strategy_policy_name": strategy_policy_name,
            "strategy_policy_version": strategy_policy_version,
            "strategy_policy_config": strategy_policy_config,
            "strategy_policy_config_hash": strategy_policy_config_hash,
            "regime_snapshots": regime_snapshot_tuple,
            "regime_transitions": regime_transition_tuple,
            "activation_decisions": activation_decision_tuple,
            "regime_report": regime_report,
            "ml_mode": ml_mode.value,
            "ml_model_id": ml_model_id,
            "ml_model_family": ml_model_family,
            "ml_model_version": ml_model_version,
            "ml_model_status": ml_model_status,
            "ml_model_artifact_hash": ml_model_artifact_hash,
            "ml_base_feature_schema_version": ml_base_feature_schema_version,
            "ml_feature_schema_version": ml_feature_schema_version,
            "ml_feature_names": ml_feature_names,
            "ml_label_config": ml_label_config,
            "ml_split_config": ml_split_config,
            "ml_model_config": ml_model_config,
            "ml_threshold": ml_threshold,
            "ml_training_period": ml_training_period,
            "ml_validation_period": ml_validation_period,
            "ml_test_period": ml_test_period,
            "ml_predictions": ml_prediction_tuple,
            "ml_decisions": ml_decision_tuple,
            "portfolio_engine_name": (
                self._portfolio_engine.engine_name
                if self._portfolio_engine is not None
                else "unavailable"
            ),
            "portfolio_engine_version": (
                self._portfolio_engine.engine_version
                if self._portfolio_engine is not None
                else "0"
            ),
            "portfolio_config": (
                self._portfolio_engine.config_parameters
                if self._portfolio_engine is not None
                else ()
            ),
            "portfolio_config_hash": (
                self._portfolio_engine.config_hash
                if self._portfolio_engine is not None
                else "0" * 64
            ),
            "portfolio_opportunities": portfolio_opportunity_tuple,
            "portfolio_decisions": portfolio_decision_tuple,
            "portfolio_plans": portfolio_plan_tuple,
            "portfolio_targets": portfolio_target_tuple,
            "portfolio_sleeves": portfolio_sleeve_tuple,
            "portfolio_metrics": portfolio_metrics,
        }
        result_values["result_hash"] = stable_result_hash(result_values)
        return BacktestResult(**result_values)

    @staticmethod
    def _replace_order(
        order: BacktestOrder,
        orders: list[BacktestOrder],
        order_indexes: dict[str, int],
    ) -> None:
        orders[order_indexes[order.order_id]] = order

    def _scorer_for(self, strategy_name: str) -> MLScorer | None:
        if self._ml_scorers:
            return self._ml_scorers.get(strategy_name)
        return self._ml_scorer

    @staticmethod
    def _strategy_portfolio_snapshot(
        portfolio: PortfolioSnapshot,
        strategy_name: str,
        sleeve_state: tuple[StrategySleeveState, ...],
    ) -> PortfolioSnapshot:
        """Expose only this strategy's logical holdings, never future state.

        Physical cash/equity remain shared. A same-symbol position can therefore
        be attributed to several strategy sleeves without duplicating it in the
        real ledger.
        """

        active_symbols = {
            item.symbol
            for item in sleeve_state
            if item.strategy_name == strategy_name
            and item.target_weight_contribution > Decimal("0")
        }
        return PortfolioSnapshot(
            as_of=portfolio.as_of,
            cash=portfolio.cash,
            total_equity=portfolio.total_equity,
            positions=tuple(
                item for item in portfolio.positions if item.symbol in active_symbols
            ),
        )

    @staticmethod
    def _validate_risk_decision(
        order: OrderRequest, decision: RiskDecision
    ) -> None:
        if not isinstance(decision, RiskDecision):
            raise BacktestConfigurationError("RiskEngine must return RiskDecision")
        if decision.order_id != order.order_id:
            raise BacktestConfigurationError(
                "RiskEngine returned a decision for another order"
            )
        if decision.requested_quantity != order.quantity:
            raise BacktestConfigurationError(
                "RiskDecision must record the exact requested quantity"
            )
        if (
            decision.approved_quantity is None
            or decision.approved_quantity > order.quantity
        ):
            raise BacktestConfigurationError(
                "RiskEngine may never increase or omit approved quantity"
            )

    @staticmethod
    def _signal_for_intent(
        strategy: BacktestStrategy, intent: OrderIntent
    ) -> StrategySignal:
        if intent.signal_id is None:
            raise BacktestConfigurationError(
                "policy-filtered strategies must link every intent to a signal"
            )
        matches = tuple(
            signal for signal in strategy.signals if signal.signal_id == intent.signal_id
        )
        if len(matches) != 1:
            raise BacktestConfigurationError(
                "policy-filtered intent must reference exactly one emitted signal"
            )
        return matches[0]

    @staticmethod
    def _validate_activation_decision(
        intent: OrderIntent,
        signal: StrategySignal,
        decision: ActivationDecision,
    ) -> None:
        if not isinstance(decision, ActivationDecision):
            raise BacktestConfigurationError(
                "ActivationPolicy must return ActivationDecision"
            )
        if decision.signal_id != signal.signal_id:
            raise BacktestConfigurationError(
                "ActivationDecision must reference the exact strategy signal"
            )
        if decision.proposed_quantity != intent.quantity:
            raise BacktestConfigurationError(
                "ActivationDecision must record the exact strategy proposal"
            )
        if (
            decision.allocation_multiplier > Decimal("1")
            or decision.adjusted_quantity > intent.quantity
        ):
            raise BacktestConfigurationError(
                "ActivationPolicy may never increase the strategy proposal"
            )

    @staticmethod
    def _validate_ml_decision(
        signal: StrategySignal,
        prediction: MLPrediction | None,
        decision: MLFilterDecision,
    ) -> None:
        if not isinstance(decision, MLFilterDecision):
            raise BacktestConfigurationError("MLScorer must return MLFilterDecision")
        if decision.signal_id != signal.signal_id:
            raise BacktestConfigurationError(
                "MLFilterDecision must reference the exact strategy signal"
            )
        if prediction is not None:
            if not isinstance(prediction, MLPrediction):
                raise BacktestConfigurationError("MLScorer returned invalid prediction")
            if prediction.prediction_id != decision.prediction_id:
                raise BacktestConfigurationError(
                    "ML decision and prediction lineage do not match"
                )
            if prediction.timestamp != signal.timestamp or prediction.symbol != signal.symbol:
                raise BacktestConfigurationError(
                    "ML prediction must describe the exact signal event"
                )
        elif decision.prediction_id is not None:
            raise BacktestConfigurationError(
                "ML decision references a missing prediction"
            )
        if (
            signal.action is StrategySignalAction.EXIT_LONG
            and decision.status is MLFilterStatus.BLOCK
        ):
            raise BacktestConfigurationError("ML must never block EXIT_LONG")

    @staticmethod
    def _portfolio_with_pending_buys(
        portfolio: PortfolioSnapshot,
        reservations: dict[str, tuple[str, Decimal, Decimal]],
    ) -> PortfolioSnapshot:
        quantities: dict[str, tuple[Decimal, Decimal]] = {
            item.symbol: (item.quantity, item.average_price)
            for item in portfolio.positions
        }
        reserved_cost = Decimal("0")
        for symbol, quantity, price in reservations.values():
            held, average = quantities.get(symbol, (Decimal("0"), price))
            combined = held + quantity
            combined_average = (
                (held * average + quantity * price) / combined
                if combined > Decimal("0")
                else price
            )
            quantities[symbol] = (combined, combined_average)
            reserved_cost += quantity * price
        return PortfolioSnapshot(
            as_of=portfolio.as_of,
            cash=max(Decimal("0"), portfolio.cash - reserved_cost),
            total_equity=portfolio.total_equity,
            positions=tuple(
                Position(symbol, quantity, average)
                for symbol, (quantity, average) in sorted(quantities.items())
            ),
        )

    @staticmethod
    def _pending_sell_quantities(
        reservations: dict[str, tuple[str, Decimal]],
    ) -> tuple[tuple[str, Decimal], ...]:
        by_symbol: dict[str, Decimal] = {}
        for symbol, quantity in reservations.values():
            by_symbol[symbol] = by_symbol.get(symbol, Decimal("0")) + quantity
        return tuple(sorted(by_symbol.items()))

    @staticmethod
    def _validate_datasets(
        datasets: tuple[BacktestDataset, ...],
        asset_universe: tuple[str, ...],
        profile_timeframes: tuple[str, ...],
        config: BacktestConfig,
    ) -> list[str]:
        warnings: list[str] = []
        for dataset in datasets:
            reference = dataset.reference
            if reference.symbol not in asset_universe:
                raise BacktestDataError(
                    f"dataset symbol {reference.symbol!r} is outside the active profile universe"
                )
            if reference.timeframe not in profile_timeframes:
                raise BacktestDataError(
                    f"dataset timeframe {reference.timeframe!r} is outside the active profile"
                )
            if reference.data_kind not in {
                DataKind.RAW_WITH_ADJUSTED_CLOSE.value,
                DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE.value,
            }:
                raise BacktestDataError(
                    "backtests require raw OHLC with explicit corporate actions; "
                    f"unsupported data_kind={reference.data_kind!r}"
                )
            report = dataset.quality_report
            if report.quality_status is QualityStatus.FAIL:
                raise BacktestDataError(
                    f"DataQuality FAIL for {reference.symbol} {reference.timeframe}"
                )
            if report.quality_status is QualityStatus.WARNING:
                message = (
                    f"DataQuality WARNING for {reference.symbol} {reference.timeframe}: "
                    f"missing_expected_bars={report.missing_expected_bar_count}, "
                    f"unexpected_gaps={report.unexpected_gap_count}"
                )
                if config.data_quality_policy is DataQualityPolicy.STRICT:
                    raise BacktestDataError(message)
                warnings.append(message)
                warnings.extend(report.warnings)
        return warnings

    @staticmethod
    def _merge_bars(datasets: tuple[BacktestDataset, ...]) -> tuple[MarketBar, ...]:
        bars = tuple(
            sorted(
                (bar for dataset in datasets for bar in dataset.bars),
                key=_event_key,
            )
        )
        keys = [_event_key(bar) for bar in bars]
        if len(keys) != len(set(keys)):
            raise BacktestDataError(
                "duplicate symbol/timeframe/timestamp bars across datasets"
            )
        return bars

    @staticmethod
    def _merge_actions(
        datasets: tuple[BacktestDataset, ...],
    ) -> tuple[CorporateAction, ...]:
        by_key = {
            _action_key(action): action
            for dataset in datasets
            for action in dataset.corporate_actions
        }
        return tuple(by_key[key] for key in sorted(by_key))
