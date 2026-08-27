"""Chronological, offline, provider-neutral historical simulation engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from trading_ai.core.models import BacktestResult, MarketBar, TradingContext
from trading_ai.data.models import (
    CorporateAction,
    DataKind,
    Dividend,
    QualityStatus,
    StockSplit,
)


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
        code_version: str | None = None,
    ) -> None:
        self._execution_model_factory = (
            execution_model_factory or BarExecutionModel
        )
        self._metrics_engine = metrics_engine or MetricsEngine()
        self._code_version = code_version

    def run(
        self,
        strategy: BacktestStrategy,
        datasets: Sequence[BacktestDataset],
        context: TradingContext,
        config: BacktestConfig,
    ) -> BacktestResult:
        settings = load_runtime_settings(context.environment, context.profile)
        profile = settings.profile
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
        history: list[MarketBar] = []
        last_prices: dict[str, Decimal] = {}
        equity_curve: list[EquityPoint] = []
        action_index = 0
        strategy.reset()

        for timestamp, grouped in groupby(bars, key=lambda bar: bar.timestamp):
            current_bars = tuple(grouped)
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

                history.append(bar)
                strategy_context = StrategyContext(
                    current_time=bar.timestamp,
                    current_bar=bar,
                    history=tuple(history),
                    portfolio=ledger.snapshot(bar.timestamp, last_prices),
                    trading_context=context,
                    profile=profile,
                )
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
                    order = BacktestOrder(
                        order_id=f"order-{len(orders) + 1:06d}",
                        symbol=intent.symbol,
                        timeframe=timeframe,
                        side=intent.side,
                        quantity=intent.quantity,
                        order_type=intent.order_type,
                        created_at=bar.timestamp,
                        limit_price=intent.limit_price,
                    )
                    order_indexes[order.order_id] = len(orders)
                    orders.append(order)
                    pending_order_ids.append(order.order_id)
            equity_curve.append(ledger.equity_point(timestamp, last_prices))

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
        strategy_parameters = tuple(sorted(strategy.parameters))
        run_identity = {
            "strategy_name": strategy.name,
            "strategy_version": strategy.version,
            "strategy_parameters": strategy_parameters,
            "datasets": references,
            "config": config,
            "context": context,
            "code_version": code_version,
            "source_hash_sha256": source_hash,
        }
        run_id = f"bt-{stable_hash(run_identity)[:24]}"
        created_at = datetime.now(timezone.utc)
        result_values: dict[str, Any] = {
            "run_id": run_id,
            "status": BacktestStatus.COMPLETED.value,
            "started_at": bars[0].timestamp,
            "completed_at": bars[-1].timestamp,
            "created_at": created_at,
            "strategy_name": strategy.name,
            "strategy_version": strategy.version,
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
            "ledger_entries": ledger.entries,
            "warnings": tuple(dict.fromkeys(warnings)),
            "benchmark": benchmark,
            "code_version": code_version,
            "source_hash_sha256": source_hash,
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
