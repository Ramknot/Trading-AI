"""Explainable, long-only Lot 3 quantitative research baselines."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from trading_ai.backtesting.models import (
    OrderIntent,
    StrategyContext,
    StrategySignal,
    StrategySignalAction,
)
from trading_ai.backtesting.strategy import BacktestStrategy
from trading_ai.core.models import OrderSide, OrderType, Position
from trading_ai.features import (
    FEATURE_ENGINE_VERSION,
    FEATURE_SCHEMA_VERSION,
    FeatureEngine,
    FeatureRequest,
    FeatureSnapshot,
)
from trading_ai.strategies.config import (
    BreakoutConfig,
    MeanReversionConfig,
    MomentumConfig,
    TrendConfig,
)
from trading_ai.strategies.sizing import BaselineSizer
from trading_ai.regimes.models import StructureRegime
from trading_ai.regimes.models import ActivationDecision, ActivationStatus


def _symbols(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if not normalized or any(not value or not value.strip() for value in normalized):
        raise ValueError("symbols must contain non-empty configured values")
    return normalized


def _position(context: StrategyContext, symbol: str) -> Position | None:
    return next(
        (position for position in context.portfolio.positions if position.symbol == symbol),
        None,
    )


def _feature_metadata(
    snapshot: FeatureSnapshot, names: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    values = []
    for name in names:
        value = snapshot.get(name)
        if value is None:
            raise ValueError(f"feature {name} is unavailable")
        values.append((name, format(value, ".15g")))
    return tuple(sorted(values))


class _FeatureBaseline(BacktestStrategy):
    """Shared signal lineage without shared trading rules."""

    def __init__(self, feature_engine: FeatureEngine | None = None) -> None:
        self._feature_engine = feature_engine or FeatureEngine()
        self._signals: list[StrategySignal] = []
        self._signal_sequence = 0

    @property
    def signals(self) -> tuple[StrategySignal, ...]:
        return tuple(self._signals)

    def reset(self) -> None:
        self._signals.clear()
        self._signal_sequence = 0
        self._feature_engine.clear_cache()
        self._reset_state()

    def _reset_state(self) -> None:
        pass

    def on_activation_decision(self, decision: ActivationDecision) -> None:
        """Release a proposal marker when regime policy blocks the entry.

        This feedback changes no signal, multiplier, risk decision, or order. It
        only permits a fresh evaluation on a later bar if the regime changes.
        """

        if decision.status is not ActivationStatus.BLOCK:
            return
        signal = next(
            (item for item in self._signals if item.signal_id == decision.signal_id),
            None,
        )
        if signal is None:
            return
        if signal.action is StrategySignalAction.ENTER_LONG:
            self._entry_submitted.discard(signal.symbol)
        elif signal.action is StrategySignalAction.EXIT_LONG:
            self._exit_submitted.discard(signal.symbol)

    def on_ml_decision(self, decision: object) -> None:
        """Release only entry proposals that the ML gate stopped."""

        if getattr(getattr(decision, "status", None), "value", None) not in {
            "BLOCK",
            "UNAVAILABLE",
        }:
            return
        signal = next(
            (
                item
                for item in self._signals
                if item.signal_id == getattr(decision, "signal_id", None)
            ),
            None,
        )
        if signal is not None and signal.action is StrategySignalAction.ENTER_LONG:
            self._entry_submitted.discard(signal.symbol)

    def _emit_signal(
        self,
        context: StrategyContext,
        *,
        symbol: str,
        action: StrategySignalAction,
        reason: str,
        features_used: tuple[tuple[str, str], ...],
        strength: float = 1.0,
    ) -> StrategySignal:
        self._signal_sequence += 1
        signal = StrategySignal(
            signal_id=f"signal-{self.name}-{self._signal_sequence:06d}",
            strategy_name=self.name,
            strategy_version=self.version,
            symbol=symbol,
            timeframe=context.current_bar.timeframe,
            timestamp=context.current_time,
            action=action,
            strength=strength,
            reason=reason,
            features_used=features_used,
        )
        self._signals.append(signal)
        return signal

    @staticmethod
    def _profile_capped_sizer(
        context: StrategyContext, configured: BaselineSizer
    ) -> BaselineSizer:
        profile_limit = Decimal(str(context.profile.max_exposure))
        return BaselineSizer(
            allocation_fraction=min(
                configured.allocation_fraction,
                profile_limit,
            ),
            quantity_step=configured.quantity_step,
        )

    @staticmethod
    def _lineage_parameters(
        request: FeatureRequest | None = None,
    ) -> tuple[tuple[str, str], ...]:
        parameters = [
            ("feature_engine_version", FEATURE_ENGINE_VERSION),
            ("feature_schema_version", FEATURE_SCHEMA_VERSION),
        ]
        if request is not None:
            parameters.extend(
                (f"feature.{name}", value)
                for name, value in request.to_parameters()
            )
        return tuple(sorted(parameters))


class TrendFollowingStrategy(_FeatureBaseline):
    """Long when fast EMA exceeds slow EMA with positive fast slope."""

    version = "1.0"

    def __init__(
        self,
        symbols: Sequence[str],
        timeframe: str,
        config: TrendConfig | None = None,
        *,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        super().__init__(feature_engine)
        self.symbols = _symbols(symbols)
        if not timeframe.strip():
            raise ValueError("timeframe must not be empty")
        self.timeframe = timeframe
        self.config = config or TrendConfig()
        self._request = FeatureRequest(
            sma_windows=(self.config.fast_window, self.config.slow_window),
            ema_windows=(self.config.fast_window, self.config.slow_window),
            return_windows=(1,),
            structure_windows=(self.config.slow_window,),
            slope_lookback=self.config.slope_lookback,
        )
        self._sizer = BaselineSizer(self.config.allocation_fraction)
        self._entry_submitted: set[str] = set()
        self._exit_submitted: set[str] = set()

    @property
    def name(self) -> str:
        return "trend"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    *self.config.to_parameters(),
                    *self._lineage_parameters(self._request),
                    ("symbols", ",".join(self.symbols)),
                    ("sizing_policy", "total_fraction_capped_by_profile_exposure"),
                    ("timeframe", self.timeframe),
                )
            )
        )

    def _reset_state(self) -> None:
        self._entry_submitted.clear()
        self._exit_submitted.clear()

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        bar = context.current_bar
        if bar.symbol not in self.symbols or bar.timeframe != self.timeframe:
            return ()
        position = _position(context, bar.symbol)
        if position is not None and position.quantity > Decimal("0"):
            self._entry_submitted.discard(bar.symbol)
        else:
            self._exit_submitted.discard(bar.symbol)

        snapshot = self._feature_engine.compute(
            context.history_for(bar.symbol, self.timeframe), self._request
        )
        fast_name = f"ema_{self.config.fast_window}"
        slow_name = f"ema_{self.config.slow_window}"
        slope_name = (
            f"ema_{self.config.fast_window}_slope_{self.config.slope_lookback}"
        )
        if not snapshot.is_available(fast_name, slow_name, slope_name):
            return ()
        fast = snapshot.get(fast_name)
        slow = snapshot.get(slow_name)
        slope = snapshot.get(slope_name)
        assert fast is not None and slow is not None and slope is not None
        feature_values = _feature_metadata(
            snapshot, (fast_name, slow_name, slope_name)
        )
        feature_values = tuple(
            sorted((*feature_values, ("close", str(bar.close))))
        )
        held = position is not None and position.quantity > Decimal("0")
        bullish = fast > slow and float(bar.close) > slow and slope > 0.0

        if bullish and not held and bar.symbol not in self._entry_submitted:
            quantity = self._profile_capped_sizer(
                context, self._sizer
            ).entry_quantity(
                context.portfolio,
                bar.close,
                slots=len(self.symbols),
            )
            if quantity is None:
                return ()
            signal = self._emit_signal(
                context,
                symbol=bar.symbol,
                action=StrategySignalAction.ENTER_LONG,
                reason=(
                    f"EMA{self.config.fast_window} > EMA{self.config.slow_window}; "
                    f"close > EMA{self.config.slow_window}; fast EMA slope > 0"
                ),
                features_used=feature_values,
            )
            self._entry_submitted.add(bar.symbol)
            return (
                OrderIntent(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                ),
            )
        if (
            held
            and fast <= slow
            and bar.symbol not in self._exit_submitted
        ):
            signal = self._emit_signal(
                context,
                symbol=bar.symbol,
                action=StrategySignalAction.EXIT_LONG,
                reason=f"EMA{self.config.fast_window} <= EMA{self.config.slow_window}",
                features_used=feature_values,
            )
            self._exit_submitted.add(bar.symbol)
            return (
                OrderIntent(
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                ),
            )
        return ()


class BreakoutStrategy(_FeatureBaseline):
    """Enter above a prior range and exit below a shorter prior range."""

    version = "1.0"

    def __init__(
        self,
        symbols: Sequence[str],
        timeframe: str,
        config: BreakoutConfig | None = None,
        *,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        super().__init__(feature_engine)
        self.symbols = _symbols(symbols)
        if not timeframe.strip():
            raise ValueError("timeframe must not be empty")
        self.timeframe = timeframe
        self.config = config or BreakoutConfig()
        self._request = FeatureRequest(
            sma_windows=(self.config.entry_window,),
            ema_windows=(self.config.entry_window,),
            return_windows=(1,),
            structure_windows=(self.config.entry_window, self.config.exit_window),
            slope_lookback=1,
        )
        self._sizer = BaselineSizer(self.config.allocation_fraction)
        self._entry_submitted: set[str] = set()
        self._exit_submitted: set[str] = set()

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    *self.config.to_parameters(),
                    *self._lineage_parameters(self._request),
                    ("symbols", ",".join(self.symbols)),
                    ("sizing_policy", "total_fraction_capped_by_profile_exposure"),
                    ("timeframe", self.timeframe),
                )
            )
        )

    def _reset_state(self) -> None:
        self._entry_submitted.clear()
        self._exit_submitted.clear()

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        bar = context.current_bar
        if bar.symbol not in self.symbols or bar.timeframe != self.timeframe:
            return ()
        position = _position(context, bar.symbol)
        held = position is not None and position.quantity > Decimal("0")
        if held:
            self._entry_submitted.discard(bar.symbol)
        else:
            self._exit_submitted.discard(bar.symbol)
        snapshot = self._feature_engine.compute(
            context.history_for(bar.symbol, self.timeframe), self._request
        )
        entry_name = f"previous_high_{self.config.entry_window}"
        exit_name = f"previous_low_{self.config.exit_window}"
        if not snapshot.is_available(entry_name, exit_name):
            return ()
        previous_high = snapshot.get(entry_name)
        previous_low = snapshot.get(exit_name)
        assert previous_high is not None and previous_low is not None
        feature_values = _feature_metadata(snapshot, (entry_name, exit_name))
        feature_values = tuple(
            sorted((*feature_values, ("close", str(bar.close))))
        )

        if (
            not held
            and float(bar.close) > previous_high
            and bar.symbol not in self._entry_submitted
        ):
            quantity = self._profile_capped_sizer(
                context, self._sizer
            ).entry_quantity(
                context.portfolio,
                bar.close,
                slots=len(self.symbols),
            )
            if quantity is None:
                return ()
            signal = self._emit_signal(
                context,
                symbol=bar.symbol,
                action=StrategySignalAction.ENTER_LONG,
                reason=(
                    f"close > previous {self.config.entry_window}-bar high "
                    "(current bar excluded)"
                ),
                features_used=feature_values,
            )
            self._entry_submitted.add(bar.symbol)
            return (
                OrderIntent(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                ),
            )
        if (
            held
            and float(bar.close) < previous_low
            and bar.symbol not in self._exit_submitted
        ):
            signal = self._emit_signal(
                context,
                symbol=bar.symbol,
                action=StrategySignalAction.EXIT_LONG,
                reason=(
                    f"close < previous {self.config.exit_window}-bar low "
                    "(current bar excluded)"
                ),
                features_used=feature_values,
            )
            self._exit_submitted.add(bar.symbol)
            return (
                OrderIntent(
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                ),
            )
        return ()


class MomentumStrategy(_FeatureBaseline):
    """Select positive top-K returns on exact common market snapshots."""

    version = "1.0"

    def __init__(
        self,
        symbols: Sequence[str],
        timeframe: str,
        config: MomentumConfig | None = None,
        *,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        super().__init__(feature_engine)
        self.symbols = _symbols(symbols)
        if not timeframe.strip():
            raise ValueError("timeframe must not be empty")
        self.timeframe = timeframe
        self.config = config or MomentumConfig()
        if self.config.top_k > len(self.symbols):
            raise ValueError("top_k cannot exceed the configured symbol count")
        self._sizer = BaselineSizer(self.config.allocation_fraction)
        self._last_coherent_time = None
        self._coherent_count = 0
        self._entry_submitted: set[str] = set()
        self._exit_submitted: set[str] = set()

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    *self.config.to_parameters(),
                    *self._lineage_parameters(),
                    ("symbols", ",".join(self.symbols)),
                    ("timeframe", self.timeframe),
                    ("ranking_policy", "exact_timestamp_no_forward_fill"),
                    ("sizing_policy", "total_fraction_capped_by_profile_exposure"),
                )
            )
        )

    def _reset_state(self) -> None:
        self._last_coherent_time = None
        self._coherent_count = 0
        self._entry_submitted.clear()
        self._exit_submitted.clear()

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        bar = context.current_bar
        if bar.symbol not in self.symbols or bar.timeframe != self.timeframe:
            return ()
        ranking = self._feature_engine.relative_strength(
            context.history,
            symbols=self.symbols,
            timeframe=self.timeframe,
            as_of=context.current_time,
            lookback=self.config.lookback,
        )
        if ranking.missing_symbols or self._last_coherent_time == context.current_time:
            return ()
        self._last_coherent_time = context.current_time
        self._coherent_count += 1
        if (self._coherent_count - 1) % self.config.rebalance_every != 0:
            return ()

        by_symbol = {item.symbol: item for item in ranking.values}
        target_limit = min(self.config.top_k, context.profile.max_positions)
        targets = tuple(
            symbol
            for symbol in ranking.ranked_symbols
            if Decimal(str(by_symbol[symbol].rolling_return))
            > self.config.minimum_return
        )[:target_limit]
        target_set = set(targets)
        positions = {
            position.symbol: position
            for position in context.portfolio.positions
            if position.quantity > Decimal("0")
        }
        for symbol in tuple(self._entry_submitted):
            if symbol in positions:
                self._entry_submitted.discard(symbol)
        for symbol in tuple(self._exit_submitted):
            if symbol not in positions:
                self._exit_submitted.discard(symbol)

        intents: list[OrderIntent] = []
        for symbol in sorted(set(positions).intersection(self.symbols) - target_set):
            if symbol in self._exit_submitted:
                continue
            observation = by_symbol[symbol]
            features = tuple(
                sorted(
                    (
                        (f"return_{self.config.lookback}", format(observation.rolling_return, ".15g")),
                        ("relative_strength_percentile", format(observation.percentile, ".15g")),
                        ("relative_strength_rank", format(observation.rank, ".15g")),
                    )
                )
            )
            signal = self._emit_signal(
                context,
                symbol=symbol,
                action=StrategySignalAction.EXIT_LONG,
                reason=f"asset left positive top-{target_limit} relative-strength set",
                features_used=features,
            )
            self._exit_submitted.add(symbol)
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=positions[symbol].quantity,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                )
            )

        current_bars = {
            item.symbol: item
            for item in context.history
            if item.timeframe == self.timeframe
            and item.timestamp == context.current_time
            and item.symbol in target_set
        }
        available_cash = context.portfolio.cash
        for symbol in targets:
            if symbol in positions or symbol in self._entry_submitted:
                continue
            current_bar = current_bars[symbol]
            quantity = self._profile_capped_sizer(
                context, self._sizer
            ).entry_quantity(
                context.portfolio,
                current_bar.close,
                slots=len(targets),
                available_cash=available_cash,
            )
            if quantity is None:
                continue
            observation = by_symbol[symbol]
            features = tuple(
                sorted(
                    (
                        (f"return_{self.config.lookback}", format(observation.rolling_return, ".15g")),
                        ("relative_strength_percentile", format(observation.percentile, ".15g")),
                        ("relative_strength_rank", format(observation.rank, ".15g")),
                    )
                )
            )
            signal = self._emit_signal(
                context,
                symbol=symbol,
                action=StrategySignalAction.ENTER_LONG,
                reason=f"asset entered positive top-{target_limit} relative-strength set",
                features_used=features,
            )
            self._entry_submitted.add(symbol)
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                )
            )
            available_cash -= quantity * current_bar.close
        return tuple(intents)


class MeanReversionStrategy(_FeatureBaseline):
    """Long-only z-score baseline eligible only in a non-HIGH RANGE regime."""

    version = "1.0"

    def __init__(
        self,
        symbols: Sequence[str],
        timeframe: str,
        config: MeanReversionConfig | None = None,
        *,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        super().__init__(feature_engine)
        self.symbols = _symbols(symbols)
        if not timeframe.strip():
            raise ValueError("timeframe must not be empty")
        self.timeframe = timeframe
        self.config = config or MeanReversionConfig()
        self._request = FeatureRequest(
            sma_windows=(self.config.lookback,),
            ema_windows=(self.config.lookback,),
            return_windows=(1,),
            structure_windows=(self.config.lookback,),
            efficiency_windows=(self.config.lookback,),
            zscore_windows=(self.config.lookback,),
            slope_lookback=1,
        )
        self._sizer = BaselineSizer(self.config.allocation_fraction)
        self._entry_submitted: set[str] = set()
        self._exit_submitted: set[str] = set()

    @property
    def name(self) -> str:
        return "mean-reversion"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    *self.config.to_parameters(),
                    *self._lineage_parameters(self._request),
                    ("symbols", ",".join(self.symbols)),
                    ("sizing_policy", "total_fraction_capped_by_profile_exposure"),
                    ("timeframe", self.timeframe),
                )
            )
        )

    def _reset_state(self) -> None:
        self._entry_submitted.clear()
        self._exit_submitted.clear()

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        bar = context.current_bar
        if bar.symbol not in self.symbols or bar.timeframe != self.timeframe:
            return ()
        position = _position(context, bar.symbol)
        held = position is not None and position.quantity > Decimal("0")
        if held:
            self._entry_submitted.discard(bar.symbol)
        else:
            self._exit_submitted.discard(bar.symbol)

        snapshot = self._feature_engine.compute(
            context.history_for(bar.symbol, self.timeframe), self._request
        )
        zscore_name = f"price_zscore_{self.config.lookback}"
        zscore = snapshot.get(zscore_name)
        if zscore is None:
            return ()
        regime = context.current_regime
        regime_features = (
            (
                ("regime_structure", regime.structure_regime.value),
                ("regime_volatility", regime.volatility_regime.value),
            )
            if regime is not None
            else (
                ("regime_structure", "unavailable"),
                ("regime_volatility", "unavailable"),
            )
        )
        features_used = tuple(
            sorted(
                (
                    (zscore_name, format(zscore, ".15g")),
                    *regime_features,
                )
            )
        )

        incompatible_structure = (
            regime is not None
            and regime.structure_regime is not StructureRegime.RANGE
        )
        if held and bar.symbol not in self._exit_submitted and (
            zscore >= float(self.config.exit_zscore) or incompatible_structure
        ):
            reason = (
                "Mean Reversion exit: structure left RANGE"
                if incompatible_structure
                else f"price z-score >= {self.config.exit_zscore}"
            )
            signal = self._emit_signal(
                context,
                symbol=bar.symbol,
                action=StrategySignalAction.EXIT_LONG,
                reason=reason,
                features_used=features_used,
            )
            self._exit_submitted.add(bar.symbol)
            return (
                OrderIntent(
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                ),
            )

        if (
            not held
            and zscore <= float(self.config.entry_zscore)
            and bar.symbol not in self._entry_submitted
        ):
            quantity = self._profile_capped_sizer(context, self._sizer).entry_quantity(
                context.portfolio,
                bar.close,
                slots=len(self.symbols),
            )
            if quantity is None:
                return ()
            signal = self._emit_signal(
                context,
                symbol=bar.symbol,
                action=StrategySignalAction.ENTER_LONG,
                reason=f"price z-score <= {self.config.entry_zscore}",
                features_used=features_used,
            )
            self._entry_submitted.add(bar.symbol)
            return (
                OrderIntent(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    timeframe=self.timeframe,
                    signal_id=signal.signal_id,
                ),
            )
        return ()
