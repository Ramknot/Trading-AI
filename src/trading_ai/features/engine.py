"""Provider-neutral, decision-free Feature Engine."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime

from trading_ai.core.models import MarketBar
from trading_ai.features.exceptions import FeatureInputError
from trading_ai.features.models import (
    FEATURE_ENGINE_VERSION,
    FeatureRequest,
    FeatureSnapshot,
    FeatureValue,
    RelativeStrengthSnapshot,
)
from trading_ai.features.momentum import (
    rate_of_change,
    relative_strength_values,
    rolling_return,
    simple_return,
)
from trading_ai.features.structure import (
    distance_to_level,
    previous_rolling_high,
    previous_rolling_low,
)
from trading_ai.features.trend import (
    exponential_moving_average_series,
    moving_average_slope,
    price_to_average_distance,
    simple_moving_average_series,
)
from trading_ai.features.volatility import (
    average_true_range_series,
    rolling_volatility,
    true_range,
)
from trading_ai.features.volume import relative_volume, rolling_average_volume


class FeatureEngine:
    """Compute stable snapshots from one immutable past-and-present history.

    A bounded exact-input cache avoids duplicate work without changing feature
    semantics. Passing ``as_of`` is safe even when the caller owns later bars:
    every bar after that timestamp is discarded before calculation.
    """

    version = FEATURE_ENGINE_VERSION

    def __init__(self, *, cache_size: int = 128) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self._cache_size = cache_size
        self._cache: OrderedDict[
            tuple[tuple[MarketBar, ...], FeatureRequest, datetime | None],
            FeatureSnapshot,
        ] = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def compute(
        self,
        bars: Sequence[MarketBar],
        request: FeatureRequest | None = None,
        *,
        as_of: datetime | None = None,
    ) -> FeatureSnapshot:
        """Compute features for the latest bar no later than ``as_of``."""

        normalized = self._validated_history(bars, as_of=as_of)
        definition = request or FeatureRequest()
        key = (normalized, definition, as_of)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        closes = [float(bar.close) for bar in normalized]
        highs = [float(bar.high) for bar in normalized]
        lows = [float(bar.low) for bar in normalized]
        volumes = [float(bar.volume) for bar in normalized]
        values: dict[str, float | None] = {}

        for window in definition.sma_windows:
            series = simple_moving_average_series(closes, window)
            average = series[-1]
            values[f"sma_{window}"] = average
            values[f"sma_{window}_slope_{definition.slope_lookback}"] = (
                moving_average_slope(series, definition.slope_lookback)
            )
            values[f"price_to_sma_{window}"] = price_to_average_distance(
                closes[-1], average
            )

        for window in definition.ema_windows:
            series = exponential_moving_average_series(closes, window)
            average = series[-1]
            values[f"ema_{window}"] = average
            values[f"ema_{window}_slope_{definition.slope_lookback}"] = (
                moving_average_slope(series, definition.slope_lookback)
            )
            values[f"price_to_ema_{window}"] = price_to_average_distance(
                closes[-1], average
            )

        values["simple_return"] = simple_return(closes)
        for window in definition.return_windows:
            values[f"return_{window}"] = rolling_return(closes, window)
            values[f"roc_{window}"] = rate_of_change(closes, window)

        previous_close = closes[-2] if len(closes) > 1 else None
        values["true_range"] = true_range(highs[-1], lows[-1], previous_close)
        atr = average_true_range_series(
            highs, lows, closes, definition.atr_window
        )
        values[f"atr_{definition.atr_window}"] = atr[-1]
        values[f"rolling_vol_{definition.volatility_window}"] = (
            rolling_volatility(closes, definition.volatility_window)
        )

        average_volume = rolling_average_volume(volumes, definition.volume_window)
        values[f"average_volume_{definition.volume_window}"] = average_volume
        values[f"relative_volume_{definition.volume_window}"] = relative_volume(
            volumes, definition.volume_window
        )

        for window in definition.structure_windows:
            previous_high = previous_rolling_high(highs, window)
            previous_low = previous_rolling_low(lows, window)
            values[f"previous_high_{window}"] = previous_high
            values[f"previous_low_{window}"] = previous_low
            values[f"distance_to_previous_high_{window}"] = distance_to_level(
                closes[-1], previous_high
            )
            values[f"distance_to_previous_low_{window}"] = distance_to_level(
                closes[-1], previous_low
            )

        latest = normalized[-1]
        snapshot = FeatureSnapshot(
            symbol=latest.symbol,
            timestamp=latest.timestamp,
            timeframe=latest.timeframe,
            values=tuple(
                FeatureValue(name=name, value=value)
                for name, value in sorted(values.items())
            ),
        )
        self._remember(key, snapshot)
        return snapshot

    def relative_strength(
        self,
        history: Sequence[MarketBar],
        *,
        symbols: Sequence[str],
        timeframe: str,
        as_of: datetime,
        lookback: int,
    ) -> RelativeStrengthSnapshot:
        """Rank only assets with an actual bar at exactly ``as_of``.

        Missing assets and insufficient warm-up are reported, never forward-
        filled. The caller can therefore require a coherent complete snapshot.
        """

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise FeatureInputError("as_of must be timezone-aware")
        if lookback < 1:
            raise ValueError("lookback must be positive")
        normalized_symbols = tuple(sorted(set(symbols)))
        if not normalized_symbols or any(not symbol.strip() for symbol in normalized_symbols):
            raise ValueError("symbols must contain non-empty values")
        if not timeframe.strip():
            raise ValueError("timeframe must not be empty")

        grouped: dict[str, list[MarketBar]] = {
            symbol: [] for symbol in normalized_symbols
        }
        for bar in history:
            if (
                bar.symbol in grouped
                and bar.timeframe == timeframe
                and bar.timestamp <= as_of
            ):
                grouped[bar.symbol].append(bar)

        returns_by_symbol: dict[str, float] = {}
        missing: list[str] = []
        for symbol in normalized_symbols:
            bars = tuple(grouped[symbol])
            if not bars:
                missing.append(symbol)
                continue
            validated = self._validated_history(bars)
            if validated[-1].timestamp != as_of:
                missing.append(symbol)
                continue
            value = rolling_return(
                [float(bar.close) for bar in validated], lookback
            )
            if value is None:
                missing.append(symbol)
                continue
            returns_by_symbol[symbol] = value

        return RelativeStrengthSnapshot(
            timestamp=as_of,
            timeframe=timeframe,
            lookback=lookback,
            values=relative_strength_values(returns_by_symbol),
            missing_symbols=tuple(missing),
        )

    @staticmethod
    def _validated_history(
        bars: Sequence[MarketBar], *, as_of: datetime | None = None
    ) -> tuple[MarketBar, ...]:
        materialized = tuple(bars)
        if as_of is not None:
            if as_of.tzinfo is None or as_of.utcoffset() is None:
                raise FeatureInputError("as_of must be timezone-aware")
            materialized = tuple(
                bar for bar in materialized if bar.timestamp <= as_of
            )
        if not materialized:
            raise FeatureInputError("feature history must contain at least one bar")
        first = materialized[0]
        if any(
            bar.symbol != first.symbol or bar.timeframe != first.timeframe
            for bar in materialized
        ):
            raise FeatureInputError(
                "one feature history must contain one symbol and one timeframe"
            )
        timestamps = [bar.timestamp for bar in materialized]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise FeatureInputError(
                "feature history timestamps must be unique and strictly increasing"
            )
        return materialized

    def _remember(
        self,
        key: tuple[tuple[MarketBar, ...], FeatureRequest, datetime | None],
        snapshot: FeatureSnapshot,
    ) -> None:
        if self._cache_size == 0:
            return
        self._cache[key] = snapshot
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
