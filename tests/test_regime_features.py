from __future__ import annotations

from math import isfinite

import pytest

from backtest_support import bar
from trading_ai.features import FEATURE_SCHEMA_VERSION, FeatureEngine, FeatureRequest
from trading_ai.features.structure import efficiency_ratio, price_zscore
from trading_ai.features.volatility import (
    rolling_percentile,
    rolling_volatility_series,
)


def _request() -> FeatureRequest:
    return FeatureRequest(
        sma_windows=(2,),
        ema_windows=(2,),
        return_windows=(1,),
        structure_windows=(2,),
        efficiency_windows=(3,),
        zscore_windows=(3,),
        slope_lookback=1,
        atr_window=2,
        volatility_window=2,
        volatility_percentile_window=3,
        volume_window=2,
    )


def _bars(closes: list[float]):
    return tuple(
        bar(
            index,
            opening=str(value),
            high=str(value + 0.5),
            low=str(value - 0.5),
            close=str(value),
        )
        for index, value in enumerate(closes)
    )


def test_efficiency_ratio_directional_noisy_and_flat() -> None:
    assert efficiency_ratio([1, 2, 3, 4, 5], 4) == pytest.approx(1.0)
    assert efficiency_ratio([1, 2, 1, 2, 1], 4) == pytest.approx(0.0)
    assert efficiency_ratio([3, 3, 3, 3], 3) is None
    assert efficiency_ratio([1, 2, 3], 3) is None


def test_price_zscore_uses_population_std_and_handles_zero_std() -> None:
    assert price_zscore([1, 2, 3], 3) == pytest.approx(1.224744871391589)
    assert price_zscore([2, 2, 2], 3) is None
    assert price_zscore([1, 2], 3) is None


def test_rolling_volatility_percentile_has_explicit_warmup_and_ties() -> None:
    series = rolling_volatility_series([100, 101, 99, 102, 98, 103], 2)
    assert series[:2] == (None, None)
    assert all(value is None or isfinite(value) for value in series)
    assert rolling_percentile(series[:4], 3) is None
    assert rolling_percentile((0.1, 0.2, 0.3), 3) == pytest.approx(5 / 6)
    assert rolling_percentile((0.2, 0.2, 0.2), 3) == pytest.approx(0.5)


def test_feature_engine_emits_lot5_names_with_schema_1_1() -> None:
    snapshot = FeatureEngine().compute(
        _bars([100, 101, 99, 102, 100, 103]), _request()
    )

    assert FEATURE_SCHEMA_VERSION == "1.1"
    assert snapshot.schema_version == "1.1"
    assert snapshot.get("efficiency_ratio_3") is not None
    assert snapshot.get("price_zscore_3") is not None
    assert snapshot.get("volatility_percentile_2_3") is not None
    assert all(value is None or isfinite(value) for value in snapshot.to_dict().values())


@pytest.mark.parametrize(
    "feature_name",
    ("efficiency_ratio_3", "price_zscore_3", "volatility_percentile_2_3"),
)
def test_lot5_features_are_invariant_when_future_bars_are_appended(
    feature_name: str,
) -> None:
    observed = _bars([100, 101, 99, 102, 100, 103])
    future = _bars([500, 50, 900])
    shifted_future = tuple(
        bar(
            len(observed) + index,
            opening=str(item.close),
            high=str(item.close + 1),
            low=str(item.close - 1),
            close=str(item.close),
        )
        for index, item in enumerate(future)
    )
    engine = FeatureEngine()
    at_time = observed[-1].timestamp

    before = engine.compute(observed, _request(), as_of=at_time)
    after = engine.compute(
        (*observed, *shifted_future), _request(), as_of=at_time
    )

    assert after.get(feature_name) == before.get(feature_name)
