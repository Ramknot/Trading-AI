from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_ai.core.models import MarketBar
from trading_ai.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureEngine,
    FeatureInputError,
    FeatureRequest,
    FeatureSnapshot,
    FeatureValue,
)
from trading_ai.features.momentum import relative_strength_values


START = datetime(2024, 1, 2, 21, tzinfo=timezone.utc)


def _bars(
    closes: list[float],
    *,
    symbol: str = "AAPL",
    timeframe: str = "1d",
    start: datetime = START,
    volumes: list[float] | None = None,
) -> tuple[MarketBar, ...]:
    result = []
    for index, close in enumerate(closes):
        result.append(
            MarketBar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=start + timedelta(days=index),
                open=Decimal(str(close)),
                high=Decimal(str(close + 1)),
                low=Decimal(str(close - 1)),
                close=Decimal(str(close)),
                volume=Decimal(str((volumes or [100.0] * len(closes))[index])),
                source="synthetic",
            )
        )
    return tuple(result)


def _request() -> FeatureRequest:
    return FeatureRequest(
        sma_windows=(3,),
        ema_windows=(3,),
        return_windows=(1, 3),
        structure_windows=(3,),
        slope_lookback=1,
        atr_window=3,
        volatility_window=3,
        volume_window=3,
    )


def test_feature_engine_computes_stable_named_trend_momentum_and_volume() -> None:
    snapshot = FeatureEngine().compute(
        _bars([10, 11, 12, 13], volumes=[100, 100, 100, 200]), _request()
    )

    assert snapshot.schema_version == FEATURE_SCHEMA_VERSION
    assert snapshot.get("sma_3") == pytest.approx(12.0)
    assert snapshot.get("ema_3") == pytest.approx(12.0)
    assert snapshot.get("ema_3_slope_1") == pytest.approx(1.0)
    assert snapshot.get("price_to_ema_3") == pytest.approx(1 / 12)
    assert snapshot.get("simple_return") == pytest.approx(13 / 12 - 1)
    assert snapshot.get("return_3") == pytest.approx(0.3)
    assert snapshot.get("roc_3") == pytest.approx(30.0)
    assert snapshot.get("average_volume_3") == pytest.approx(400 / 3)
    assert snapshot.get("relative_volume_3") == pytest.approx(1.5)


def test_atr_true_range_and_rolling_volatility_are_available_after_warmup() -> None:
    snapshot = FeatureEngine().compute(_bars([10, 12, 11, 14]), _request())

    assert snapshot.get("true_range") == pytest.approx(4.0)
    assert snapshot.get("atr_3") == pytest.approx(26 / 9)
    assert snapshot.get("rolling_vol_3") is not None


def test_structure_reference_excludes_current_bar() -> None:
    bars = list(_bars([10, 11, 12]))
    bars.append(
        MarketBar(
            symbol="AAPL",
            timeframe="1d",
            timestamp=START + timedelta(days=3),
            open=Decimal("15"),
            high=Decimal("100"),
            low=Decimal("14"),
            close=Decimal("99"),
            volume=Decimal("100"),
            source="synthetic",
        )
    )

    snapshot = FeatureEngine().compute(tuple(bars), _request())

    assert snapshot.get("previous_high_3") == pytest.approx(13.0)
    assert snapshot.get("previous_low_3") == pytest.approx(9.0)
    assert snapshot.get("distance_to_previous_high_3") == pytest.approx(99 / 13 - 1)


def test_warmup_is_explicit_and_zero_average_volume_never_returns_infinity() -> None:
    snapshot = FeatureEngine().compute(
        _bars([10, 11], volumes=[0, 0]), _request()
    )

    assert snapshot.get("ema_3") is None
    assert snapshot.get("return_3") is None
    assert snapshot.get("atr_3") is None
    assert snapshot.get("previous_high_3") is None
    assert snapshot.get("relative_volume_3") is None


def test_appending_future_bars_cannot_change_features_at_t() -> None:
    engine = FeatureEngine()
    history = _bars([10, 11, 12, 13, 14])
    with_future = history + _bars(
        [50, 2], start=history[-1].timestamp + timedelta(days=1)
    )

    before = engine.compute(history, _request())
    after = engine.compute(with_future, _request(), as_of=history[-1].timestamp)

    for feature_name in (
        "sma_3",
        "ema_3",
        "return_3",
        "atr_3",
        "previous_high_3",
    ):
        assert after.get(feature_name) == before.get(feature_name)


def test_feature_engine_rejects_disordered_duplicate_and_mixed_histories() -> None:
    bars = _bars([10, 11])
    engine = FeatureEngine()

    with pytest.raises(FeatureInputError, match="strictly increasing"):
        engine.compute(tuple(reversed(bars)), _request())
    with pytest.raises(FeatureInputError, match="strictly increasing"):
        engine.compute((bars[0], bars[0]), _request())
    with pytest.raises(FeatureInputError, match="one symbol"):
        engine.compute((bars[0], _bars([11], symbol="MSFT")[0]), _request())
    with pytest.raises(FeatureInputError, match="timezone-aware"):
        engine.compute(bars, _request(), as_of=datetime(2024, 1, 2))


def test_relative_strength_uses_only_exact_timestamp_and_reports_missing() -> None:
    aapl = _bars([100, 110, 120, 130], symbol="AAPL")
    msft = _bars([100, 105, 110, 115], symbol="MSFT")
    meta = _bars([100, 90, 80], symbol="META")
    history = tuple(sorted(aapl + msft + meta, key=lambda bar: (bar.timestamp, bar.symbol)))

    snapshot = FeatureEngine().relative_strength(
        history,
        symbols=("AAPL", "MSFT", "META"),
        timeframe="1d",
        as_of=aapl[-1].timestamp,
        lookback=3,
    )

    assert snapshot.ranked_symbols == ("AAPL", "MSFT")
    assert snapshot.missing_symbols == ("META",)
    assert snapshot.values[0].rolling_return > snapshot.values[1].rolling_return


def test_relative_strength_ties_share_rank_but_selection_order_is_stable() -> None:
    values = relative_strength_values({"MSFT": 0.1, "AAPL": 0.1, "META": -0.1})

    by_symbol = {item.symbol: item for item in values}
    assert by_symbol["AAPL"].rank == by_symbol["MSFT"].rank == 1.5
    assert by_symbol["AAPL"].percentile == by_symbol["MSFT"].percentile == 0.75
    assert tuple(item.symbol for item in values) == ("AAPL", "MSFT", "META")


def test_relative_strength_at_t_is_unchanged_by_future_cross_asset_bars() -> None:
    engine = FeatureEngine()
    aapl = _bars([100, 110, 120, 130], symbol="AAPL")
    msft = _bars([100, 105, 110, 115], symbol="MSFT")
    as_of = aapl[-1].timestamp
    initial = tuple(sorted(aapl + msft, key=lambda bar: (bar.timestamp, bar.symbol)))
    future = initial + tuple(
        sorted(
            _bars([1000], symbol="AAPL", start=as_of + timedelta(days=1))
            + _bars([2], symbol="MSFT", start=as_of + timedelta(days=1)),
            key=lambda bar: (bar.timestamp, bar.symbol),
        )
    )

    before = engine.relative_strength(
        initial,
        symbols=("AAPL", "MSFT"),
        timeframe="1d",
        as_of=as_of,
        lookback=3,
    )
    after = engine.relative_strength(
        future,
        symbols=("AAPL", "MSFT"),
        timeframe="1d",
        as_of=as_of,
        lookback=3,
    )

    assert after == before


def test_feature_models_and_requests_are_immutable() -> None:
    snapshot = FeatureSnapshot(
        symbol="AAPL",
        timestamp=START,
        timeframe="1d",
        values=(FeatureValue("alpha", 1.0),),
    )
    request = _request()

    with pytest.raises(FrozenInstanceError):
        snapshot.symbol = "MSFT"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        request.atr_window = 10  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        FeatureValue("bad", float("inf"))
    with pytest.raises(ValueError, match="timezone-aware"):
        FeatureSnapshot(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2),
            timeframe="1d",
            values=(FeatureValue("alpha", 1.0),),
        )


def test_feature_cache_is_exact_and_bounded() -> None:
    engine = FeatureEngine(cache_size=1)
    bars = _bars([10, 11, 12, 13])

    first = engine.compute(bars, _request())
    repeated = engine.compute(bars, _request())
    engine.compute(_bars([20, 21, 22, 23]), _request())

    assert repeated is first
    assert len(engine._cache) == 1
