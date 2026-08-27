from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from backtest_support import START, bar
from trading_ai.core.config import load_runtime_settings
from trading_ai.features import FeatureEngine, FeatureSnapshot, FeatureValue
from trading_ai.regimes import (
    BalancedRegimeDetector,
    StructureRegime,
    VolatilityRegime,
    load_balanced_regime_config,
)


PROFILE = load_runtime_settings().profile


def _config(*, confirmation_bars: int = 2):
    return replace(
        load_balanced_regime_config(PROFILE),
        fast_ema_window=2,
        slow_ema_window=4,
        slope_lookback=1,
        efficiency_window=3,
        trend_efficiency_threshold=Decimal("0.60"),
        range_efficiency_threshold=Decimal("0.40"),
        min_trend_separation=Decimal("0.01"),
        max_range_separation=Decimal("0.005"),
        max_range_price_distance=Decimal("0.05"),
        min_normalized_slope=Decimal("0.001"),
        volatility_window=2,
        volatility_percentile_window=3,
        low_volatility_percentile=Decimal("0.25"),
        high_volatility_percentile=Decimal("0.75"),
        confirmation_bars=confirmation_bars,
    )


def _features(
    index: int,
    *,
    kind: str,
    volatility_percentile: float | None = 0.5,
) -> FeatureSnapshot:
    if kind == "up":
        fast, slow, slope, distance, efficiency = 105.0, 100.0, 2.0, 0.03, 0.9
    elif kind == "down":
        fast, slow, slope, distance, efficiency = 95.0, 100.0, -2.0, -0.03, 0.9
    elif kind == "range":
        fast, slow, slope, distance, efficiency = 100.1, 100.0, 0.0, 0.004, 0.1
    elif kind == "unknown":
        fast = slow = slope = distance = efficiency = None
    else:
        raise AssertionError(kind)
    values = {
        "efficiency_ratio_3": efficiency,
        "ema_2": fast,
        "ema_2_slope_1": slope,
        "ema_4": slow,
        "price_to_ema_4": distance,
        "volatility_percentile_2_3": volatility_percentile,
    }
    return FeatureSnapshot(
        symbol="AAPL",
        timestamp=START + timedelta(days=index),
        timeframe="1d",
        values=tuple(
            FeatureValue(name, value) for name, value in sorted(values.items())
        ),
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    (
        ("up", StructureRegime.TREND_UP),
        ("down", StructureRegime.TREND_DOWN),
        ("range", StructureRegime.RANGE),
    ),
)
def test_structure_regimes_are_confirmed_chronologically(
    kind: str, expected: StructureRegime
) -> None:
    detector = BalancedRegimeDetector(_config(confirmation_bars=2))

    first = detector.evaluate(_features(0, kind=kind))
    second = detector.evaluate(_features(1, kind=kind))

    assert first.structure_regime is StructureRegime.UNKNOWN
    assert first.candidate_structure_regime is expected
    assert first.confirmation_progress == 1
    assert second.structure_regime is expected
    assert second.bars_in_current_structure_regime == 1
    assert second.transition_from is StructureRegime.UNKNOWN


def test_unknown_is_conservative_for_missing_critical_features() -> None:
    detector = BalancedRegimeDetector(_config(confirmation_bars=1))
    snapshot = detector.evaluate(_features(0, kind="unknown"))

    assert snapshot.structure_regime is StructureRegime.UNKNOWN
    assert "STRUCTURE_INPUT_UNAVAILABLE" in snapshot.reason_codes


@pytest.mark.parametrize(
    ("percentile", "expected"),
    ((0.1, VolatilityRegime.LOW), (0.5, VolatilityRegime.NORMAL), (0.9, VolatilityRegime.HIGH)),
)
def test_volatility_regime_is_independent_from_structure(
    percentile: float, expected: VolatilityRegime
) -> None:
    detector = BalancedRegimeDetector(_config(confirmation_bars=1))
    snapshot = detector.evaluate(
        _features(0, kind="up", volatility_percentile=percentile)
    )
    assert snapshot.structure_regime is StructureRegime.TREND_UP
    assert snapshot.volatility_regime is expected


def test_volatility_warmup_remains_unknown() -> None:
    detector = BalancedRegimeDetector(_config(confirmation_bars=1))
    snapshot = detector.evaluate(
        _features(0, kind="range", volatility_percentile=None)
    )
    assert snapshot.volatility_regime is VolatilityRegime.UNKNOWN


def test_confirmation_prevents_one_bar_regime_flip() -> None:
    detector = BalancedRegimeDetector(_config(confirmation_bars=3))
    for index in range(3):
        established = detector.evaluate(_features(index, kind="range"))
    false_break = detector.evaluate(_features(3, kind="up"))

    assert established.structure_regime is StructureRegime.RANGE
    assert false_break.structure_regime is StructureRegime.RANGE
    assert false_break.candidate_structure_regime is StructureRegime.TREND_UP
    assert false_break.confirmation_progress == 1


def test_range_to_trend_transition_is_unique_and_counts_bars() -> None:
    detector = BalancedRegimeDetector(_config(confirmation_bars=2))
    detector.evaluate(_features(0, kind="range"))
    detector.evaluate(_features(1, kind="range"))
    detector.evaluate(_features(2, kind="up"))
    changed = detector.evaluate(_features(3, kind="up"))
    continued = detector.evaluate(_features(4, kind="up"))

    structure_transitions = [
        item
        for item in detector.transitions
        if item.from_structure is StructureRegime.RANGE
        and item.to_structure is StructureRegime.TREND_UP
    ]
    assert len(structure_transitions) == 1
    assert structure_transitions[0].timestamp == START + timedelta(days=3)
    assert changed.bars_in_current_structure_regime == 1
    assert continued.bars_in_current_structure_regime == 2


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


def _detect_through(bars, target_index: int):
    detector = BalancedRegimeDetector(_config(confirmation_bars=2))
    engine = FeatureEngine()
    result = None
    for index in range(target_index + 1):
        timestamp = bars[index].timestamp
        snapshot = engine.compute(
            bars,
            detector.feature_request,
            as_of=timestamp,
        )
        result = detector.evaluate(snapshot)
    assert result is not None
    return result


def test_detector_is_append_future_invariant() -> None:
    observed = _bars([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
    future = tuple(
        bar(
            10 + index,
            opening=str(value),
            high=str(value + 1),
            low=str(value - 1),
            close=str(value),
        )
        for index, value in enumerate((20, 500, 10, 900))
    )

    before = _detect_through(observed, len(observed) - 1)
    after = _detect_through((*observed, *future), len(observed) - 1)

    assert after == before


def test_detector_rejects_out_of_order_evaluation() -> None:
    detector = BalancedRegimeDetector(_config(confirmation_bars=1))
    detector.evaluate(_features(1, kind="up"))
    with pytest.raises(ValueError, match="chronologically"):
        detector.evaluate(_features(0, kind="up"))
