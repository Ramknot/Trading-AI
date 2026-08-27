"""Explicit deterministic regime helpers restricted to offline tests."""

from __future__ import annotations

from collections.abc import Callable

from trading_ai.features import FeatureRequest, FeatureSnapshot
from trading_ai.regimes.base import RegimeDetector
from trading_ai.regimes.models import (
    RegimeSnapshot,
    RegimeTransition,
    StructureRegime,
    VolatilityRegime,
)


RegimeResolver = Callable[
    [FeatureSnapshot], tuple[StructureRegime, VolatilityRegime]
]


class ScriptedTestRegimeDetector(RegimeDetector):
    """Return test-scripted current regimes without production activation paths."""

    def __init__(self, resolver: RegimeResolver) -> None:
        self._resolver = resolver
        self._transitions: list[RegimeTransition] = []
        self._previous: dict[
            tuple[str, str], tuple[StructureRegime, VolatilityRegime, int]
        ] = {}

    @property
    def detector_name(self) -> str:
        return "scripted-test-regime"

    @property
    def detector_version(self) -> str:
        return "test-only"

    @property
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        return (("mode", "explicit-test-only"),)

    @property
    def config_hash(self) -> str:
        return "e" * 64

    @property
    def feature_request(self) -> FeatureRequest:
        return FeatureRequest(
            sma_windows=(2,),
            ema_windows=(2,),
            return_windows=(1,),
            structure_windows=(2,),
            efficiency_windows=(2,),
            zscore_windows=(2,),
            slope_lookback=1,
            atr_window=2,
            volatility_window=2,
            volatility_percentile_window=2,
            volume_window=2,
        )

    @property
    def transitions(self) -> tuple[RegimeTransition, ...]:
        return tuple(self._transitions)

    def reset(self) -> None:
        self._transitions.clear()
        self._previous.clear()

    def evaluate(self, features: FeatureSnapshot) -> RegimeSnapshot:
        structure, volatility = self._resolver(features)
        key = (features.symbol, features.timeframe)
        previous_structure, previous_volatility, bars = self._previous.get(
            key,
            (StructureRegime.UNKNOWN, VolatilityRegime.UNKNOWN, 0),
        )
        bars = bars + 1 if previous_structure is structure else 1
        transition_from = (
            previous_structure if previous_structure is not structure else None
        )
        transition_reason = (
            "explicit test regime changed" if transition_from is not None else None
        )
        snapshot = RegimeSnapshot(
            snapshot_id=(
                f"test-regime-{features.symbol}-{features.timeframe}-"
                f"{features.timestamp.isoformat()}"
            ),
            symbol=features.symbol,
            timestamp=features.timestamp,
            timeframe=features.timeframe,
            structure_regime=structure,
            volatility_regime=volatility,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            config_hash=self.config_hash,
            bars_in_current_structure_regime=bars,
            evidence=(("source", "scripted-test-only"),),
            reason_codes=("SCRIPTED_TEST_REGIME",),
            transition_from=transition_from,
            transition_reason=transition_reason,
        )
        if (
            previous_structure is not structure
            or previous_volatility is not volatility
        ):
            self._transitions.append(
                RegimeTransition(
                    transition_id=(
                        f"test-transition-{len(self._transitions) + 1:06d}"
                    ),
                    symbol=features.symbol,
                    timestamp=features.timestamp,
                    timeframe=features.timeframe,
                    from_structure=previous_structure,
                    to_structure=structure,
                    from_volatility=previous_volatility,
                    to_volatility=volatility,
                    reason="explicit test regime changed",
                )
            )
        self._previous[key] = (structure, volatility, bars)
        return snapshot
