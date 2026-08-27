"""Deterministic, explainable two-axis Balanced regime detector."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

from trading_ai.core.models import TradingProfile
from trading_ai.features import FeatureRequest, FeatureSnapshot
from trading_ai.regimes.base import RegimeDetector
from trading_ai.regimes.config import (
    BalancedRegimeConfig,
    load_balanced_regime_config,
    regime_config_hash,
)
from trading_ai.regimes.exceptions import RegimeInputError
from trading_ai.regimes.models import (
    RegimeSnapshot,
    RegimeTransition,
    StructureRegime,
    VolatilityRegime,
)


@dataclass(slots=True)
class _DetectorState:
    structure: StructureRegime = StructureRegime.UNKNOWN
    volatility: VolatilityRegime = VolatilityRegime.UNKNOWN
    bars_in_structure: int = 0
    candidate: StructureRegime = StructureRegime.UNKNOWN
    candidate_progress: int = 0
    last_snapshot: RegimeSnapshot | None = None


def _stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _format(value: float | None) -> str:
    return "unavailable" if value is None else format(value, ".15g")


class BalancedRegimeDetector(RegimeDetector):
    """Rule-based structure plus volatility context, with confirmation state."""

    def __init__(self, config: BalancedRegimeConfig) -> None:
        if not config.enabled:
            raise RegimeInputError("disabled regime configuration cannot be activated")
        self.config = config
        self._config_hash = regime_config_hash(config)
        self._states: dict[tuple[str, str], _DetectorState] = {}
        self._transitions: list[RegimeTransition] = []

    @classmethod
    def from_profile(cls, profile: TradingProfile) -> BalancedRegimeDetector:
        return cls(load_balanced_regime_config(profile))

    @property
    def detector_name(self) -> str:
        return self.config.detector_name

    @property
    def detector_version(self) -> str:
        return self.config.detector_version

    @property
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        return self.config.to_parameters()

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def feature_request(self) -> FeatureRequest:
        return self.config.feature_request

    @property
    def transitions(self) -> tuple[RegimeTransition, ...]:
        return tuple(self._transitions)

    def reset(self) -> None:
        self._states.clear()
        self._transitions.clear()

    def evaluate(self, features: FeatureSnapshot) -> RegimeSnapshot:
        key = (features.symbol, features.timeframe)
        state = self._states.setdefault(key, _DetectorState())
        if state.last_snapshot is not None:
            if features.timestamp < state.last_snapshot.timestamp:
                raise RegimeInputError("regime snapshots must be evaluated chronologically")
            if features.timestamp == state.last_snapshot.timestamp:
                return state.last_snapshot

        candidate, evidence, structure_codes = self._classify_structure(features)
        volatility, volatility_codes = self._classify_volatility(features)
        previous_structure = state.structure
        previous_volatility = state.volatility
        transition_from: StructureRegime | None = None
        transition_reason: str | None = None

        if candidate is StructureRegime.UNKNOWN:
            state.candidate = StructureRegime.UNKNOWN
            state.candidate_progress = 0
            if state.structure is not StructureRegime.UNKNOWN:
                transition_from = state.structure
                state.structure = StructureRegime.UNKNOWN
                state.bars_in_structure = 1
                transition_reason = "critical structure evidence became unavailable or contradictory"
            else:
                state.bars_in_structure += 1
        elif candidate is state.structure:
            state.candidate = StructureRegime.UNKNOWN
            state.candidate_progress = 0
            state.bars_in_structure += 1
        else:
            if candidate is state.candidate:
                state.candidate_progress += 1
            else:
                state.candidate = candidate
                state.candidate_progress = 1
            if state.candidate_progress >= self.config.confirmation_bars:
                transition_from = state.structure
                state.structure = candidate
                state.bars_in_structure = 1
                state.candidate = StructureRegime.UNKNOWN
                state.candidate_progress = 0
                transition_reason = (
                    f"{candidate.value} confirmed for "
                    f"{self.config.confirmation_bars} consecutive bars"
                )
            else:
                state.bars_in_structure += 1
                structure_codes.append("STRUCTURE_CONFIRMATION_PENDING")

        state.volatility = volatility
        reason_codes = tuple(sorted(set((*structure_codes, *volatility_codes))))
        evidence_values = tuple(sorted(evidence))
        snapshot_payload = {
            "symbol": features.symbol,
            "timestamp": features.timestamp.isoformat(),
            "timeframe": features.timeframe,
            "structure": state.structure.value,
            "volatility": state.volatility.value,
            "candidate": state.candidate.value,
            "progress": state.candidate_progress,
            "bars": state.bars_in_structure,
            "config_hash": self._config_hash,
            "evidence": evidence_values,
        }
        snapshot = RegimeSnapshot(
            snapshot_id=_stable_id("regime", snapshot_payload),
            symbol=features.symbol,
            timestamp=features.timestamp,
            timeframe=features.timeframe,
            structure_regime=state.structure,
            volatility_regime=state.volatility,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            config_hash=self._config_hash,
            bars_in_current_structure_regime=max(1, state.bars_in_structure),
            evidence=evidence_values,
            reason_codes=reason_codes,
            candidate_structure_regime=state.candidate,
            confirmation_progress=state.candidate_progress,
            transition_from=transition_from,
            transition_reason=transition_reason,
        )

        if (
            previous_structure is not state.structure
            or previous_volatility is not state.volatility
        ):
            reason = transition_reason or (
                f"volatility changed from {previous_volatility.value} "
                f"to {state.volatility.value}"
            )
            transition_payload = {
                "snapshot_id": snapshot.snapshot_id,
                "from_structure": previous_structure.value,
                "to_structure": state.structure.value,
                "from_volatility": previous_volatility.value,
                "to_volatility": state.volatility.value,
                "reason": reason,
            }
            self._transitions.append(
                RegimeTransition(
                    transition_id=_stable_id("regime-transition", transition_payload),
                    symbol=features.symbol,
                    timestamp=features.timestamp,
                    timeframe=features.timeframe,
                    from_structure=previous_structure,
                    to_structure=state.structure,
                    from_volatility=previous_volatility,
                    to_volatility=state.volatility,
                    reason=reason,
                )
            )
        state.last_snapshot = snapshot
        return snapshot

    def _classify_structure(
        self, features: FeatureSnapshot
    ) -> tuple[StructureRegime, list[tuple[str, str]], list[str]]:
        fast_name = f"ema_{self.config.fast_ema_window}"
        slow_name = f"ema_{self.config.slow_ema_window}"
        slope_name = (
            f"ema_{self.config.fast_ema_window}_slope_{self.config.slope_lookback}"
        )
        distance_name = f"price_to_ema_{self.config.slow_ema_window}"
        efficiency_name = f"efficiency_ratio_{self.config.efficiency_window}"
        required = (fast_name, slow_name, slope_name, distance_name, efficiency_name)
        values = {name: features.get(name) for name in required}
        evidence = [(name, _format(value)) for name, value in values.items()]
        if any(value is None for value in values.values()):
            return StructureRegime.UNKNOWN, evidence, ["STRUCTURE_INPUT_UNAVAILABLE"]
        fast = values[fast_name]
        slow = values[slow_name]
        slope = values[slope_name]
        distance = values[distance_name]
        efficiency = values[efficiency_name]
        assert fast is not None and slow is not None and slope is not None
        assert distance is not None and efficiency is not None
        if slow == 0.0:
            return StructureRegime.UNKNOWN, evidence, ["STRUCTURE_INPUT_INVALID"]
        separation = fast / slow - 1.0
        normalized_slope = slope / slow
        evidence.extend(
            (
                ("ema_separation", _format(separation)),
                ("normalized_ema_slope", _format(normalized_slope)),
            )
        )
        trend_efficiency = float(self.config.trend_efficiency_threshold)
        range_efficiency = float(self.config.range_efficiency_threshold)
        min_separation = float(self.config.min_trend_separation)
        max_separation = float(self.config.max_range_separation)
        max_distance = float(self.config.max_range_price_distance)
        min_slope = float(self.config.min_normalized_slope)
        if (
            separation >= min_separation
            and normalized_slope >= min_slope
            and distance > 0.0
            and efficiency >= trend_efficiency
        ):
            return StructureRegime.TREND_UP, evidence, ["STRUCTURE_TREND_UP_CANDIDATE"]
        if (
            separation <= -min_separation
            and normalized_slope <= -min_slope
            and distance < 0.0
            and efficiency >= trend_efficiency
        ):
            return StructureRegime.TREND_DOWN, evidence, ["STRUCTURE_TREND_DOWN_CANDIDATE"]
        if (
            efficiency <= range_efficiency
            and abs(separation) <= max_separation
            and abs(distance) <= max_distance
        ):
            return StructureRegime.RANGE, evidence, ["STRUCTURE_RANGE_CANDIDATE"]
        return StructureRegime.UNKNOWN, evidence, ["STRUCTURE_RULES_CONTRADICTORY"]

    def _classify_volatility(
        self, features: FeatureSnapshot
    ) -> tuple[VolatilityRegime, list[str]]:
        name = (
            "volatility_percentile_"
            f"{self.config.volatility_window}_"
            f"{self.config.volatility_percentile_window}"
        )
        percentile = features.get(name)
        if percentile is None:
            return VolatilityRegime.UNKNOWN, ["VOLATILITY_INPUT_UNAVAILABLE"]
        if percentile <= float(self.config.low_volatility_percentile):
            return VolatilityRegime.LOW, ["VOLATILITY_LOW"]
        if percentile >= float(self.config.high_volatility_percentile):
            return VolatilityRegime.HIGH, ["VOLATILITY_HIGH"]
        return VolatilityRegime.NORMAL, ["VOLATILITY_NORMAL"]
