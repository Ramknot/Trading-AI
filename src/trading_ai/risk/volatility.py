"""Feature-driven volatility guard; it never defines a duplicate indicator."""

from __future__ import annotations

from trading_ai.features.models import FeatureSnapshot
from trading_ai.risk.config import BalancedRiskConfig
from trading_ai.risk.models import (
    RiskReasonCode,
    VolatilityAssessment,
    VolatilityLevel,
)


class VolatilityGuard:
    def __init__(self, config: BalancedRiskConfig) -> None:
        self._config = config

    def assess(
        self,
        snapshot: FeatureSnapshot | None,
        *,
        timeframe: str,
    ) -> VolatilityAssessment:
        threshold = self._config.threshold_for(timeframe)
        metric = None
        if snapshot is not None:
            try:
                metric = snapshot.get(self._config.volatility_feature_name)
            except KeyError:
                metric = None
        if threshold is None or metric is None:
            return VolatilityAssessment(
                level=VolatilityLevel.UNKNOWN,
                metric=None,
                multiplier=self._config.normal_volatility_multiplier,
                reason_code=RiskReasonCode.VOLATILITY_UNKNOWN,
            )
        if metric >= float(threshold.extreme):
            return VolatilityAssessment(
                level=VolatilityLevel.EXTREME,
                metric=metric,
                multiplier=self._config.extreme_volatility_multiplier,
                reason_code=RiskReasonCode.VOLATILITY_LIMIT,
            )
        if metric >= float(threshold.elevated):
            return VolatilityAssessment(
                level=VolatilityLevel.ELEVATED,
                metric=metric,
                multiplier=self._config.elevated_volatility_multiplier,
                reason_code=RiskReasonCode.VOLATILITY_LIMIT,
            )
        return VolatilityAssessment(
            level=VolatilityLevel.NORMAL,
            metric=metric,
            multiplier=self._config.normal_volatility_multiplier,
        )
