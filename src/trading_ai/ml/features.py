"""Point-in-time ML feature vectors built only from shared feature/regime models."""

from __future__ import annotations

from trading_ai.backtesting.models import StrategySignal
from trading_ai.features import FEATURE_SCHEMA_VERSION, FeatureRequest, FeatureSnapshot
from trading_ai.ml.exceptions import MLDataError
from trading_ai.ml.inputs import TabularModelInput
from trading_ai.regimes.models import RegimeSnapshot, StructureRegime, VolatilityRegime


ML_FEATURE_SCHEMA_VERSION = "1.0"

COMMON_ML_FEATURE_NAMES = (
    "return_20",
    "rolling_vol_20",
    "relative_volume_20",
    "efficiency_ratio_20",
    "volatility_percentile_20_100",
    "price_zscore_20",
    "price_to_ema_20",
    "price_to_ema_50",
    "structure_TREND_UP",
    "structure_TREND_DOWN",
    "structure_RANGE",
    "structure_UNKNOWN",
    "vol_LOW",
    "vol_NORMAL",
    "vol_HIGH",
    "vol_UNKNOWN",
)


class MLFeatureBuilder:
    """Translate shared point-in-time snapshots into stable normalized inputs."""

    schema_version = ML_FEATURE_SCHEMA_VERSION
    feature_request = FeatureRequest(
        sma_windows=(20, 50),
        ema_windows=(20, 50),
        return_windows=(1, 20),
        structure_windows=(20,),
        efficiency_windows=(20,),
        zscore_windows=(20,),
        slope_lookback=5,
        atr_window=14,
        volatility_window=20,
        volatility_percentile_window=100,
        volume_window=20,
    )

    @staticmethod
    def feature_names(strategy_name: str) -> tuple[str, ...]:
        return (
            (*COMMON_ML_FEATURE_NAMES, "relative_strength_percentile")
            if strategy_name == "momentum"
            else COMMON_ML_FEATURE_NAMES
        )
    def build(
        self,
        *,
        signal: StrategySignal,
        features: FeatureSnapshot,
        regime: RegimeSnapshot,
    ) -> TabularModelInput:
        if (
            signal.symbol != features.symbol
            or signal.symbol != regime.symbol
            or signal.timeframe != features.timeframe
            or signal.timeframe != regime.timeframe
            or signal.timestamp != features.timestamp
            or signal.timestamp != regime.timestamp
        ):
            raise MLDataError(
                "signal, FeatureSnapshot, and RegimeSnapshot must describe one event"
            )
        if features.schema_version != FEATURE_SCHEMA_VERSION:
            raise MLDataError(
                f"unsupported Feature schema {features.schema_version!r}"
            )
        shared_names = COMMON_ML_FEATURE_NAMES[:8]
        values: dict[str, float] = {}
        for name in shared_names:
            value = features.get(name)
            if value is None:
                raise MLDataError(f"required ML feature {name} is unavailable")
            values[name] = value
        for structure in StructureRegime:
            values[f"structure_{structure.value}"] = float(
                regime.structure_regime is structure
            )
        for volatility in VolatilityRegime:
            values[f"vol_{volatility.value}"] = float(
                regime.volatility_regime is volatility
            )
        if signal.strategy_name == "momentum":
            metadata = dict(signal.features_used)
            try:
                values["relative_strength_percentile"] = float(
                    metadata["relative_strength_percentile"]
                )
            except (KeyError, ValueError) as exc:
                raise MLDataError(
                    "momentum ML input requires relative_strength_percentile"
                ) from exc
        ordered_names = self.feature_names(signal.strategy_name)
        return TabularModelInput(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            timeframe=signal.timeframe,
            strategy_name=signal.strategy_name,
            strategy_version=signal.strategy_version,
            feature_schema_version=features.schema_version,
            ml_feature_schema_version=self.schema_version,
            values=tuple((name, values[name]) for name in ordered_names),
        )
