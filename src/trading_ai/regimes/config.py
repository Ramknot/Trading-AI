"""TOML configuration and stable hashes for Lot 5 regime components."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.models import TradingProfile, TradingProfileName
from trading_ai.features import FeatureRequest
from trading_ai.regimes.exceptions import RegimeConfigurationError
from trading_ai.regimes.models import (
    ActivationStatus,
    StructureRegime,
    VolatilityRegime,
)


DEFAULT_REGIME_DIRECTORY = PROJECT_ROOT / "config" / "regimes"
SUPPORTED_POLICY_STRATEGIES = (
    "breakout",
    "mean-reversion",
    "momentum",
    "trend",
)
ZERO = Decimal("0")
ONE = Decimal("1")


def _decimal(raw: object, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise RegimeConfigurationError(f"{name} must be numeric") from exc
    if not value.is_finite():
        raise RegimeConfigurationError(f"{name} must be finite")
    return value


def _integer(raw: object, name: str) -> int:
    if type(raw) is not int:
        raise RegimeConfigurationError(f"{name} must be a TOML integer")
    return raw


def _boolean(raw: object, name: str) -> bool:
    if type(raw) is not bool:
        raise RegimeConfigurationError(f"{name} must be a TOML boolean")
    return raw


def _profile_name(value: str | TradingProfileName) -> TradingProfileName:
    try:
        return (
            value
            if isinstance(value, TradingProfileName)
            else TradingProfileName(value.strip().lower())
        )
    except (AttributeError, ValueError) as exc:
        raise RegimeConfigurationError(f"unknown regime profile {value!r}") from exc


@dataclass(frozen=True, slots=True)
class BalancedRegimeConfig:
    name: TradingProfileName
    enabled: bool
    detector_name: str
    detector_version: str
    fast_ema_window: int
    slow_ema_window: int
    slope_lookback: int
    efficiency_window: int
    trend_efficiency_threshold: Decimal
    range_efficiency_threshold: Decimal
    min_trend_separation: Decimal
    max_range_separation: Decimal
    max_range_price_distance: Decimal
    min_normalized_slope: Decimal
    volatility_window: int
    volatility_percentile_window: int
    low_volatility_percentile: Decimal
    high_volatility_percentile: Decimal
    confirmation_bars: int

    def __post_init__(self) -> None:
        if not self.detector_name.strip() or not self.detector_version.strip():
            raise RegimeConfigurationError("regime detector identity must not be empty")
        integer_fields = (
            "fast_ema_window",
            "slow_ema_window",
            "slope_lookback",
            "efficiency_window",
            "volatility_window",
            "volatility_percentile_window",
            "confirmation_bars",
        )
        if any(getattr(self, field_name) < 1 for field_name in integer_fields):
            raise RegimeConfigurationError("regime windows must be positive")
        if self.volatility_window < 2 or self.volatility_percentile_window < 2:
            raise RegimeConfigurationError("volatility windows must be at least 2")
        if self.fast_ema_window >= self.slow_ema_window:
            raise RegimeConfigurationError("fast_ema_window must be below slow_ema_window")
        if not (
            ZERO
            <= self.range_efficiency_threshold
            < self.trend_efficiency_threshold
            <= ONE
        ):
            raise RegimeConfigurationError(
                "efficiency thresholds require 0 <= range < trend <= 1"
            )
        if any(
            value < ZERO
            for value in (
                self.min_trend_separation,
                self.max_range_separation,
                self.max_range_price_distance,
                self.min_normalized_slope,
            )
        ):
            raise RegimeConfigurationError("regime separation/slope limits cannot be negative")
        if self.max_range_separation >= self.min_trend_separation:
            raise RegimeConfigurationError(
                "max_range_separation must be below min_trend_separation"
            )
        if not (
            ZERO
            <= self.low_volatility_percentile
            < self.high_volatility_percentile
            <= ONE
        ):
            raise RegimeConfigurationError(
                "volatility percentiles require 0 <= low < high <= 1"
            )

    @property
    def feature_request(self) -> FeatureRequest:
        return FeatureRequest(
            sma_windows=(self.fast_ema_window, self.slow_ema_window),
            ema_windows=(self.fast_ema_window, self.slow_ema_window),
            return_windows=(1,),
            structure_windows=(self.efficiency_window,),
            efficiency_windows=(self.efficiency_window,),
            zscore_windows=(self.efficiency_window,),
            slope_lookback=self.slope_lookback,
            volatility_window=self.volatility_window,
            volatility_percentile_window=self.volatility_percentile_window,
        )

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    ("confirmation_bars", str(self.confirmation_bars)),
                    ("detector_name", self.detector_name),
                    ("detector_version", self.detector_version),
                    ("efficiency_window", str(self.efficiency_window)),
                    ("enabled", str(self.enabled).lower()),
                    ("fast_ema_window", str(self.fast_ema_window)),
                    (
                        "high_volatility_percentile",
                        str(self.high_volatility_percentile),
                    ),
                    (
                        "low_volatility_percentile",
                        str(self.low_volatility_percentile),
                    ),
                    ("max_range_price_distance", str(self.max_range_price_distance)),
                    ("max_range_separation", str(self.max_range_separation)),
                    ("min_normalized_slope", str(self.min_normalized_slope)),
                    ("min_trend_separation", str(self.min_trend_separation)),
                    ("name", self.name.value),
                    (
                        "range_efficiency_threshold",
                        str(self.range_efficiency_threshold),
                    ),
                    ("slope_lookback", str(self.slope_lookback)),
                    ("slow_ema_window", str(self.slow_ema_window)),
                    (
                        "trend_efficiency_threshold",
                        str(self.trend_efficiency_threshold),
                    ),
                    (
                        "volatility_percentile_window",
                        str(self.volatility_percentile_window),
                    ),
                    ("volatility_window", str(self.volatility_window)),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class StrategyPolicyRule:
    structure: StructureRegime
    strategy_name: str
    status: ActivationStatus
    multiplier: Decimal

    def __post_init__(self) -> None:
        if self.strategy_name not in SUPPORTED_POLICY_STRATEGIES:
            raise RegimeConfigurationError(
                f"unsupported policy strategy {self.strategy_name!r}"
            )
        if not ZERO <= self.multiplier <= ONE:
            raise RegimeConfigurationError("policy multiplier must be between 0 and 1")
        if self.status is ActivationStatus.ALLOW and self.multiplier != ONE:
            raise RegimeConfigurationError("ALLOW policy rules require multiplier 1")
        if self.status is ActivationStatus.REDUCE and not ZERO < self.multiplier < ONE:
            raise RegimeConfigurationError("REDUCE rules require multiplier in (0, 1)")
        if self.status is ActivationStatus.BLOCK and self.multiplier != ZERO:
            raise RegimeConfigurationError("BLOCK policy rules require multiplier 0")


@dataclass(frozen=True, slots=True)
class VolatilityPolicyOverlay:
    volatility: VolatilityRegime
    strategy_name: str
    status: ActivationStatus
    multiplier: Decimal

    def __post_init__(self) -> None:
        StrategyPolicyRule(
            StructureRegime.UNKNOWN,
            self.strategy_name,
            self.status,
            self.multiplier,
        )


@dataclass(frozen=True, slots=True)
class BalancedStrategyPolicyConfig:
    name: TradingProfileName
    enabled: bool
    policy_name: str
    policy_version: str
    structure_rules: tuple[StrategyPolicyRule, ...]
    volatility_overlays: tuple[VolatilityPolicyOverlay, ...]

    def __post_init__(self) -> None:
        if not self.policy_name.strip() or not self.policy_version.strip():
            raise RegimeConfigurationError("strategy policy identity must not be empty")
        rule_keys = [
            (rule.structure.value, rule.strategy_name)
            for rule in self.structure_rules
        ]
        if rule_keys != sorted(rule_keys) or len(rule_keys) != len(set(rule_keys)):
            raise RegimeConfigurationError("strategy policy rules must be sorted and unique")
        expected = {
            (structure.value, strategy)
            for structure in StructureRegime
            for strategy in SUPPORTED_POLICY_STRATEGIES
        }
        if set(rule_keys) != expected:
            raise RegimeConfigurationError(
                "strategy policy must define every structure/strategy combination"
            )
        overlay_keys = [
            (overlay.volatility.value, overlay.strategy_name)
            for overlay in self.volatility_overlays
        ]
        if overlay_keys != sorted(overlay_keys) or len(overlay_keys) != len(
            set(overlay_keys)
        ):
            raise RegimeConfigurationError("volatility overlays must be sorted and unique")

    def rule_for(
        self, structure: StructureRegime, strategy_name: str
    ) -> StrategyPolicyRule:
        try:
            return next(
                rule
                for rule in self.structure_rules
                if rule.structure is structure and rule.strategy_name == strategy_name
            )
        except StopIteration as exc:
            raise RegimeConfigurationError(
                f"no activation rule for {structure.value}/{strategy_name}"
            ) from exc

    def overlay_for(
        self, volatility: VolatilityRegime, strategy_name: str
    ) -> VolatilityPolicyOverlay | None:
        return next(
            (
                overlay
                for overlay in self.volatility_overlays
                if overlay.volatility is volatility
                and overlay.strategy_name == strategy_name
            ),
            None,
        )

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        values: list[tuple[str, str]] = [
            ("enabled", str(self.enabled).lower()),
            ("name", self.name.value),
            ("policy_name", self.policy_name),
            ("policy_version", self.policy_version),
        ]
        for rule in self.structure_rules:
            prefix = f"structure.{rule.structure.value}.{rule.strategy_name}"
            values.extend(
                ((f"{prefix}.status", rule.status.value), (f"{prefix}.multiplier", str(rule.multiplier)))
            )
        for overlay in self.volatility_overlays:
            prefix = f"volatility.{overlay.volatility.value}.{overlay.strategy_name}"
            values.extend(
                ((f"{prefix}.status", overlay.status.value), (f"{prefix}.multiplier", str(overlay.multiplier)))
            )
        return tuple(sorted(values))


def _stable_hash(parameters: tuple[tuple[str, str], ...]) -> str:
    payload = json.dumps(
        list(parameters), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def regime_config_hash(config: BalancedRegimeConfig) -> str:
    return _stable_hash(config.to_parameters())


def strategy_policy_config_hash(config: BalancedStrategyPolicyConfig) -> str:
    return _stable_hash(config.to_parameters())


def inspect_regime_config(
    profile_name: str | TradingProfileName,
    *,
    regime_directory: Path = DEFAULT_REGIME_DIRECTORY,
) -> BalancedRegimeConfig:
    name = _profile_name(profile_name)
    path = regime_directory / f"{name.value}.toml"
    if not path.is_file():
        raise RegimeConfigurationError(f"regime configuration not found: {path}")
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        config = BalancedRegimeConfig(
            name=TradingProfileName(str(raw["name"])),
            enabled=_boolean(raw["enabled"], "enabled"),
            detector_name=str(raw["detector_name"]),
            detector_version=str(raw["detector_version"]),
            fast_ema_window=_integer(raw["fast_ema_window"], "fast_ema_window"),
            slow_ema_window=_integer(raw["slow_ema_window"], "slow_ema_window"),
            slope_lookback=_integer(raw["slope_lookback"], "slope_lookback"),
            efficiency_window=_integer(raw["efficiency_window"], "efficiency_window"),
            trend_efficiency_threshold=_decimal(
                raw["trend_efficiency_threshold"], "trend_efficiency_threshold"
            ),
            range_efficiency_threshold=_decimal(
                raw["range_efficiency_threshold"], "range_efficiency_threshold"
            ),
            min_trend_separation=_decimal(raw["min_trend_separation"], "min_trend_separation"),
            max_range_separation=_decimal(raw["max_range_separation"], "max_range_separation"),
            max_range_price_distance=_decimal(
                raw["max_range_price_distance"], "max_range_price_distance"
            ),
            min_normalized_slope=_decimal(raw["min_normalized_slope"], "min_normalized_slope"),
            volatility_window=_integer(raw["volatility_window"], "volatility_window"),
            volatility_percentile_window=_integer(
                raw["volatility_percentile_window"], "volatility_percentile_window"
            ),
            low_volatility_percentile=_decimal(
                raw["low_volatility_percentile"], "low_volatility_percentile"
            ),
            high_volatility_percentile=_decimal(
                raw["high_volatility_percentile"], "high_volatility_percentile"
            ),
            confirmation_bars=_integer(raw["confirmation_bars"], "confirmation_bars"),
        )
    except RegimeConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RegimeConfigurationError(f"invalid regime configuration {path}: {exc}") from exc
    if config.name is not name:
        raise RegimeConfigurationError(
            f"regime configuration declares {config.name.value!r}, expected {name.value!r}"
        )
    return config


def _policy_rule(
    structure: StructureRegime,
    strategy_name: str,
    raw: object,
) -> StrategyPolicyRule:
    if not isinstance(raw, dict):
        raise RegimeConfigurationError("policy rule must be a TOML table")
    try:
        return StrategyPolicyRule(
            structure=structure,
            strategy_name=strategy_name,
            status=ActivationStatus(str(raw["status"])),
            multiplier=_decimal(raw["multiplier"], "policy multiplier"),
        )
    except (KeyError, ValueError) as exc:
        raise RegimeConfigurationError(f"invalid policy rule: {exc}") from exc


def inspect_strategy_policy_config(
    profile_name: str | TradingProfileName,
    *,
    regime_directory: Path = DEFAULT_REGIME_DIRECTORY,
) -> BalancedStrategyPolicyConfig:
    name = _profile_name(profile_name)
    path = regime_directory / f"strategy_policy_{name.value}.toml"
    if not path.is_file():
        raise RegimeConfigurationError(f"strategy policy configuration not found: {path}")
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        structure_raw = raw["structure"]
        volatility_raw = raw.get("volatility", {})
        rules = tuple(
            sorted(
                (
                    _policy_rule(structure, strategy_name, structure_raw[structure.value][strategy_name])
                    for structure in StructureRegime
                    for strategy_name in SUPPORTED_POLICY_STRATEGIES
                ),
                key=lambda rule: (rule.structure.value, rule.strategy_name),
            )
        )
        overlays = tuple(
            sorted(
                (
                    VolatilityPolicyOverlay(
                        volatility=VolatilityRegime(volatility_name),
                        strategy_name=strategy_name,
                        status=ActivationStatus(str(values["status"])),
                        multiplier=_decimal(values["multiplier"], "overlay multiplier"),
                    )
                    for volatility_name, strategies in volatility_raw.items()
                    for strategy_name, values in strategies.items()
                ),
                key=lambda overlay: (overlay.volatility.value, overlay.strategy_name),
            )
        )
        config = BalancedStrategyPolicyConfig(
            name=TradingProfileName(str(raw["name"])),
            enabled=_boolean(raw["enabled"], "enabled"),
            policy_name=str(raw["policy_name"]),
            policy_version=str(raw["policy_version"]),
            structure_rules=rules,
            volatility_overlays=overlays,
        )
    except RegimeConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RegimeConfigurationError(f"invalid strategy policy {path}: {exc}") from exc
    if config.name is not name:
        raise RegimeConfigurationError(
            f"strategy policy declares {config.name.value!r}, expected {name.value!r}"
        )
    return config


def load_balanced_regime_config(
    profile: TradingProfile,
    *,
    regime_directory: Path = DEFAULT_REGIME_DIRECTORY,
) -> BalancedRegimeConfig:
    config = inspect_regime_config(profile.name, regime_directory=regime_directory)
    if profile.name is not TradingProfileName.BALANCED:
        raise RegimeConfigurationError("aggressive regime detection remains locked")
    if not profile.enabled or not config.enabled:
        raise RegimeConfigurationError("Balanced profile and regime config must be enabled")
    return config


def load_balanced_strategy_policy_config(
    profile: TradingProfile,
    *,
    regime_directory: Path = DEFAULT_REGIME_DIRECTORY,
) -> BalancedStrategyPolicyConfig:
    config = inspect_strategy_policy_config(
        profile.name, regime_directory=regime_directory
    )
    if profile.name is not TradingProfileName.BALANCED:
        raise RegimeConfigurationError("aggressive strategy policy remains locked")
    if not profile.enabled or not config.enabled:
        raise RegimeConfigurationError("Balanced profile and strategy policy must be enabled")
    return config
