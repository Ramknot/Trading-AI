"""Dedicated, profile-bounded TOML configuration for Balanced risk."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.models import TradingProfile, TradingProfileName
from trading_ai.risk.exceptions import RiskConfigurationError
from trading_ai.risk.models import UnknownRiskPolicy


DEFAULT_RISK_DIRECTORY = PROJECT_ROOT / "config" / "risk"
DEFAULT_ASSET_GROUPS_PATH = DEFAULT_RISK_DIRECTORY / "asset_groups.toml"
ZERO = Decimal("0")
ONE = Decimal("1")


def _decimal(raw: object, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise RiskConfigurationError(f"{name} must be numeric") from exc
    if not value.is_finite():
        raise RiskConfigurationError(f"{name} must be finite")
    return value


def _boolean(raw: object, name: str) -> bool:
    if type(raw) is not bool:
        raise RiskConfigurationError(f"{name} must be a TOML boolean")
    return raw


def _integer(raw: object, name: str) -> int:
    if type(raw) is not int:
        raise RiskConfigurationError(f"{name} must be a TOML integer")
    return raw


@dataclass(frozen=True, slots=True)
class VolatilityThreshold:
    timeframe: str
    elevated: Decimal
    extreme: Decimal

    def __post_init__(self) -> None:
        if not self.timeframe.strip():
            raise RiskConfigurationError("volatility timeframe must not be empty")
        if self.elevated < ZERO or self.extreme <= self.elevated:
            raise RiskConfigurationError(
                "volatility thresholds require 0 <= elevated < extreme"
            )


@dataclass(frozen=True, slots=True)
class RiskAssetGroups:
    """Configuration-only symbol classification used for concentration limits."""

    groups: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.groups]
        if names != sorted(names) or len(names) != len(set(names)):
            raise RiskConfigurationError("asset groups must have unique sorted names")
        symbols: list[str] = []
        for name, members in self.groups:
            if not name.strip() or not members:
                raise RiskConfigurationError("asset groups require a name and members")
            if tuple(sorted(set(members))) != members:
                raise RiskConfigurationError(
                    f"asset group {name} symbols must be sorted and unique"
                )
            symbols.extend(members)
        if len(symbols) != len(set(symbols)):
            raise RiskConfigurationError("a symbol may belong to only one risk group")

    @property
    def symbol_mapping(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (symbol, group)
                for group, symbols in self.groups
                for symbol in symbols
            )
        )

    def group_for(self, symbol: str) -> str | None:
        return next(
            (group for item, group in self.symbol_mapping if item == symbol),
            None,
        )


@dataclass(frozen=True, slots=True)
class BalancedRiskConfig:
    """Immutable research limits; never an assertion that trading is safe."""

    name: TradingProfileName
    enabled: bool
    engine_name: str
    engine_version: str
    max_positions: int
    max_portfolio_exposure: Decimal
    max_single_position_exposure: Decimal
    max_group_exposure: Decimal
    max_trade_risk_fraction: Decimal
    daily_loss_limit: Decimal
    soft_drawdown_limit: Decimal
    hard_drawdown_limit: Decimal
    high_correlation_threshold: Decimal
    max_highly_correlated_exposure: Decimal
    correlation_min_observations: int
    correlation_unknown_policy: UnknownRiskPolicy
    unknown_group_policy: UnknownRiskPolicy
    volatility_feature_name: str
    missing_volatility_policy: UnknownRiskPolicy
    normal_volatility_multiplier: Decimal
    elevated_volatility_multiplier: Decimal
    extreme_volatility_multiplier: Decimal
    reduced_risk_multiplier: Decimal
    risk_day_timezone: str
    quantity_step: Decimal
    volatility_thresholds: tuple[VolatilityThreshold, ...]

    def __post_init__(self) -> None:
        if not self.engine_name.strip() or not self.engine_version.strip():
            raise RiskConfigurationError("risk engine identity must not be empty")
        if self.max_positions < 1 or self.correlation_min_observations < 2:
            raise RiskConfigurationError(
                "max_positions must be positive and correlation observations >= 2"
            )
        fractions = (
            "max_portfolio_exposure",
            "max_single_position_exposure",
            "max_group_exposure",
            "max_trade_risk_fraction",
            "daily_loss_limit",
            "soft_drawdown_limit",
            "hard_drawdown_limit",
            "high_correlation_threshold",
            "max_highly_correlated_exposure",
        )
        for field_name in fractions:
            value = getattr(self, field_name)
            if value <= ZERO or value > ONE:
                raise RiskConfigurationError(f"{field_name} must be in (0, 1]")
        if self.max_single_position_exposure > self.max_portfolio_exposure:
            raise RiskConfigurationError(
                "single-position exposure cannot exceed portfolio exposure"
            )
        if self.max_group_exposure > self.max_portfolio_exposure:
            raise RiskConfigurationError("group exposure cannot exceed portfolio exposure")
        if self.max_highly_correlated_exposure > self.max_portfolio_exposure:
            raise RiskConfigurationError(
                "correlated exposure cannot exceed portfolio exposure"
            )
        if self.soft_drawdown_limit >= self.hard_drawdown_limit:
            raise RiskConfigurationError(
                "soft_drawdown_limit must be below hard_drawdown_limit"
            )
        multipliers = (
            self.normal_volatility_multiplier,
            self.elevated_volatility_multiplier,
            self.extreme_volatility_multiplier,
            self.reduced_risk_multiplier,
        )
        if any(value < ZERO or value > ONE for value in multipliers):
            raise RiskConfigurationError("risk multipliers must be between 0 and 1")
        if not (
            self.normal_volatility_multiplier
            >= self.elevated_volatility_multiplier
            >= self.extreme_volatility_multiplier
        ):
            raise RiskConfigurationError(
                "volatility multipliers must never increase with volatility"
            )
        if self.quantity_step <= ZERO or not self.quantity_step.is_finite():
            raise RiskConfigurationError("quantity_step must be positive and finite")
        try:
            ZoneInfo(self.risk_day_timezone)
        except ZoneInfoNotFoundError as exc:
            raise RiskConfigurationError(
                f"unknown risk_day_timezone {self.risk_day_timezone!r}"
            ) from exc
        threshold_names = [item.timeframe for item in self.volatility_thresholds]
        if threshold_names != sorted(threshold_names) or len(threshold_names) != len(
            set(threshold_names)
        ):
            raise RiskConfigurationError(
                "volatility thresholds must use unique sorted timeframes"
            )

    def threshold_for(self, timeframe: str) -> VolatilityThreshold | None:
        return next(
            (item for item in self.volatility_thresholds if item.timeframe == timeframe),
            None,
        )

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        values: dict[str, str] = {
            "name": self.name.value,
            "enabled": str(self.enabled).lower(),
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "max_positions": str(self.max_positions),
            "max_portfolio_exposure": str(self.max_portfolio_exposure),
            "max_single_position_exposure": str(self.max_single_position_exposure),
            "max_group_exposure": str(self.max_group_exposure),
            "max_trade_risk_fraction": str(self.max_trade_risk_fraction),
            "daily_loss_limit": str(self.daily_loss_limit),
            "soft_drawdown_limit": str(self.soft_drawdown_limit),
            "hard_drawdown_limit": str(self.hard_drawdown_limit),
            "high_correlation_threshold": str(self.high_correlation_threshold),
            "max_highly_correlated_exposure": str(
                self.max_highly_correlated_exposure
            ),
            "correlation_min_observations": str(self.correlation_min_observations),
            "correlation_unknown_policy": self.correlation_unknown_policy.value,
            "unknown_group_policy": self.unknown_group_policy.value,
            "volatility_feature_name": self.volatility_feature_name,
            "missing_volatility_policy": self.missing_volatility_policy.value,
            "normal_volatility_multiplier": str(self.normal_volatility_multiplier),
            "elevated_volatility_multiplier": str(
                self.elevated_volatility_multiplier
            ),
            "extreme_volatility_multiplier": str(self.extreme_volatility_multiplier),
            "reduced_risk_multiplier": str(self.reduced_risk_multiplier),
            "risk_day_timezone": self.risk_day_timezone,
            "quantity_step": str(self.quantity_step),
        }
        for threshold in self.volatility_thresholds:
            values[f"volatility.{threshold.timeframe}.elevated"] = str(
                threshold.elevated
            )
            values[f"volatility.{threshold.timeframe}.extreme"] = str(
                threshold.extreme
            )
        return tuple(sorted(values.items()))


def inspect_risk_config(
    profile_name: str | TradingProfileName,
    *,
    risk_directory: Path = DEFAULT_RISK_DIRECTORY,
) -> BalancedRiskConfig:
    """Parse a risk schema without authorizing its use."""

    try:
        name = (
            profile_name
            if isinstance(profile_name, TradingProfileName)
            else TradingProfileName(profile_name.strip().lower())
        )
    except (AttributeError, ValueError) as exc:
        raise RiskConfigurationError(f"unknown risk profile {profile_name!r}") from exc
    path = risk_directory / f"{name.value}.toml"
    if not path.is_file():
        raise RiskConfigurationError(f"risk configuration not found: {path}")
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        thresholds_raw = raw["volatility_thresholds"]
        config = BalancedRiskConfig(
            name=TradingProfileName(str(raw["name"])),
            enabled=_boolean(raw["enabled"], "enabled"),
            engine_name=str(raw["engine_name"]),
            engine_version=str(raw["engine_version"]),
            max_positions=_integer(raw["max_positions"], "max_positions"),
            max_portfolio_exposure=_decimal(
                raw["max_portfolio_exposure"], "max_portfolio_exposure"
            ),
            max_single_position_exposure=_decimal(
                raw["max_single_position_exposure"],
                "max_single_position_exposure",
            ),
            max_group_exposure=_decimal(raw["max_group_exposure"], "max_group_exposure"),
            max_trade_risk_fraction=_decimal(
                raw["max_trade_risk_fraction"], "max_trade_risk_fraction"
            ),
            daily_loss_limit=_decimal(raw["daily_loss_limit"], "daily_loss_limit"),
            soft_drawdown_limit=_decimal(
                raw["soft_drawdown_limit"], "soft_drawdown_limit"
            ),
            hard_drawdown_limit=_decimal(
                raw["hard_drawdown_limit"], "hard_drawdown_limit"
            ),
            high_correlation_threshold=_decimal(
                raw["high_correlation_threshold"], "high_correlation_threshold"
            ),
            max_highly_correlated_exposure=_decimal(
                raw["max_highly_correlated_exposure"],
                "max_highly_correlated_exposure",
            ),
            correlation_min_observations=_integer(
                raw["correlation_min_observations"],
                "correlation_min_observations",
            ),
            correlation_unknown_policy=UnknownRiskPolicy(
                str(raw["correlation_unknown_policy"])
            ),
            unknown_group_policy=UnknownRiskPolicy(str(raw["unknown_group_policy"])),
            volatility_feature_name=str(raw["volatility_feature_name"]),
            missing_volatility_policy=UnknownRiskPolicy(
                str(raw["missing_volatility_policy"])
            ),
            normal_volatility_multiplier=_decimal(
                raw["normal_volatility_multiplier"],
                "normal_volatility_multiplier",
            ),
            elevated_volatility_multiplier=_decimal(
                raw["elevated_volatility_multiplier"],
                "elevated_volatility_multiplier",
            ),
            extreme_volatility_multiplier=_decimal(
                raw["extreme_volatility_multiplier"],
                "extreme_volatility_multiplier",
            ),
            reduced_risk_multiplier=_decimal(
                raw["reduced_risk_multiplier"], "reduced_risk_multiplier"
            ),
            risk_day_timezone=str(raw["risk_day_timezone"]),
            quantity_step=_decimal(raw["quantity_step"], "quantity_step"),
            volatility_thresholds=tuple(
                sorted(
                    (
                        VolatilityThreshold(
                            timeframe=str(timeframe),
                            elevated=_decimal(values["elevated"], "elevated"),
                            extreme=_decimal(values["extreme"], "extreme"),
                        )
                        for timeframe, values in thresholds_raw.items()
                    ),
                    key=lambda item: item.timeframe,
                )
            ),
        )
    except RiskConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RiskConfigurationError(f"invalid risk configuration {path}: {exc}") from exc
    if config.name is not name:
        raise RiskConfigurationError(
            f"risk configuration declares {config.name.value!r}, expected {name.value!r}"
        )
    return config


def load_asset_groups(path: Path = DEFAULT_ASSET_GROUPS_PATH) -> RiskAssetGroups:
    if not path.is_file():
        raise RiskConfigurationError(f"asset-group configuration not found: {path}")
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        groups = RiskAssetGroups(
            groups=tuple(
                sorted(
                    (
                        str(group),
                        tuple(sorted(str(symbol) for symbol in symbols)),
                    )
                    for group, symbols in raw["groups"].items()
                )
            )
        )
    except RiskConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RiskConfigurationError(f"invalid asset-group configuration {path}: {exc}") from exc
    return groups


def load_balanced_risk_config(
    profile: TradingProfile,
    *,
    risk_directory: Path = DEFAULT_RISK_DIRECTORY,
    asset_groups_path: Path = DEFAULT_ASSET_GROUPS_PATH,
) -> tuple[BalancedRiskConfig, RiskAssetGroups]:
    """Authorize only an enabled Balanced config bounded by its profile."""

    config = inspect_risk_config(profile.name, risk_directory=risk_directory)
    if profile.name is not TradingProfileName.BALANCED:
        raise RiskConfigurationError("aggressive risk remains unconditionally locked")
    if not profile.enabled or not config.enabled:
        raise RiskConfigurationError("Balanced profile and risk config must be enabled")
    if config.max_positions > profile.max_positions:
        raise RiskConfigurationError("risk max_positions exceeds the trading profile")
    if config.max_portfolio_exposure > Decimal(str(profile.max_exposure)):
        raise RiskConfigurationError(
            "risk portfolio exposure exceeds the trading profile"
        )
    if config.max_trade_risk_fraction > Decimal(str(profile.risk_budget)):
        raise RiskConfigurationError("risk trade budget exceeds the trading profile")
    groups = load_asset_groups(asset_groups_path)
    mapped = {symbol for symbol, _ in groups.symbol_mapping}
    missing = sorted(set(profile.asset_universe) - mapped)
    if missing:
        raise RiskConfigurationError(
            "profile symbols missing from asset-group configuration: " + ", ".join(missing)
        )
    return config, groups


def risk_config_hash(
    config: BalancedRiskConfig, groups: RiskAssetGroups
) -> str:
    payload = {
        "config": list(config.to_parameters()),
        "asset_groups": [
            [name, list(symbols)] for name, symbols in groups.groups
        ],
    }
    normalized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()
