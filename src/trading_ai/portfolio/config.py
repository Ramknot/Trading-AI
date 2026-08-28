"""Configuration-only Balanced portfolio construction with hard-limit checks."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.models import TradingProfile, TradingProfileName
from trading_ai.portfolio.exceptions import PortfolioConfigurationError
from trading_ai.portfolio.models import (
    MixedCurrencyPolicy,
    StrategySleeve,
    UnknownCorrelationPolicy,
)
from trading_ai.risk.config import BalancedRiskConfig


DEFAULT_PORTFOLIO_DIRECTORY = PROJECT_ROOT / "config" / "portfolio"
DEFAULT_ASSET_CURRENCIES_PATH = DEFAULT_PORTFOLIO_DIRECTORY / "asset_currencies.toml"
ZERO = Decimal("0")
ONE = Decimal("1")
BALANCED_SLEEVE_NAMES = frozenset(
    {"trend", "momentum", "breakout", "mean-reversion"}
)


def _decimal(raw: object, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise PortfolioConfigurationError(f"{name} must be numeric") from exc
    if not value.is_finite():
        raise PortfolioConfigurationError(f"{name} must be finite")
    return value


def _integer(raw: object, name: str) -> int:
    if type(raw) is not int:
        raise PortfolioConfigurationError(f"{name} must be a TOML integer")
    return raw


def _boolean(raw: object, name: str) -> bool:
    if type(raw) is not bool:
        raise PortfolioConfigurationError(f"{name} must be a TOML boolean")
    return raw


@dataclass(frozen=True, slots=True)
class AssetCurrencyMap:
    """Configuration-driven quote currencies; unknown is never assumed USD."""

    currencies: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        symbols = [symbol for symbol, _ in self.currencies]
        if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
            raise PortfolioConfigurationError(
                "asset currencies must use unique sorted symbols"
            )
        for symbol, currency in self.currencies:
            if not symbol.strip() or len(currency.strip()) != 3:
                raise PortfolioConfigurationError(
                    "asset currencies require non-empty symbols and ISO-like codes"
                )

    def currency_for(self, symbol: str) -> str | None:
        return next(
            (currency for item, currency in self.currencies if item == symbol),
            None,
        )


@dataclass(frozen=True, slots=True)
class BalancedPortfolioConfig:
    name: TradingProfileName
    enabled: bool
    engine_name: str
    engine_version: str
    base_currency: str
    max_target_exposure: Decimal
    max_target_per_symbol: Decimal
    max_unique_positions: int
    min_cash_fraction: Decimal
    min_rebalance_weight: Decimal
    max_entry_turnover_per_cycle: Decimal
    soft_correlation_threshold: Decimal
    correlation_min_observations: int
    unknown_correlation_policy: UnknownCorrelationPolicy
    mixed_currency_policy: MixedCurrencyPolicy
    quantity_step: Decimal
    strategy_sleeves: tuple[StrategySleeve, ...]

    def __post_init__(self) -> None:
        if not self.engine_name.strip() or not self.engine_version.strip():
            raise PortfolioConfigurationError("portfolio engine identity is required")
        if len(self.base_currency.strip()) != 3:
            raise PortfolioConfigurationError("base_currency must be an ISO-like code")
        for field_name in (
            "max_target_exposure",
            "max_target_per_symbol",
            "min_rebalance_weight",
            "max_entry_turnover_per_cycle",
            "soft_correlation_threshold",
        ):
            value = getattr(self, field_name)
            if value <= ZERO or value > ONE:
                raise PortfolioConfigurationError(f"{field_name} must be in (0, 1]")
        if self.min_cash_fraction < ZERO or self.min_cash_fraction > ONE:
            raise PortfolioConfigurationError("min_cash_fraction must be in [0, 1]")
        if self.max_target_exposure > ONE - self.min_cash_fraction:
            raise PortfolioConfigurationError(
                "max_target_exposure violates the configured cash floor"
            )
        if self.max_target_per_symbol > self.max_target_exposure:
            raise PortfolioConfigurationError(
                "per-symbol target cannot exceed total target exposure"
            )
        if self.max_unique_positions < 1:
            raise PortfolioConfigurationError("max_unique_positions must be positive")
        if self.correlation_min_observations < 2:
            raise PortfolioConfigurationError(
                "correlation_min_observations must be at least two"
            )
        if self.quantity_step <= ZERO or not self.quantity_step.is_finite():
            raise PortfolioConfigurationError("quantity_step must be positive")
        names = [item.strategy_name for item in self.strategy_sleeves]
        if names != sorted(names) or len(names) != len(set(names)):
            raise PortfolioConfigurationError("strategy sleeves must be unique and sorted")
        if set(names) != BALANCED_SLEEVE_NAMES:
            raise PortfolioConfigurationError(
                "portfolio config must define all four Balanced strategy sleeves"
            )
        total = sum((item.budget_weight for item in self.strategy_sleeves), ZERO)
        if total > self.max_target_exposure:
            raise PortfolioConfigurationError(
                "strategy sleeve budgets exceed max_target_exposure"
            )

    @property
    def sleeve_budget_total(self) -> Decimal:
        return sum((item.budget_weight for item in self.strategy_sleeves), ZERO)

    def sleeve_for(self, strategy_name: str) -> StrategySleeve | None:
        return next(
            (item for item in self.strategy_sleeves if item.strategy_name == strategy_name),
            None,
        )

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        values = {
            "name": self.name.value,
            "enabled": str(self.enabled).lower(),
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "base_currency": self.base_currency,
            "max_target_exposure": str(self.max_target_exposure),
            "max_target_per_symbol": str(self.max_target_per_symbol),
            "max_unique_positions": str(self.max_unique_positions),
            "min_cash_fraction": str(self.min_cash_fraction),
            "min_rebalance_weight": str(self.min_rebalance_weight),
            "max_entry_turnover_per_cycle": str(self.max_entry_turnover_per_cycle),
            "soft_correlation_threshold": str(self.soft_correlation_threshold),
            "correlation_min_observations": str(self.correlation_min_observations),
            "unknown_correlation_policy": self.unknown_correlation_policy.value,
            "mixed_currency_policy": self.mixed_currency_policy.value,
            "quantity_step": str(self.quantity_step),
        }
        values.update(
            {
                f"sleeve.{item.strategy_name}": str(item.budget_weight)
                for item in self.strategy_sleeves
            }
        )
        return tuple(sorted(values.items()))


def inspect_portfolio_config(
    profile_name: str | TradingProfileName,
    *,
    portfolio_directory: Path = DEFAULT_PORTFOLIO_DIRECTORY,
) -> BalancedPortfolioConfig:
    try:
        name = (
            profile_name
            if isinstance(profile_name, TradingProfileName)
            else TradingProfileName(profile_name.strip().lower())
        )
    except (AttributeError, ValueError) as exc:
        raise PortfolioConfigurationError(
            f"unknown portfolio profile {profile_name!r}"
        ) from exc
    path = portfolio_directory / f"{name.value}.toml"
    if not path.is_file():
        raise PortfolioConfigurationError(f"portfolio configuration not found: {path}")
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        config = BalancedPortfolioConfig(
            name=TradingProfileName(str(raw["name"])),
            enabled=_boolean(raw["enabled"], "enabled"),
            engine_name=str(raw["engine_name"]),
            engine_version=str(raw["engine_version"]),
            base_currency=str(raw["base_currency"]).upper(),
            max_target_exposure=_decimal(raw["max_target_exposure"], "max_target_exposure"),
            max_target_per_symbol=_decimal(raw["max_target_per_symbol"], "max_target_per_symbol"),
            max_unique_positions=_integer(raw["max_unique_positions"], "max_unique_positions"),
            min_cash_fraction=_decimal(raw["min_cash_fraction"], "min_cash_fraction"),
            min_rebalance_weight=_decimal(raw["min_rebalance_weight"], "min_rebalance_weight"),
            max_entry_turnover_per_cycle=_decimal(
                raw["max_entry_turnover_per_cycle"], "max_entry_turnover_per_cycle"
            ),
            soft_correlation_threshold=_decimal(
                raw["soft_correlation_threshold"], "soft_correlation_threshold"
            ),
            correlation_min_observations=_integer(
                raw["correlation_min_observations"], "correlation_min_observations"
            ),
            unknown_correlation_policy=UnknownCorrelationPolicy(
                str(raw["unknown_correlation_policy"])
            ),
            mixed_currency_policy=MixedCurrencyPolicy(
                str(raw["mixed_currency_policy"])
            ),
            quantity_step=_decimal(raw["quantity_step"], "quantity_step"),
            strategy_sleeves=tuple(
                sorted(
                    (
                        StrategySleeve(str(strategy), _decimal(weight, f"sleeve.{strategy}"))
                        for strategy, weight in raw["strategy_sleeves"].items()
                    ),
                    key=lambda item: item.strategy_name,
                )
            ),
        )
    except PortfolioConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise PortfolioConfigurationError(
            f"invalid portfolio configuration {path}: {exc}"
        ) from exc
    if config.name is not name:
        raise PortfolioConfigurationError(
            f"portfolio config declares {config.name.value!r}, expected {name.value!r}"
        )
    return config


def load_asset_currencies(
    path: Path = DEFAULT_ASSET_CURRENCIES_PATH,
) -> AssetCurrencyMap:
    if not path.is_file():
        raise PortfolioConfigurationError(f"asset currency config not found: {path}")
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        return AssetCurrencyMap(
            tuple(
                sorted(
                    (str(symbol), str(currency).upper())
                    for symbol, currency in raw["currencies"].items()
                )
            )
        )
    except PortfolioConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise PortfolioConfigurationError(
            f"invalid asset currency configuration {path}: {exc}"
        ) from exc


def load_balanced_portfolio_config(
    profile: TradingProfile,
    risk_config: BalancedRiskConfig,
    *,
    portfolio_directory: Path = DEFAULT_PORTFOLIO_DIRECTORY,
    asset_currencies_path: Path = DEFAULT_ASSET_CURRENCIES_PATH,
) -> tuple[BalancedPortfolioConfig, AssetCurrencyMap]:
    """Authorize only enabled Balanced allocation within profile/risk ceilings."""

    config = inspect_portfolio_config(
        profile.name, portfolio_directory=portfolio_directory
    )
    if profile.name is not TradingProfileName.BALANCED:
        raise PortfolioConfigurationError("aggressive portfolio remains locked")
    if not profile.enabled or not config.enabled:
        raise PortfolioConfigurationError("Balanced profile/portfolio must be enabled")
    profile_exposure = Decimal(str(profile.max_exposure))
    hard_exposure = min(profile_exposure, risk_config.max_portfolio_exposure)
    if config.max_target_exposure > hard_exposure:
        raise PortfolioConfigurationError("portfolio exposure exceeds profile/risk hard limit")
    if config.max_target_per_symbol > risk_config.max_single_position_exposure:
        raise PortfolioConfigurationError("portfolio symbol target exceeds Risk hard limit")
    if config.max_unique_positions > min(profile.max_positions, risk_config.max_positions):
        raise PortfolioConfigurationError("portfolio positions exceed profile/risk hard limit")
    currencies = load_asset_currencies(asset_currencies_path)
    missing = sorted(set(profile.asset_universe) - {item for item, _ in currencies.currencies})
    if missing:
        raise PortfolioConfigurationError(
            "profile symbols missing currency metadata: " + ", ".join(missing)
        )
    return config, currencies


def portfolio_config_hash(
    config: BalancedPortfolioConfig,
    currencies: AssetCurrencyMap,
) -> str:
    payload = {
        "config": list(config.to_parameters()),
        "asset_currencies": [list(item) for item in currencies.currencies],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
