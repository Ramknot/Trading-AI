"""Non-optimized research-validation thresholds loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trading_ai.core.hashing import stable_hash
from trading_ai.core.config import PROJECT_ROOT
from trading_ai.validation.exceptions import ValidationError


DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "config" / "validation" / "balanced.toml"


def _bool(raw: object, name: str) -> bool:
    if type(raw) is not bool:
        raise ValidationError(f"{name} must be a TOML boolean")
    return raw


def _decimal(raw: object, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric") from exc
    if not value.is_finite():
        raise ValidationError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    name: str
    version: str
    enabled: bool
    minimum_closed_trades: int
    minimum_net_return: Decimal
    minimum_net_expectancy: Decimal
    minimum_profit_factor: Decimal
    maximum_drawdown: Decimal
    minimum_subperiods: int
    symbol_concentration_warning_fraction: Decimal
    cost_stress_multipliers: tuple[Decimal, ...]
    require_final_oos: bool
    require_complete_variable_costs: bool
    require_verified_tariff_for_period: bool
    require_operating_costs_for_pass: bool

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValidationError("validation name and version must not be empty")
        if not self.enabled:
            raise ValidationError("Balanced research validation must be enabled")
        if self.minimum_closed_trades < 1 or self.minimum_subperiods < 2:
            raise ValidationError("validation sample thresholds must be positive")
        if self.minimum_profit_factor < 0:
            raise ValidationError("minimum_profit_factor must be non-negative")
        if not Decimal("0") <= self.maximum_drawdown <= Decimal("1"):
            raise ValidationError("maximum_drawdown must be in [0, 1]")
        if not Decimal("0") < self.symbol_concentration_warning_fraction <= Decimal("1"):
            raise ValidationError("symbol concentration threshold must be in (0, 1]")
        if tuple(sorted(set(self.cost_stress_multipliers))) != self.cost_stress_multipliers:
            raise ValidationError("cost stress multipliers must be sorted and unique")
        if not self.cost_stress_multipliers or self.cost_stress_multipliers[0] != Decimal("1"):
            raise ValidationError("cost stress must include baseline 1.0")
        if any(item <= 0 for item in self.cost_stress_multipliers):
            raise ValidationError("cost stress multipliers must be positive")


def load_validation_config(path: Path = DEFAULT_VALIDATION_PATH) -> tuple[ValidationConfig, str]:
    if not path.is_file():
        raise ValidationError(f"validation configuration not found: {path}")
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        config = ValidationConfig(
            name=str(raw["name"]),
            version=str(raw["version"]),
            enabled=_bool(raw["enabled"], "enabled"),
            minimum_closed_trades=int(raw["minimum_closed_trades"]),
            minimum_net_return=_decimal(raw["minimum_net_return"], "minimum_net_return"),
            minimum_net_expectancy=_decimal(raw["minimum_net_expectancy"], "minimum_net_expectancy"),
            minimum_profit_factor=_decimal(raw["minimum_profit_factor"], "minimum_profit_factor"),
            maximum_drawdown=_decimal(raw["maximum_drawdown"], "maximum_drawdown"),
            minimum_subperiods=int(raw["minimum_subperiods"]),
            symbol_concentration_warning_fraction=_decimal(
                raw["symbol_concentration_warning_fraction"],
                "symbol_concentration_warning_fraction",
            ),
            cost_stress_multipliers=tuple(
                sorted(
                    _decimal(item, "cost_stress_multipliers")
                    for item in raw["cost_stress_multipliers"]
                )
            ),
            require_final_oos=_bool(raw["require_final_oos"], "require_final_oos"),
            require_complete_variable_costs=_bool(
                raw["require_complete_variable_costs"],
                "require_complete_variable_costs",
            ),
            require_verified_tariff_for_period=_bool(
                raw["require_verified_tariff_for_period"],
                "require_verified_tariff_for_period",
            ),
            require_operating_costs_for_pass=_bool(
                raw["require_operating_costs_for_pass"],
                "require_operating_costs_for_pass",
            ),
        )
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError(f"invalid validation configuration: {exc}") from exc
    return config, stable_hash(raw)
