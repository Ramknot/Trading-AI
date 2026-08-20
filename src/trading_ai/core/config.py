"""TOML profile loading and fail-closed runtime validation."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trading_ai.core.environment import policy_for
from trading_ai.core.exceptions import (
    ConfigurationError,
    LiveTradingLockedError,
    ProfileDisabledError,
)
from trading_ai.core.models import (
    ExecutionEnvironment,
    TradingContext,
    TradingProfile,
    TradingProfileName,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_DIRECTORY = PROJECT_ROOT / "config" / "profiles"


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Validated startup settings. Construction succeeds only when safe."""

    environment: ExecutionEnvironment
    profile: TradingProfile

    @property
    def context(self) -> TradingContext:
        return TradingContext(self.environment, self.profile.name)

    @property
    def live_allowed(self) -> bool:
        return policy_for(self.environment).external_order_transmission_allowed


def parse_environment(value: str | ExecutionEnvironment) -> ExecutionEnvironment:
    if isinstance(value, ExecutionEnvironment):
        return value
    try:
        return ExecutionEnvironment(value.strip().upper())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in ExecutionEnvironment)
        raise ConfigurationError(f"unknown environment {value!r}; expected {choices}") from exc


def parse_profile_name(value: str | TradingProfileName) -> TradingProfileName:
    if isinstance(value, TradingProfileName):
        return value
    try:
        return TradingProfileName(value.strip().lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in TradingProfileName)
        raise ConfigurationError(f"unknown profile {value!r}; expected {choices}") from exc


def inspect_profile(
    profile_name: str | TradingProfileName,
    profile_directory: Path = DEFAULT_PROFILE_DIRECTORY,
) -> TradingProfile:
    """Load and validate a profile schema without authorizing activation."""

    name = parse_profile_name(profile_name)
    path = profile_directory / f"{name.value}.toml"
    if not path.is_file():
        raise ConfigurationError(f"profile file not found: {path}")
    try:
        with path.open("rb") as profile_file:
            raw = tomllib.load(profile_file)
        profile = TradingProfile(
            name=TradingProfileName(str(raw["name"])),
            enabled=bool(raw["enabled"]),
            timeframes=tuple(str(value) for value in raw["timeframes"]),
            asset_universe=tuple(str(value) for value in raw["asset_universe"]),
            max_positions=int(raw["max_positions"]),
            max_exposure=float(raw["max_exposure"]),
            max_turnover=float(raw["max_turnover"]),
            allow_short=bool(raw["allow_short"]),
            risk_budget=float(raw["risk_budget"]),
            signal_threshold=float(raw["signal_threshold"]),
        )
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"invalid profile file {path}: {exc}") from exc
    if profile.name is not name:
        raise ConfigurationError(
            f"profile file {path} declares {profile.name.value!r}, expected {name.value!r}"
        )
    return profile


def load_profile(
    profile_name: str | TradingProfileName,
    profile_directory: Path = DEFAULT_PROFILE_DIRECTORY,
) -> TradingProfile:
    """Load an activatable profile, rejecting all Lot 0 locked profiles."""

    profile = inspect_profile(profile_name, profile_directory)
    if profile.name is TradingProfileName.AGGRESSIVE:
        raise ProfileDisabledError(
            "aggressive is architecture-only and unconditionally locked in Lot 0"
        )
    if not profile.enabled:
        raise ProfileDisabledError(f"profile {profile.name.value!r} is disabled")
    return profile


def load_runtime_settings(
    environment: str | ExecutionEnvironment | None = None,
    profile_name: str | TradingProfileName | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    profile_directory: Path = DEFAULT_PROFILE_DIRECTORY,
) -> RuntimeSettings:
    """Load safe runtime settings from arguments or environment variables.

    Lot 0 has no unlock input for LIVE: requesting it always fails closed.
    """

    values = os.environ if environ is None else environ
    selected_environment = parse_environment(
        environment or values.get("TRADING_AI_ENV", ExecutionEnvironment.PAPER.value)
    )
    selected_profile = profile_name or values.get(
        "TRADING_AI_PROFILE", TradingProfileName.BALANCED.value
    )
    if selected_environment is ExecutionEnvironment.LIVE:
        raise LiveTradingLockedError(
            "LIVE startup is locked in Lot 0 and has no configuration override"
        )
    profile = load_profile(selected_profile, profile_directory)
    if not policy_for(selected_environment).startup_allowed:
        raise ConfigurationError(
            f"startup is not allowed in {selected_environment.value}"
        )
    return RuntimeSettings(environment=selected_environment, profile=profile)
