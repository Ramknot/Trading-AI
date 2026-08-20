"""Diagnostic health checks used by the CLI and monitoring adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trading_ai.core.config import (
    DEFAULT_PROFILE_DIRECTORY,
    inspect_profile,
    load_runtime_settings,
    parse_environment,
    parse_profile_name,
)
from trading_ai.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class HealthReport:
    environment: str
    profile: str
    profile_enabled: bool
    live_allowed: bool
    configuration_valid: bool
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def doctor(
    environment: str,
    profile_name: str,
    *,
    profile_directory: Path = DEFAULT_PROFILE_DIRECTORY,
) -> HealthReport:
    """Return a deterministic health report without leaking configuration."""

    environment_label = str(environment).upper()
    profile_label = str(profile_name).lower()
    profile_enabled = False
    try:
        parsed_environment = parse_environment(environment)
        parsed_profile = parse_profile_name(profile_name)
        inspected = inspect_profile(parsed_profile, profile_directory)
        profile_enabled = inspected.enabled
        settings = load_runtime_settings(
            parsed_environment,
            parsed_profile,
            profile_directory=profile_directory,
        )
    except ConfigurationError as exc:
        return HealthReport(
            environment=environment_label,
            profile=profile_label,
            profile_enabled=profile_enabled,
            live_allowed=False,
            configuration_valid=False,
            status="BLOCKED",
            message=str(exc),
        )
    return HealthReport(
        environment=settings.environment.value,
        profile=settings.profile.name.value,
        profile_enabled=settings.profile.enabled,
        live_allowed=settings.live_allowed,
        configuration_valid=True,
        status="OK",
        message="configuration is valid; DenyAllRiskEngine remains active",
    )
