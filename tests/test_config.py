from pathlib import Path

import pytest

from trading_ai.core.config import (
    inspect_profile,
    load_profile,
    load_runtime_settings,
    parse_environment,
)
from trading_ai.core.exceptions import LiveTradingLockedError, ProfileDisabledError
from trading_ai.core.models import ExecutionEnvironment, TradingProfileName


def test_balanced_profile_loads_and_is_enabled() -> None:
    profile = load_profile(TradingProfileName.BALANCED)

    assert profile.name is TradingProfileName.BALANCED
    assert profile.enabled is True
    assert profile.max_positions == 5
    assert profile.timeframes
    assert profile.asset_universe


def test_default_runtime_is_balanced_paper() -> None:
    settings = load_runtime_settings(environ={})

    assert settings.environment is ExecutionEnvironment.PAPER
    assert settings.profile.name is TradingProfileName.BALANCED
    assert settings.live_allowed is False


def test_aggressive_profile_is_present_but_refused() -> None:
    inspected = inspect_profile(TradingProfileName.AGGRESSIVE)

    assert inspected.enabled is False
    with pytest.raises(ProfileDisabledError, match="unconditionally locked"):
        load_runtime_settings("PAPER", "aggressive")


def test_aggressive_cannot_be_enabled_by_editing_toml(tmp_path: Path) -> None:
    source = Path("config/profiles/aggressive.toml").read_text(encoding="utf-8")
    (tmp_path / "aggressive.toml").write_text(
        source.replace("enabled = false", "enabled = true"), encoding="utf-8"
    )

    with pytest.raises(ProfileDisabledError, match="unconditionally locked"):
        load_profile("aggressive", tmp_path)


def test_live_startup_is_always_refused() -> None:
    with pytest.raises(LiveTradingLockedError, match="no configuration override"):
        load_runtime_settings("LIVE", "balanced")


def test_environments_parse_to_distinct_values() -> None:
    parsed = {parse_environment(value) for value in ("DEV", "TEST", "PAPER", "LIVE")}

    assert parsed == set(ExecutionEnvironment)
