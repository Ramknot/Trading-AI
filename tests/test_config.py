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


BALANCED_V1_UNIVERSE = (
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "ASML",
    "SAP",
    "MC.PA",
    "AIR.PA",
)


def test_balanced_profile_loads_and_is_enabled() -> None:
    profile = load_profile(TradingProfileName.BALANCED)

    assert profile.name is TradingProfileName.BALANCED
    assert profile.enabled is True
    assert profile.max_positions == 5
    assert profile.timeframes == ("1h", "4h", "1d")
    assert profile.asset_universe == BALANCED_V1_UNIVERSE
    assert profile.allow_short is False
    assert "BTC-USD" not in profile.asset_universe
    assert "ETH-USD" not in profile.asset_universe


def test_default_runtime_is_balanced_paper() -> None:
    settings = load_runtime_settings(environ={})

    assert settings.environment is ExecutionEnvironment.PAPER
    assert settings.profile.name is TradingProfileName.BALANCED
    assert settings.live_allowed is False


def test_aggressive_profile_is_present_but_refused() -> None:
    inspected = inspect_profile(TradingProfileName.AGGRESSIVE)

    assert inspected.enabled is False
    assert not any("BTC" in symbol or "ETH" in symbol for symbol in inspected.asset_universe)
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


def test_market_universes_are_not_hard_coded_in_core_components() -> None:
    profiles = (inspect_profile("balanced"), inspect_profile("aggressive"))
    configured_symbols = {
        symbol for profile in profiles for symbol in profile.asset_universe
    }
    component_roots = (
        Path("src/trading_ai/data"),
        Path("src/trading_ai/strategies"),
        Path("src/trading_ai/backtesting"),
        Path("src/trading_ai/portfolio"),
        Path("src/trading_ai/risk"),
    )

    for component_root in component_roots:
        for path in component_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for symbol in configured_symbols:
                assert f'"{symbol}"' not in source
                assert f"'{symbol}'" not in source
