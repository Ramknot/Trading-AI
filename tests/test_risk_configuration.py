"""Offline configuration and schema-boundary tests for Balanced risk."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from trading_ai.core.config import inspect_profile, load_runtime_settings
from trading_ai.core.models import TradingProfileName
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.config import (
    DEFAULT_ASSET_GROUPS_PATH,
    DEFAULT_RISK_DIRECTORY,
    BalancedRiskConfig,
    inspect_risk_config,
    load_asset_groups,
    load_balanced_risk_config,
    risk_config_hash,
)
from trading_ai.risk.exceptions import RiskConfigurationError


def test_balanced_risk_config_loads_research_defaults() -> None:
    profile = load_runtime_settings().profile
    config, groups = load_balanced_risk_config(profile)

    assert config.name is TradingProfileName.BALANCED
    assert config.enabled is True
    assert (config.engine_name, config.engine_version) == ("balanced-risk", "1.0")
    assert config.max_positions == 5
    assert config.max_portfolio_exposure == Decimal("0.6")
    assert config.max_single_position_exposure == Decimal("0.15")
    assert config.max_trade_risk_fraction == Decimal("0.01")
    assert {symbol for symbol, _ in groups.symbol_mapping} == set(
        profile.asset_universe
    )


def test_aggressive_risk_schema_exists_but_is_disabled_and_not_loadable() -> None:
    aggressive_profile = inspect_profile("aggressive")
    config = inspect_risk_config("aggressive")

    assert config.enabled is False
    assert config.name is TradingProfileName.AGGRESSIVE
    with pytest.raises(RiskConfigurationError, match="aggressive"):
        load_balanced_risk_config(aggressive_profile)


def test_risk_config_hash_is_stable_and_covers_asset_groups() -> None:
    config = inspect_risk_config("balanced")
    groups = load_asset_groups()
    first = risk_config_hash(config, groups)
    second = risk_config_hash(config, groups)

    assert first == second
    assert len(first) == 64
    renamed = replace(
        groups,
        groups=tuple(
            sorted(
                ("RENAMED" if name == "US_TECH" else name, symbols)
                for name, symbols in groups.groups
            )
        ),
    )
    assert risk_config_hash(config, renamed) != first


def test_risk_config_is_immutable() -> None:
    config = inspect_risk_config("balanced")
    with pytest.raises(FrozenInstanceError):
        config.max_positions = 99  # type: ignore[misc]


def test_profile_bound_violation_fails_closed_during_load(tmp_path) -> None:
    profile = load_runtime_settings().profile
    source = (DEFAULT_RISK_DIRECTORY / "balanced.toml").read_text(encoding="utf-8")
    (tmp_path / "balanced.toml").write_text(
        source.replace(
            "max_portfolio_exposure = 0.60",
            "max_portfolio_exposure = 0.61",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RiskConfigurationError, match="exceeds"):
        load_balanced_risk_config(
            profile,
            risk_directory=tmp_path,
            asset_groups_path=DEFAULT_ASSET_GROUPS_PATH,
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "max_single_position_exposure = 0.15",
            "max_single_position_exposure = -0.15",
            "max_single_position_exposure",
        ),
        (
            "soft_drawdown_limit = 0.05",
            "soft_drawdown_limit = 0.11",
            "soft_drawdown_limit",
        ),
        (
            "reduced_risk_multiplier = 0.50",
            "reduced_risk_multiplier = 1.10",
            "multipliers",
        ),
        (
            "enabled = true",
            'enabled = "true"',
            "TOML boolean",
        ),
    ],
)
def test_negative_or_incoherent_risk_values_are_rejected(
    tmp_path, old: str, new: str, message: str
) -> None:
    source = (DEFAULT_RISK_DIRECTORY / "balanced.toml").read_text(encoding="utf-8")
    (tmp_path / "balanced.toml").write_text(
        source.replace(old, new), encoding="utf-8"
    )

    with pytest.raises(RiskConfigurationError, match=message):
        inspect_risk_config("balanced", risk_directory=tmp_path)


def test_direct_engine_construction_cannot_bypass_profile_limits() -> None:
    profile = load_runtime_settings().profile
    config, groups = load_balanced_risk_config(profile)
    permissive = replace(config, max_portfolio_exposure=Decimal("0.61"))

    with pytest.raises(ValueError, match="exposure exceeds"):
        BalancedRiskEngine(profile, permissive, groups)


def test_asset_group_mapping_is_configuration_driven() -> None:
    source = DEFAULT_ASSET_GROUPS_PATH.read_text(encoding="utf-8")
    risk_python = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (DEFAULT_RISK_DIRECTORY.parent.parent / "src" / "trading_ai" / "risk").glob("*.py")
    )

    assert "AAPL" in source
    assert '"AAPL"' not in risk_python and "'AAPL'" not in risk_python
