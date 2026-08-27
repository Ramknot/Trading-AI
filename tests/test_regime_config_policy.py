from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from decimal import Decimal

import pytest

from backtest_support import START
from trading_ai.backtesting.models import StrategySignal, StrategySignalAction
from trading_ai.core.config import inspect_profile, load_runtime_settings
from trading_ai.regimes import (
    ActivationStatus,
    BalancedStrategyActivationPolicy,
    RegimeConfigurationError,
    RegimeSnapshot,
    StructureRegime,
    VolatilityRegime,
    inspect_regime_config,
    inspect_strategy_policy_config,
    load_balanced_regime_config,
    load_balanced_strategy_policy_config,
    regime_config_hash,
    strategy_policy_config_hash,
)


PROFILE = load_runtime_settings().profile


def _snapshot(
    structure: StructureRegime,
    volatility: VolatilityRegime = VolatilityRegime.NORMAL,
) -> RegimeSnapshot:
    return RegimeSnapshot(
        snapshot_id=f"snapshot-{structure.value}-{volatility.value}",
        symbol="AAPL",
        timestamp=START,
        timeframe="1d",
        structure_regime=structure,
        volatility_regime=volatility,
        detector_name="test-detector",
        detector_version="1",
        config_hash="a" * 64,
        bars_in_current_structure_regime=1,
        evidence=(),
        reason_codes=("TEST",),
    )


def _signal(strategy: str, action: StrategySignalAction) -> StrategySignal:
    return StrategySignal(
        signal_id=f"signal-{strategy}-{action.value}",
        strategy_name=strategy,
        strategy_version="1.0",
        symbol="AAPL",
        timeframe="1d",
        timestamp=START,
        action=action,
        strength=1.0,
        reason="test signal",
        features_used=(),
    )


def test_balanced_regime_and_policy_configs_load_with_stable_hashes() -> None:
    regime = load_balanced_regime_config(PROFILE)
    policy = load_balanced_strategy_policy_config(PROFILE)

    assert regime.enabled is True
    assert regime.detector_name == "balanced-regime"
    assert policy.policy_name == "balanced-strategy-policy"
    assert regime_config_hash(regime) == regime_config_hash(regime)
    assert strategy_policy_config_hash(policy) == strategy_policy_config_hash(policy)
    assert len(regime_config_hash(regime)) == 64
    assert len(strategy_policy_config_hash(policy)) == 64


def test_aggressive_regime_schema_exists_but_cannot_be_activated() -> None:
    profile = inspect_profile("aggressive")
    assert inspect_regime_config("aggressive").enabled is False
    assert inspect_strategy_policy_config("aggressive").enabled is False
    with pytest.raises(RegimeConfigurationError, match="aggressive"):
        load_balanced_regime_config(profile)
    with pytest.raises(RegimeConfigurationError, match="aggressive"):
        load_balanced_strategy_policy_config(profile)


def test_regime_config_is_immutable_and_rejects_incoherent_thresholds() -> None:
    config = load_balanced_regime_config(PROFILE)
    with pytest.raises(FrozenInstanceError):
        config.confirmation_bars = 1  # type: ignore[misc]
    with pytest.raises(RegimeConfigurationError, match="range < trend"):
        replace(
            config,
            range_efficiency_threshold=Decimal("0.6"),
            trend_efficiency_threshold=Decimal("0.5"),
        )
    with pytest.raises(RegimeConfigurationError, match="fast_ema_window"):
        replace(config, fast_ema_window=config.slow_ema_window)


@pytest.mark.parametrize(
    ("structure", "strategy", "expected", "multiplier"),
    (
        (StructureRegime.TREND_UP, "trend", ActivationStatus.ALLOW, "1.0"),
        (StructureRegime.TREND_UP, "momentum", ActivationStatus.ALLOW, "1.0"),
        (StructureRegime.TREND_UP, "breakout", ActivationStatus.ALLOW, "1.0"),
        (StructureRegime.TREND_UP, "mean-reversion", ActivationStatus.BLOCK, "0.0"),
        (StructureRegime.RANGE, "trend", ActivationStatus.BLOCK, "0.0"),
        (StructureRegime.RANGE, "momentum", ActivationStatus.BLOCK, "0.0"),
        (StructureRegime.RANGE, "breakout", ActivationStatus.REDUCE, "0.5"),
        (StructureRegime.RANGE, "mean-reversion", ActivationStatus.ALLOW, "1.0"),
        (StructureRegime.TREND_DOWN, "trend", ActivationStatus.BLOCK, "0.0"),
        (StructureRegime.UNKNOWN, "breakout", ActivationStatus.BLOCK, "0.0"),
    ),
)
def test_balanced_strategy_activation_matrix(
    structure: StructureRegime,
    strategy: str,
    expected: ActivationStatus,
    multiplier: str,
) -> None:
    policy = BalancedStrategyActivationPolicy.from_profile(PROFILE)
    decision = policy.evaluate(
        strategy_name=strategy,
        strategy_version="1.0",
        signal=_signal(strategy, StrategySignalAction.ENTER_LONG),
        regime=_snapshot(structure),
        proposed_quantity=Decimal("10"),
    )

    assert decision.status is expected
    assert decision.allocation_multiplier == Decimal(multiplier)
    assert decision.adjusted_quantity <= decision.proposed_quantity


def test_high_volatility_blocks_mean_reversion_in_range() -> None:
    policy = BalancedStrategyActivationPolicy.from_profile(PROFILE)
    decision = policy.evaluate(
        strategy_name="mean-reversion",
        strategy_version="1.0",
        signal=_signal("mean-reversion", StrategySignalAction.ENTER_LONG),
        regime=_snapshot(StructureRegime.RANGE, VolatilityRegime.HIGH),
        proposed_quantity=Decimal("10"),
    )
    assert decision.status is ActivationStatus.BLOCK
    assert "VOLATILITY_HIGH_BLOCK" in decision.reason_codes


@pytest.mark.parametrize("structure", tuple(StructureRegime))
@pytest.mark.parametrize("volatility", tuple(VolatilityRegime))
def test_exit_long_is_always_allowed_by_policy(
    structure: StructureRegime, volatility: VolatilityRegime
) -> None:
    policy = BalancedStrategyActivationPolicy.from_profile(PROFILE)
    decision = policy.evaluate(
        strategy_name="mean-reversion",
        strategy_version="1.0",
        signal=_signal("mean-reversion", StrategySignalAction.EXIT_LONG),
        regime=_snapshot(structure, volatility),
        proposed_quantity=Decimal("3"),
    )
    assert decision.status is ActivationStatus.ALLOW
    assert decision.allocation_multiplier == Decimal("1")
    assert decision.adjusted_quantity == Decimal("3")


def test_activation_decision_is_immutable_and_timestamp_aware() -> None:
    policy = BalancedStrategyActivationPolicy.from_profile(PROFILE)
    decision = policy.evaluate(
        strategy_name="trend",
        strategy_version="1.0",
        signal=_signal("trend", StrategySignalAction.ENTER_LONG),
        regime=_snapshot(StructureRegime.TREND_UP),
        proposed_quantity=Decimal("4"),
    )
    with pytest.raises(FrozenInstanceError):
        decision.allocation_multiplier = Decimal("2")  # type: ignore[misc]
    assert decision.timestamp == START
    assert decision.timestamp + timedelta(days=1) > decision.timestamp
