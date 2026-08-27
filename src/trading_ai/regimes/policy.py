"""Configuration-driven Balanced strategy activation policy."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from trading_ai.backtesting.models import StrategySignal, StrategySignalAction
from trading_ai.core.models import TradingProfile
from trading_ai.regimes.base import ActivationPolicy
from trading_ai.regimes.config import (
    BalancedStrategyPolicyConfig,
    load_balanced_strategy_policy_config,
    strategy_policy_config_hash,
)
from trading_ai.regimes.exceptions import RegimeInputError
from trading_ai.regimes.models import (
    ActivationDecision,
    ActivationStatus,
    RegimeSnapshot,
    VolatilityRegime,
)


def _decision_id(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"activation-{hashlib.sha256(encoded).hexdigest()[:24]}"


class BalancedStrategyActivationPolicy(ActivationPolicy):
    """Filter strategy eligibility while leaving all risk limits downstream."""

    def __init__(self, config: BalancedStrategyPolicyConfig) -> None:
        if not config.enabled:
            raise RegimeInputError("disabled strategy policy cannot be activated")
        self.config = config
        self._config_hash = strategy_policy_config_hash(config)

    @classmethod
    def from_profile(
        cls, profile: TradingProfile
    ) -> BalancedStrategyActivationPolicy:
        return cls(load_balanced_strategy_policy_config(profile))

    @property
    def policy_name(self) -> str:
        return self.config.policy_name

    @property
    def policy_version(self) -> str:
        return self.config.policy_version

    @property
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        return self.config.to_parameters()

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def evaluate(
        self,
        *,
        strategy_name: str,
        strategy_version: str,
        signal: StrategySignal,
        regime: RegimeSnapshot,
        proposed_quantity: Decimal,
    ) -> ActivationDecision:
        if (
            strategy_name != signal.strategy_name
            or strategy_version != signal.strategy_version
        ):
            raise RegimeInputError("strategy identity must match the activation signal")
        if (
            signal.symbol != regime.symbol
            or signal.timeframe != regime.timeframe
            or signal.timestamp != regime.timestamp
        ):
            raise RegimeInputError("signal and regime snapshot must describe the same event")
        if proposed_quantity <= Decimal("0") or not proposed_quantity.is_finite():
            raise RegimeInputError("proposed quantity must be positive and finite")

        if signal.action is StrategySignalAction.EXIT_LONG:
            status = ActivationStatus.ALLOW
            multiplier = Decimal("1")
            codes = ("RISK_REDUCING_EXIT_ALLOWED",)
            reasons = ("EXIT_LONG is never blocked by regime eligibility.",)
        elif signal.action is not StrategySignalAction.ENTER_LONG:
            status = ActivationStatus.BLOCK
            multiplier = Decimal("0")
            codes = ("NON_ENTRY_SIGNAL_BLOCKED",)
            reasons = ("Only explicit entries or exits may create an order.",)
        else:
            rule = self.config.rule_for(regime.structure_regime, strategy_name)
            status = rule.status
            multiplier = rule.multiplier
            codes_list = [f"STRUCTURE_{regime.structure_regime.value}_{status.value}"]
            reasons_list = [
                f"{strategy_name} is {status.value.lower()} for "
                f"{regime.structure_regime.value}."
            ]
            overlay = self.config.overlay_for(regime.volatility_regime, strategy_name)
            if overlay is not None and overlay.multiplier < multiplier:
                status = overlay.status
                multiplier = overlay.multiplier
                codes_list.append(
                    f"VOLATILITY_{regime.volatility_regime.value}_{status.value}"
                )
                reasons_list.append(
                    f"{strategy_name} is {status.value.lower()} by the "
                    f"{regime.volatility_regime.value} volatility overlay."
                )
            codes = tuple(sorted(set(codes_list)))
            reasons = tuple(reasons_list)

        adjusted = proposed_quantity * multiplier
        payload = {
            "signal_id": signal.signal_id,
            "regime_snapshot_id": regime.snapshot_id,
            "strategy": strategy_name,
            "strategy_version": strategy_version,
            "status": status.value,
            "multiplier": str(multiplier),
            "proposed_quantity": str(proposed_quantity),
            "adjusted_quantity": str(adjusted),
            "policy_hash": self._config_hash,
        }
        return ActivationDecision(
            decision_id=_decision_id(payload),
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            signal_id=signal.signal_id,
            regime_snapshot_id=regime.snapshot_id,
            structure_regime=regime.structure_regime,
            volatility_regime=regime.volatility_regime,
            status=status,
            allocation_multiplier=multiplier,
            proposed_quantity=proposed_quantity,
            adjusted_quantity=adjusted,
            reason_codes=codes,
            human_readable_reasons=reasons,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            policy_config_hash=self._config_hash,
        )
