"""Provider-neutral boundaries for regime detection and activation policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from trading_ai.backtesting.models import StrategySignal
from trading_ai.features import FeatureRequest, FeatureSnapshot
from trading_ai.regimes.models import (
    ActivationDecision,
    RegimeSnapshot,
    RegimeTransition,
)


class RegimeDetector(ABC):
    """Classify feature snapshots without signals, orders, risk, or execution."""

    @property
    @abstractmethod
    def detector_name(self) -> str:
        pass

    @property
    @abstractmethod
    def detector_version(self) -> str:
        pass

    @property
    @abstractmethod
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        pass

    @property
    @abstractmethod
    def config_hash(self) -> str:
        pass

    @property
    @abstractmethod
    def feature_request(self) -> FeatureRequest:
        pass

    @property
    def transitions(self) -> tuple[RegimeTransition, ...]:
        return ()

    def reset(self) -> None:
        """Reset deterministic run state."""

    @abstractmethod
    def evaluate(self, features: FeatureSnapshot) -> RegimeSnapshot:
        """Return the regime known at the feature timestamp."""


class ActivationPolicy(ABC):
    """Decide strategy eligibility; never authorize risk or execution."""

    @property
    @abstractmethod
    def policy_name(self) -> str:
        pass

    @property
    @abstractmethod
    def policy_version(self) -> str:
        pass

    @property
    @abstractmethod
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        pass

    @property
    @abstractmethod
    def config_hash(self) -> str:
        pass

    @abstractmethod
    def evaluate(
        self,
        *,
        strategy_name: str,
        strategy_version: str,
        signal: StrategySignal,
        regime: RegimeSnapshot,
        proposed_quantity: Decimal,
    ) -> ActivationDecision:
        """Return a multiplier no greater than one for one signal proposal."""
