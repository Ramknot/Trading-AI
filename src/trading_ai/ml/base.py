"""Framework-neutral ports separating training, inference, and registry duties."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from trading_ai.ml.inputs import ModelInput
from trading_ai.ml.models import InputKind, MLMode, MLTask, ModelArtifact, ModelFamily

if TYPE_CHECKING:
    from trading_ai.backtesting.models import StrategySignal
    from trading_ai.features import FeatureSnapshot
    from trading_ai.ml.decisions import MLFilterDecision, MLPrediction
    from trading_ai.ml.evaluation import EvaluationReport
    from trading_ai.ml.models import ModelConfig
    from trading_ai.regimes.models import RegimeSnapshot
    from trading_ai.features import FeatureRequest


class ModelAdapter(ABC):
    """Frozen inference contract usable by tabular or future complex models."""

    @property
    @abstractmethod
    def model_family(self) -> ModelFamily | str: ...

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @property
    @abstractmethod
    def task(self) -> MLTask: ...

    @property
    @abstractmethod
    def input_kind(self) -> InputKind: ...

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def framework(self) -> str: ...

    @property
    @abstractmethod
    def framework_version(self) -> str: ...

    @abstractmethod
    def score_one(self, model_input: ModelInput) -> float:
        """Return one positive-class probability without changing model state."""

    @abstractmethod
    def score_batch(self, model_inputs: Sequence[ModelInput]) -> tuple[float, ...]:
        """Return probabilities in input order without fitting or updating."""

    @abstractmethod
    def serialize(self) -> bytes:
        """Return the frozen payload stored by an integrity-checking registry."""

    def interpretation(self) -> tuple[tuple[str, float], ...]:
        return ()


class ModelTrainer(ABC):
    """Explicit training port; it is intentionally absent from inference."""

    @abstractmethod
    def fit(
        self,
        model_inputs: Sequence[ModelInput],
        targets: Sequence[int],
        config: ModelConfig,
    ) -> ModelAdapter:
        """Fit only the explicitly supplied training observations."""


class ModelRegistry(ABC):
    """Storage/lifecycle contract replaceable by a future external registry."""

    @abstractmethod
    def save(self, outcome: Any) -> ModelArtifact: ...

    @abstractmethod
    def load(self, model_id: str, **compatibility: Any) -> Any: ...

    @abstractmethod
    def list(self) -> tuple[ModelArtifact, ...]: ...

    @abstractmethod
    def inspect(self, model_id: str) -> dict[str, Any]: ...


class MLScorer(ABC):
    """Score existing strategy signals without signal, sizing, risk, or broker authority."""

    @property
    @abstractmethod
    def mode(self) -> MLMode: ...

    @property
    @abstractmethod
    def artifact(self) -> ModelArtifact | None: ...

    @property
    @abstractmethod
    def feature_request(self) -> FeatureRequest: ...

    @abstractmethod
    def evaluate(
        self,
        *,
        signal: StrategySignal,
        features: FeatureSnapshot,
        regime: RegimeSnapshot,
    ) -> tuple[MLPrediction | None, MLFilterDecision]:
        """Return analytics/filtering only; never an order or a larger quantity."""


class RealTimeInferencePort(ABC):
    """Future event-driven inference port; no transport or stream is provided."""

    @abstractmethod
    def score_one(self, model_input: ModelInput) -> MLPrediction:
        """Score one already-built point-in-time input."""
