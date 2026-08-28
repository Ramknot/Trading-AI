"""Frozen-model inference and fail-closed ML signal filtering."""

from __future__ import annotations

from collections.abc import Sequence

from trading_ai.backtesting.models import StrategySignal, StrategySignalAction
from trading_ai.features import FeatureSnapshot
from trading_ai.ml.base import MLScorer, ModelAdapter, RealTimeInferencePort
from trading_ai.ml.decisions import MLFilterDecision, MLPrediction
from trading_ai.ml.exceptions import MLConfigurationError, MLDataError, MLInferenceError
from trading_ai.ml.features import MLFeatureBuilder
from trading_ai.ml.inputs import ModelInput
from trading_ai.ml.models import (
    InferenceMode,
    MLFilterStatus,
    MLMode,
    ModelArtifact,
    ModelStatus,
)
from trading_ai.regimes.models import RegimeSnapshot


class InferenceEngine(RealTimeInferencePort):
    """Score immutable inputs with one already-loaded model artifact."""

    def __init__(self, artifact: ModelArtifact, adapter: ModelAdapter) -> None:
        if artifact.model_family != adapter.model_family:
            raise MLConfigurationError("model family does not match its artifact")
        if artifact.model_version != adapter.model_version:
            raise MLConfigurationError("model version does not match its artifact")
        if artifact.task is not adapter.task:
            raise MLConfigurationError("model task does not match its artifact")
        if artifact.input_kind is not adapter.input_kind:
            raise MLConfigurationError("model input kind does not match its artifact")
        if artifact.feature_names != adapter.feature_names:
            raise MLConfigurationError("model feature order does not match its artifact")
        if artifact.framework != adapter.framework:
            raise MLConfigurationError("model framework does not match its artifact")
        self.artifact = artifact
        self._adapter = adapter

    def _validate_input(self, model_input: ModelInput) -> None:
        if model_input.input_kind is not self.artifact.input_kind:
            raise MLInferenceError("input kind is incompatible with model artifact")
        if model_input.feature_names != self.artifact.feature_names:
            raise MLInferenceError("input feature schema/order is incompatible")
        compatibility = (
            ("strategy_name", self.artifact.strategy_name),
            ("strategy_version", self.artifact.strategy_version),
            ("timeframe", self.artifact.timeframe),
            ("feature_schema_version", self.artifact.feature_schema_version),
            ("ml_feature_schema_version", self.artifact.ml_feature_schema_version),
        )
        for field_name, expected in compatibility:
            actual = getattr(model_input, field_name, expected)
            if actual != expected:
                raise MLInferenceError(
                    f"model input {field_name}={actual!r} is incompatible with {expected!r}"
                )

    def _prediction(
        self,
        model_input: ModelInput,
        probability: float,
        mode: InferenceMode,
    ) -> MLPrediction:
        return MLPrediction.create(
            timestamp=model_input.timestamp,
            symbol=model_input.symbol,
            strategy_name=model_input.strategy_name,
            strategy_version=model_input.strategy_version,
            model_id=self.artifact.model_id,
            model_family=self.artifact.model_family,
            model_version=self.artifact.model_version,
            probability_positive=probability,
            feature_schema_version=model_input.feature_schema_version,
            ml_feature_schema_version=model_input.ml_feature_schema_version,
            input_hash=model_input.input_hash,
            model_artifact_hash=self.artifact.artifact_checksum,
            inference_mode=mode,
            technical_latency_ms=None,
        )

    def score_one(self, model_input: ModelInput) -> MLPrediction:
        self._validate_input(model_input)
        return self._prediction(
            model_input, self._adapter.score_one(model_input), InferenceMode.ONE
        )

    def score_batch(
        self, model_inputs: Sequence[ModelInput]
    ) -> tuple[MLPrediction, ...]:
        values = tuple(model_inputs)
        for item in values:
            self._validate_input(item)
        probabilities = self._adapter.score_batch(values)
        if len(probabilities) != len(values):
            raise MLInferenceError("adapter returned an unexpected prediction count")
        return tuple(
            self._prediction(item, probability, InferenceMode.BATCH)
            for item, probability in zip(values, probabilities, strict=True)
        )


class SignalMLScorer(MLScorer):
    """PASS/BLOCK filter for candidate entries; exits always continue."""

    def __init__(
        self,
        *,
        mode: MLMode = MLMode.DISABLED,
        inference_engine: InferenceEngine | None = None,
        threshold: float = 0.55,
        feature_builder: MLFeatureBuilder | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise MLConfigurationError("ML threshold must be in [0, 1]")
        if mode is not MLMode.DISABLED and inference_engine is None:
            raise MLConfigurationError("active ML mode requires an explicit model")
        if (
            mode is MLMode.FILTER
            and inference_engine is not None
            and inference_engine.artifact.status is not ModelStatus.APPROVED
        ):
            raise MLConfigurationError("FILTER mode requires an APPROVED model")
        self._mode = mode
        self.inference_engine = inference_engine
        self.threshold = threshold
        self.feature_builder = feature_builder or MLFeatureBuilder()

    @property
    def mode(self) -> MLMode:
        return self._mode

    @property
    def artifact(self) -> ModelArtifact | None:
        return self.inference_engine.artifact if self.inference_engine else None

    @property
    def feature_request(self):
        return self.feature_builder.feature_request

    def _decision(
        self,
        signal: StrategySignal,
        *,
        status: MLFilterStatus,
        reason_code: str,
        human_reason: str,
        prediction: MLPrediction | None = None,
    ) -> MLFilterDecision:
        return MLFilterDecision.create(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            signal_id=signal.signal_id,
            mode=self.mode,
            status=status,
            threshold=self.threshold if self.mode is MLMode.FILTER else None,
            probability=prediction.probability_positive if prediction else None,
            reason_code=reason_code,
            human_reason=human_reason,
            model_id=self.artifact.model_id if self.artifact else None,
            prediction_id=prediction.prediction_id if prediction else None,
        )

    def evaluate(
        self,
        *,
        signal: StrategySignal,
        features: FeatureSnapshot,
        regime: RegimeSnapshot,
    ) -> tuple[MLPrediction | None, MLFilterDecision]:
        if signal.action is StrategySignalAction.EXIT_LONG:
            return None, self._decision(
                signal,
                status=MLFilterStatus.NOT_APPLICABLE,
                reason_code="EXIT_NOT_FILTERED",
                human_reason="EXIT_LONG is never blocked by the ML filter.",
            )
        if self.mode is MLMode.DISABLED:
            return None, self._decision(
                signal,
                status=MLFilterStatus.NOT_APPLICABLE,
                reason_code="ML_DISABLED",
                human_reason="Quantitative signal continues without ML scoring.",
            )
        try:
            model_input = self.feature_builder.build(
                signal=signal, features=features, regime=regime
            )
            prediction = self.inference_engine.score_one(model_input)
        except (MLDataError, MLInferenceError) as exc:
            return None, self._decision(
                signal,
                status=MLFilterStatus.UNAVAILABLE,
                reason_code="ML_INPUT_OR_MODEL_UNAVAILABLE",
                human_reason=f"ML prediction unavailable: {exc}",
            )
        if self.mode is MLMode.SCORE_ONLY:
            return prediction, self._decision(
                signal,
                status=MLFilterStatus.PASS,
                reason_code="SCORE_RECORDED_NO_FILTER",
                human_reason="Prediction recorded; SCORE_ONLY cannot alter trading.",
                prediction=prediction,
            )
        passed = prediction.probability_positive >= self.threshold
        return prediction, self._decision(
            signal,
            status=MLFilterStatus.PASS if passed else MLFilterStatus.BLOCK,
            reason_code="ML_THRESHOLD_PASS" if passed else "ML_THRESHOLD_BLOCK",
            human_reason=(
                f"P(success)={prediction.probability_positive:.6f} "
                f"{'meets' if passed else 'is below'} threshold={self.threshold:.6f}."
            ),
            prediction=prediction,
        )
