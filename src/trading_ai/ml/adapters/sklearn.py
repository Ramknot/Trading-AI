"""Deterministic scikit-learn tabular classifier adapter."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from math import isfinite
from typing import Any

from trading_ai.ml.base import ModelAdapter, ModelTrainer
from trading_ai.ml.exceptions import MLDataError, MLInferenceError
from trading_ai.ml.inputs import ModelInput, TabularModelInput
from trading_ai.ml.models import InputKind, MLTask, ModelConfig, ModelFamily


class SklearnClassifierAdapter(ModelAdapter):
    """Frozen positive-class scorer; it exposes no training operation."""

    def __init__(
        self,
        estimator: Any,
        *,
        family: ModelFamily,
        model_version: str,
        feature_names: tuple[str, ...],
    ) -> None:
        if not feature_names or len(feature_names) != len(set(feature_names)):
            raise ValueError("feature_names must be non-empty and unique")
        if not hasattr(estimator, "predict_proba"):
            raise TypeError("classifier must expose predict_proba")
        self._estimator = estimator
        self._family = family
        self._model_version = model_version
        self._feature_names = feature_names

    @property
    def model_family(self) -> ModelFamily:
        return self._family

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def task(self) -> MLTask:
        return MLTask.SIGNAL_QUALITY_BINARY

    @property
    def input_kind(self) -> InputKind:
        return InputKind.TABULAR

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def framework(self) -> str:
        return "scikit-learn"

    @property
    def framework_version(self) -> str:
        import sklearn

        return sklearn.__version__

    @property
    def preprocessing_mean(self) -> tuple[float, ...] | None:
        """Expose fitted TRAIN-only scaler state for leakage audits."""

        named_steps = getattr(self._estimator, "named_steps", None)
        scaler = named_steps.get("scaler") if named_steps is not None else None
        mean = getattr(scaler, "mean_", None)
        return None if mean is None else tuple(float(value) for value in mean)

    def _row(self, model_input: ModelInput) -> tuple[float, ...]:
        if not isinstance(model_input, TabularModelInput):
            raise MLInferenceError("scikit-learn adapter requires TabularModelInput")
        if model_input.feature_names != self.feature_names:
            raise MLInferenceError("model input feature ordering does not match artifact")
        return model_input.numeric_values

    def score_one(self, model_input: ModelInput) -> float:
        return self.score_batch((model_input,))[0]

    def score_batch(self, model_inputs: Sequence[ModelInput]) -> tuple[float, ...]:
        if not model_inputs:
            return ()
        rows = [self._row(item) for item in model_inputs]
        try:
            probabilities = self._estimator.predict_proba(rows)
        except Exception as exc:
            raise MLInferenceError("scikit-learn probability inference failed") from exc
        classes = tuple(int(value) for value in self._estimator.classes_)
        if 1 not in classes:
            raise MLInferenceError("classifier artifact has no positive class")
        positive_index = classes.index(1)
        values = tuple(float(row[positive_index]) for row in probabilities)
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise MLInferenceError("classifier returned an invalid probability")
        return values

    def serialize(self) -> bytes:
        import joblib

        buffer = BytesIO()
        joblib.dump(
            {
                "estimator": self._estimator,
                "family": self._family.value,
                "model_version": self._model_version,
                "feature_names": self._feature_names,
            },
            buffer,
            compress=0,
            protocol=5,
        )
        return buffer.getvalue()

    @classmethod
    def deserialize(cls, payload: bytes) -> SklearnClassifierAdapter:
        import joblib

        try:
            data = joblib.load(BytesIO(payload))
            return cls(
                data["estimator"],
                family=ModelFamily(data["family"]),
                model_version=data["model_version"],
                feature_names=tuple(data["feature_names"]),
            )
        except Exception as exc:
            raise MLInferenceError("invalid scikit-learn model payload") from exc

    def interpretation(self) -> tuple[tuple[str, float], ...]:
        estimator = self._estimator
        if self._family is ModelFamily.LOGISTIC:
            estimator = estimator.named_steps["classifier"]
            values = estimator.coef_[0]
        else:
            values = estimator.feature_importances_
        return tuple(
            (name, float(value))
            for name, value in zip(self.feature_names, values, strict=True)
        )


class SklearnClassifierTrainer(ModelTrainer):
    """Explicit fitting implementation kept out of inference consumers."""

    def fit(
        self,
        model_inputs: Sequence[ModelInput],
        targets: Sequence[int],
        config: ModelConfig,
    ) -> SklearnClassifierAdapter:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if not model_inputs or len(model_inputs) != len(targets):
            raise MLDataError("training inputs and targets must have equal non-zero length")
        if set(targets) != {0, 1}:
            raise MLDataError("binary training requires both class 0 and class 1")
        if not all(isinstance(item, TabularModelInput) for item in model_inputs):
            raise MLDataError("Lot 6 scikit-learn training accepts tabular inputs only")
        feature_names = model_inputs[0].feature_names
        if any(item.feature_names != feature_names for item in model_inputs):
            raise MLDataError("all training inputs must use identical feature ordering")
        rows = [item.numeric_values for item in model_inputs]
        if config.family is ModelFamily.LOGISTIC:
            estimator = Pipeline(
                (
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=config.logistic_c,
                            max_iter=config.logistic_max_iter,
                            random_state=config.random_state,
                        ),
                    ),
                )
            )
        elif config.family is ModelFamily.RANDOM_FOREST:
            estimator = RandomForestClassifier(
                n_estimators=config.random_forest_estimators,
                max_depth=config.random_forest_max_depth,
                min_samples_leaf=config.random_forest_min_samples_leaf,
                random_state=config.random_state,
                n_jobs=1,
            )
        elif config.family is ModelFamily.GRADIENT_BOOSTING:
            estimator = GradientBoostingClassifier(
                n_estimators=config.gradient_boosting_estimators,
                learning_rate=config.gradient_boosting_learning_rate,
                max_depth=config.gradient_boosting_max_depth,
                random_state=config.random_state,
            )
        else:  # pragma: no cover
            raise MLDataError(f"unsupported model family {config.family}")
        estimator.fit(rows, tuple(targets))
        return SklearnClassifierAdapter(
            estimator,
            family=config.family,
            model_version=config.model_version,
            feature_names=feature_names,
        )
