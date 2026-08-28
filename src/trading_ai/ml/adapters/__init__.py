"""Framework-specific adapters isolated behind the generic ML contracts."""

from trading_ai.ml.adapters.sklearn import (
    SklearnClassifierAdapter,
    SklearnClassifierTrainer,
)

__all__ = ["SklearnClassifierAdapter", "SklearnClassifierTrainer"]
