"""Framework-neutral ML training, registry, inference, and scoring platform."""

from trading_ai.ml.base import (
    MLScorer,
    ModelAdapter,
    ModelRegistry,
    ModelTrainer,
    RealTimeInferencePort,
)
from trading_ai.ml.datasets import (
    DatasetBuildReport,
    DatasetBuildResult,
    SignalTrainingDataset,
    SignalTrainingDatasetBuilder,
    SignalTrainingExample,
)
from trading_ai.ml.decisions import MLFilterDecision, MLPrediction
from trading_ai.ml.evaluation import ClassificationMetrics, EvaluationReport
from trading_ai.ml.features import ML_FEATURE_SCHEMA_VERSION, MLFeatureBuilder
from trading_ai.ml.inference import InferenceEngine, SignalMLScorer
from trading_ai.ml.inputs import InferenceRequest, ModelInput, TabularModelInput
from trading_ai.ml.labels import LabelBuilder, LabelConfig
from trading_ai.ml.models import (
    InputKind,
    InferenceMode,
    MLFilterStatus,
    MLMode,
    MLTask,
    ModelArtifact,
    ModelConfig,
    ModelFamily,
    ModelStatus,
    RegistryEventType,
    TimeRange,
)
from trading_ai.ml.registry import LocalModelRegistry
from trading_ai.ml.splits import PurgedWalkForwardSplitter, TemporalSplitConfig
from trading_ai.ml.training import TrainingConfig, TrainingOutcome, TrainingPipeline

__all__ = [
    "ClassificationMetrics",
    "DatasetBuildReport",
    "DatasetBuildResult",
    "EvaluationReport",
    "InferenceEngine",
    "InferenceMode",
    "InferenceRequest",
    "InputKind",
    "LabelBuilder",
    "LabelConfig",
    "LocalModelRegistry",
    "MLFilterDecision",
    "MLFilterStatus",
    "MLMode",
    "MLPrediction",
    "MLScorer",
    "MLTask",
    "MLFeatureBuilder",
    "ML_FEATURE_SCHEMA_VERSION",
    "ModelAdapter",
    "ModelArtifact",
    "ModelConfig",
    "ModelFamily",
    "ModelInput",
    "ModelRegistry",
    "ModelStatus",
    "ModelTrainer",
    "PurgedWalkForwardSplitter",
    "RealTimeInferencePort",
    "RegistryEventType",
    "SignalMLScorer",
    "SignalTrainingDataset",
    "SignalTrainingDatasetBuilder",
    "SignalTrainingExample",
    "TabularModelInput",
    "TemporalSplitConfig",
    "TimeRange",
    "TrainingConfig",
    "TrainingOutcome",
    "TrainingPipeline",
]
