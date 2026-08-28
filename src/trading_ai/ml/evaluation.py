"""Deterministic classification metrics and non-causal interpretation reports."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from trading_ai.ml.exceptions import MLDataError


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_probability: float
    positive_rate: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    sample_count: int
    class_distribution: tuple[tuple[int, int], ...]
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    log_loss: float
    brier_score: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    calibration: tuple[CalibrationBin, ...]

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise ValueError("classification metrics require observations")
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "log_loss",
            "brier_score",
        ):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("roc_auc", "pr_auc"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when defined")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    validation: ClassificationMetrics
    final_test: ClassificationMetrics
    walk_forward_folds: int
    interpretation_kind: str
    feature_interpretation: tuple[tuple[str, float], ...]
    leakage_safeguards: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.walk_forward_folds < 1:
            raise ValueError("walk_forward_folds must be positive")
        if not self.interpretation_kind.strip():
            raise ValueError("interpretation_kind must not be empty")
        names = [name for name, _ in self.feature_interpretation]
        if len(names) != len(set(names)):
            raise ValueError("feature interpretation names must be unique")
        if tuple(sorted(set(self.leakage_safeguards))) != self.leakage_safeguards:
            raise ValueError("leakage safeguards must be sorted and unique")


def evaluate_binary_classification(
    targets: tuple[int, ...],
    probabilities: tuple[float, ...],
    *,
    calibration_bins: int = 10,
) -> ClassificationMetrics:
    """Compute fixed analytical metrics; no threshold or model is selected here."""

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if not targets or len(targets) != len(probabilities):
        raise MLDataError("targets and probabilities must have equal non-zero length")
    if any(target not in {0, 1} for target in targets):
        raise MLDataError("classification targets must be binary")
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise MLDataError("probabilities must be finite values in [0, 1]")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")
    predicted = tuple(int(value >= 0.5) for value in probabilities)
    classes = set(targets)
    accuracy = float(accuracy_score(targets, predicted))
    balanced_accuracy = (
        float(balanced_accuracy_score(targets, predicted))
        if len(classes) == 2
        else accuracy
    )
    roc_auc = float(roc_auc_score(targets, probabilities)) if len(classes) == 2 else None
    pr_auc = (
        float(average_precision_score(targets, probabilities))
        if len(classes) == 2
        else None
    )
    matrix = confusion_matrix(targets, predicted, labels=[0, 1])
    calibration: list[CalibrationBin] = []
    for index in range(calibration_bins):
        lower = index / calibration_bins
        upper = (index + 1) / calibration_bins
        members = [
            (target, probability)
            for target, probability in zip(targets, probabilities)
            if lower <= probability < upper
            or (index == calibration_bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        calibration.append(
            CalibrationBin(
                lower_bound=lower,
                upper_bound=upper,
                sample_count=len(members),
                mean_probability=sum(item[1] for item in members) / len(members),
                positive_rate=sum(item[0] for item in members) / len(members),
            )
        )
    return ClassificationMetrics(
        sample_count=len(targets),
        class_distribution=tuple(
            (label, targets.count(label)) for label in (0, 1)
        ),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision=float(precision_score(targets, predicted, zero_division=0)),
        recall=float(recall_score(targets, predicted, zero_division=0)),
        f1=float(f1_score(targets, predicted, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss=float(log_loss(targets, probabilities, labels=[0, 1])),
        brier_score=float(brier_score_loss(targets, probabilities)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
        calibration=tuple(calibration),
    )
