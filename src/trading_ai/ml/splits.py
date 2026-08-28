"""Purged chronological TRAIN/VALIDATION/TEST and expanding walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_ai.ml.exceptions import MLConfigurationError, MLDataError
from trading_ai.ml.models import TimeRange


@dataclass(frozen=True, slots=True)
class TemporalSplitConfig:
    training: TimeRange
    validation: TimeRange
    final_test: TimeRange
    embargo_bars: int = 1
    walk_forward_folds: int = 3

    def __post_init__(self) -> None:
        if self.training.end > self.validation.start:
            raise MLConfigurationError("TRAIN must end before VALIDATION starts")
        if self.validation.end > self.final_test.start:
            raise MLConfigurationError("VALIDATION must end before FINAL TEST starts")
        if type(self.embargo_bars) is not int or self.embargo_bars < 0:
            raise MLConfigurationError("embargo_bars must be a non-negative integer")
        if type(self.walk_forward_folds) is not int or self.walk_forward_folds < 1:
            raise MLConfigurationError("walk_forward_folds must be positive")

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    ("embargo_bars", str(self.embargo_bars)),
                    ("final_test_end", self.final_test.end.isoformat()),
                    ("final_test_start", self.final_test.start.isoformat()),
                    ("training_end", self.training.end.isoformat()),
                    ("training_start", self.training.start.isoformat()),
                    ("validation_end", self.validation.end.isoformat()),
                    ("validation_start", self.validation.start.isoformat()),
                    ("walk_forward_folds", str(self.walk_forward_folds)),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class TemporalPartition:
    training: tuple[object, ...]
    validation: tuple[object, ...]
    final_test: tuple[object, ...]
    purged_count: int
    embargoed_count: int


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    training: tuple[object, ...]
    validation: tuple[object, ...]
    validation_start: datetime
    validation_end: datetime


class PurgedWalkForwardSplitter:
    """Use shared UTC boundaries for every asset; no shuffle is available."""

    def __init__(self, config: TemporalSplitConfig) -> None:
        self.config = config

    @staticmethod
    def _timestamps(examples: tuple[object, ...]) -> tuple[datetime, ...]:
        return tuple(sorted({item.model_input.timestamp for item in examples}))

    def _embargo(
        self, examples: tuple[object, ...]
    ) -> tuple[tuple[object, ...], int]:
        if self.config.embargo_bars == 0:
            return examples, 0
        timestamps = self._timestamps(examples)
        excluded = set(timestamps[: self.config.embargo_bars])
        retained = tuple(
            item for item in examples if item.model_input.timestamp not in excluded
        )
        return retained, len(examples) - len(retained)

    def partition(self, examples: tuple[object, ...]) -> TemporalPartition:
        if not examples:
            raise MLDataError("temporal split requires examples")
        ordered = tuple(
            sorted(examples, key=lambda item: (item.model_input.timestamp, item.model_input.symbol))
        )
        training_raw = tuple(
            item
            for item in ordered
            if self.config.training.contains(item.model_input.timestamp)
        )
        validation_raw = tuple(
            item
            for item in ordered
            if self.config.validation.contains(item.model_input.timestamp)
        )
        test_raw = tuple(
            item
            for item in ordered
            if self.config.final_test.contains(item.model_input.timestamp)
        )
        training = tuple(
            item
            for item in training_raw
            if item.label_end_timestamp < self.config.validation.start
        )
        validation = tuple(
            item
            for item in validation_raw
            if item.label_end_timestamp < self.config.final_test.start
        )
        purged = (len(training_raw) - len(training)) + (
            len(validation_raw) - len(validation)
        )
        validation, validation_embargo = self._embargo(validation)
        final_test, test_embargo = self._embargo(test_raw)
        if not training or not validation or not final_test:
            raise MLDataError("TRAIN, VALIDATION, and FINAL TEST must all contain examples")
        return TemporalPartition(
            training=training,
            validation=validation,
            final_test=final_test,
            purged_count=purged,
            embargoed_count=validation_embargo + test_embargo,
        )

    def walk_forward_folds(
        self, partition: TemporalPartition
    ) -> tuple[WalkForwardFold, ...]:
        validation_timestamps = self._timestamps(partition.validation)
        fold_count = min(self.config.walk_forward_folds, len(validation_timestamps))
        if fold_count < 1:
            raise MLDataError("walk-forward validation requires timestamps")
        chunks: list[tuple[datetime, ...]] = []
        for index in range(fold_count):
            start = index * len(validation_timestamps) // fold_count
            end = (index + 1) * len(validation_timestamps) // fold_count
            chunks.append(validation_timestamps[start:end])
        folds: list[WalkForwardFold] = []
        prior_validation: tuple[object, ...] = ()
        for timestamps in chunks:
            if not timestamps:
                continue
            fold_start = timestamps[0]
            validation = tuple(
                item
                for item in partition.validation
                if item.model_input.timestamp in set(timestamps)
            )
            training = tuple(
                item
                for item in (*partition.training, *prior_validation)
                if item.label_end_timestamp < fold_start
            )
            if not training or not validation:
                raise MLDataError("walk-forward fold has an empty TRAIN or VALIDATION")
            folds.append(
                WalkForwardFold(
                    training=training,
                    validation=validation,
                    validation_start=fold_start,
                    validation_end=timestamps[-1] + timedelta(microseconds=1),
                )
            )
            prior_validation = (*prior_validation, *validation)
        return tuple(folds)
