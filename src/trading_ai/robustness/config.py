"""Frozen Lot 8.2 baseline and research-plan configuration loaders."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.hashing import stable_hash
from trading_ai.robustness.exceptions import RobustnessError
from trading_ai.robustness.models import (
    DatasetFingerprint,
    PeriodClassification,
    ResearchBaselineManifest,
    ResearchPeriod,
    RobustnessResearchPlan,
)


DEFAULT_ROBUSTNESS_PATH = PROJECT_ROOT / "config" / "robustness" / "balanced.toml"


def _decimal(raw: object, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise RobustnessError(f"{name} must be numeric") from exc
    if not value.is_finite():
        raise RobustnessError(f"{name} must be finite")
    return value


def _utc(raw: object, name: str) -> datetime:
    if not isinstance(raw, datetime) or raw.tzinfo is None or raw.utcoffset() is None:
        raise RobustnessError(f"{name} must be a timezone-aware TOML datetime")
    return raw.astimezone(timezone.utc)


def _resolve_config_path(raw: str, parent: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        raise RobustnessError("robustness configuration paths must be project-relative")
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if PROJECT_ROOT.resolve() not in project_candidate.parents:
        raise RobustnessError("robustness configuration path escapes project root")
    if not project_candidate.is_file():
        fallback = (parent / candidate).resolve()
        if PROJECT_ROOT.resolve() not in fallback.parents or not fallback.is_file():
            raise RobustnessError(f"robustness configuration not found: {raw}")
        return fallback
    return project_candidate


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    name: str
    version: str
    enabled: bool
    baseline_manifest_path: Path
    research_plan_path: Path
    minimum_holdout_calendar_days: int
    minimum_bootstrap_samples: int
    bootstrap_resamples: int
    bootstrap_seed: int
    drawdown_episode_min_fraction: Decimal
    concentration_warning_fraction: Decimal
    temporal_concentration_warning_fraction: Decimal
    survivorship_policy: str
    curated_universe_status: str
    config_hash: str

    def __post_init__(self) -> None:
        if not self.enabled:
            raise RobustnessError("Balanced robustness configuration must be enabled")
        if min(
            self.minimum_holdout_calendar_days,
            self.minimum_bootstrap_samples,
            self.bootstrap_resamples,
        ) < 1:
            raise RobustnessError("robustness sample controls must be positive")
        for value in (
            self.drawdown_episode_min_fraction,
            self.concentration_warning_fraction,
            self.temporal_concentration_warning_fraction,
        ):
            if not Decimal("0") < value <= Decimal("1"):
                raise RobustnessError("robustness fractions must be in (0, 1]")


def load_robustness_config(path: Path = DEFAULT_ROBUSTNESS_PATH) -> RobustnessConfig:
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        return RobustnessConfig(
            name=str(raw["name"]),
            version=str(raw["version"]),
            enabled=bool(raw["enabled"]),
            baseline_manifest_path=_resolve_config_path(
                str(raw["baseline_manifest"]), path.parent
            ),
            research_plan_path=_resolve_config_path(str(raw["research_plan"]), path.parent),
            minimum_holdout_calendar_days=int(raw["minimum_holdout_calendar_days"]),
            minimum_bootstrap_samples=int(raw["minimum_bootstrap_samples"]),
            bootstrap_resamples=int(raw["bootstrap_resamples"]),
            bootstrap_seed=int(raw["bootstrap_seed"]),
            drawdown_episode_min_fraction=_decimal(
                raw["drawdown_episode_min_fraction"], "drawdown_episode_min_fraction"
            ),
            concentration_warning_fraction=_decimal(
                raw["concentration_warning_fraction"], "concentration_warning_fraction"
            ),
            temporal_concentration_warning_fraction=_decimal(
                raw["temporal_concentration_warning_fraction"],
                "temporal_concentration_warning_fraction",
            ),
            survivorship_policy=str(raw["survivorship_policy"]),
            curated_universe_status=str(raw["curated_universe_status"]),
            config_hash=stable_hash(raw),
        )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        if isinstance(exc, RobustnessError):
            raise
        raise RobustnessError(f"invalid robustness configuration: {exc}") from exc


def load_research_baseline_manifest(
    path: Path | None = None,
) -> ResearchBaselineManifest:
    if path is None:
        path = load_robustness_config().baseline_manifest_path
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        digest = stable_hash(raw)
        datasets = tuple(
            sorted(
                (
                    DatasetFingerprint(
                        symbol=str(item["symbol"]),
                        dataset_id=str(item["dataset_id"]),
                        checksum=str(item["checksum"]),
                        corporate_actions_dataset_id=str(
                            item["corporate_actions_dataset_id"]
                        ),
                        corporate_actions_checksum=str(item["corporate_actions_checksum"]),
                    )
                    for item in raw["datasets"]
                ),
                key=lambda item: item.symbol,
            )
        )
        period = ResearchPeriod(
            name="consumed_diagnostic",
            classification=PeriodClassification(str(raw["period_classification"])),
            start=_utc(raw["period_start"], "period_start"),
            end=_utc(raw["period_end"], "period_end"),
            label=str(raw["period_label"]),
        )
        return ResearchBaselineManifest(
            manifest_id=f"research-baseline-{digest[:24]}",
            manifest_version=str(raw["version"]),
            manifest_hash=digest,
            frozen_at=_utc(raw["frozen_at"], "frozen_at"),
            commit_sha=str(raw["commit_sha"]),
            source_hash_sha256=str(raw["source_hash_sha256"]),
            run_id=str(raw["run_id"]),
            result_hash=str(raw["result_hash"]),
            validation_id=str(raw["validation_id"]),
            validation_status=str(raw["validation_status"]),
            period=period,
            timeframe=str(raw["timeframe"]),
            universe_kind=str(raw["universe_kind"]),
            symbols=tuple(sorted(str(item) for item in raw["symbols"])),
            config_hashes=tuple(
                sorted((str(key), str(value)) for key, value in raw["config_hashes"].items())
            ),
            datasets=datasets,
            tariff_profile_id=str(raw["tariff_profile_id"]),
            tariff_status=str(raw["tariff_status"]),
            tariff_period_verified=bool(raw["tariff_period_verified"]),
            tariff_config_hash=str(raw["tariff_config_hash"]),
            closed_trades=int(raw["closed_trades"]),
            max_drawdown=_decimal(raw["max_drawdown"], "max_drawdown"),
            net_return_before_operating=_decimal(
                raw["net_return_before_operating"], "net_return_before_operating"
            ),
            top_contributor=str(raw["top_contributor"]),
            top_contributor_share=_decimal(
                raw["top_contributor_share"], "top_contributor_share"
            ),
            warnings=tuple(str(item) for item in raw.get("warnings", ())),
        )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        if isinstance(exc, RobustnessError):
            raise
        raise RobustnessError(f"invalid frozen research baseline: {exc}") from exc


def _period(name: str, raw: dict[str, Any]) -> ResearchPeriod:
    return ResearchPeriod(
        name=name,
        classification=PeriodClassification(str(raw["classification"])),
        start=_utc(raw["start"], f"{name}.start"),
        end=_utc(raw["end"], f"{name}.end"),
        label=str(raw.get("label", name.upper())),
    )


def load_research_plan(
    path: Path | None = None,
    *,
    baseline: ResearchBaselineManifest | None = None,
    frozen_at: datetime | None = None,
) -> RobustnessResearchPlan:
    settings = load_robustness_config()
    path = path or settings.research_plan_path
    baseline = baseline or load_research_baseline_manifest(settings.baseline_manifest_path)
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
        semantic_plan = dict(raw)
        semantic_plan.pop("frozen_at", None)
        digest = stable_hash(
            {
                "plan": semantic_plan,
                "baseline_manifest_hash": baseline.manifest_hash,
                "decision_config_hashes": baseline.config_hashes,
            }
        )
        periods = tuple(
            _period(name, item)
            for name, item in sorted(raw["periods"].items())
        )
        return RobustnessResearchPlan(
            plan_id=f"robustness-plan-{digest[:24]}",
            plan_name=str(raw["name"]),
            plan_version=str(raw["version"]),
            plan_hash=digest,
            frozen_at=(
                frozen_at
                or _utc(raw["frozen_at"], "research plan frozen_at")
            ).astimezone(timezone.utc),
            frozen=bool(raw["frozen"]),
            baseline_manifest_hash=baseline.manifest_hash,
            timeframe=str(raw["timeframe"]),
            universe_kind=str(raw["universe_kind"]),
            symbols=tuple(sorted(str(item) for item in raw["symbols"])),
            strategies=tuple(sorted(str(item) for item in raw["strategies"])),
            ml_modes=tuple(str(item) for item in raw["ml_modes"]),
            cost_profile=str(raw["cost_profile"]),
            validation_profile=str(raw["validation_profile"]),
            periods=periods,
            frozen_validation_criteria=tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in raw["frozen_validation_criteria"].items()
                )
            ),
            config_hashes=baseline.config_hashes,
            planned_analyses=tuple(str(item) for item in raw["planned_analyses"]),
        )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        if isinstance(exc, RobustnessError):
            raise
        raise RobustnessError(f"invalid frozen robustness research plan: {exc}") from exc
