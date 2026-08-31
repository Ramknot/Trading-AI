"""Baseline verification, decision-core hashing, and holdout access policy."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.hashing import stable_hash
from trading_ai.robustness.exceptions import (
    BaselineMismatchError,
    HoldoutGovernanceError,
)
from trading_ai.robustness.models import (
    HoldoutRecord,
    HoldoutStatus,
    PeriodClassification,
    ResearchBaselineManifest,
    RobustnessResearchPlan,
)


class HoldoutConsumer(str, Enum):
    FINAL_EVALUATION = "FINAL_EVALUATION"
    REPRODUCIBILITY = "REPRODUCIBILITY"
    ROBUSTNESS_REPORTING = "ROBUSTNESS_REPORTING"
    TRAINING_PIPELINE = "TRAINING_PIPELINE"
    EDGE_ESTIMATOR = "EDGE_ESTIMATOR"
    CONFIG_SELECTION = "CONFIG_SELECTION"


_DECISION_SOURCE_DIRS = (
    "backtesting",
    "costs",
    "features",
    "ml",
    "portfolio",
    "regimes",
    "risk",
    "strategies",
)
_DECISION_CONFIG_DIRS = (
    "costs",
    "portfolio",
    "profiles",
    "regimes",
    "risk",
    "validation",
)


def decision_core_hash(project_root: Path = PROJECT_ROOT) -> str:
    """Hash only decision code/config; reporting changes do not consume a holdout."""

    digest = hashlib.sha256()
    for directory in _DECISION_SOURCE_DIRS:
        root = project_root / "src" / "trading_ai" / directory
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    for directory in _DECISION_CONFIG_DIRS:
        root = project_root / "config" / directory
        for path in sorted(root.rglob("*.toml")):
            digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def observed_decision_config_hashes(
    summary: dict[str, Any], plan: RobustnessResearchPlan
) -> tuple[tuple[str, str], ...]:
    """Recover exported decision hashes while excluding holdout datasets.

    Dataset identities are frozen separately in every result.  Treating the
    V1 dataset hash as a decision-configuration hash would make any legitimate
    V2 holdout impossible.  Missing legacy fields retain the predeclared hash
    and remain protected by the decision-core source/config hash.
    """

    expected = dict(plan.config_hashes)
    observed = dict(expected)
    observed.pop("datasets", None)
    if isinstance(summary.get("config"), dict):
        observed["backtest"] = stable_hash(summary["config"])
    if isinstance(summary.get("strategy_parameters"), list):
        observed["strategy"] = stable_hash(summary["strategy_parameters"])
    if isinstance(summary.get("ml"), dict):
        observed["ml"] = stable_hash(summary["ml"])
    mappings = (
        ("cost", "costs", "config_hash"),
        ("portfolio", "portfolio", "config_hash"),
        ("regime", "regime", "config_hash"),
        ("policy", "regime", "policy_config_hash"),
        ("risk", "risk", "config_hash"),
    )
    for target, section, field in mappings:
        value = summary.get(section)
        if isinstance(value, dict) and value.get(field):
            observed[target] = str(value[field])
    return tuple(sorted(observed.items()))


class BaselineReproducer:
    """Verify an existing trusted export against the immutable Lot 8.1 facts."""

    @staticmethod
    def verify(
        manifest: ResearchBaselineManifest,
        *,
        summary: dict[str, Any],
        integrity_verified: bool,
    ) -> bool:
        if not integrity_verified:
            raise BaselineMismatchError("baseline export integrity is not verified")
        observed_datasets = tuple(
            sorted(
                (
                    str(item.get("symbol")),
                    str(item.get("dataset_id")),
                    str(item.get("checksum_sha256")),
                    str(item.get("corporate_actions_dataset_id")),
                    str(item.get("corporate_actions_checksum_sha256")),
                )
                for item in summary.get("dataset_references", ())
            )
        )
        expected_datasets = tuple(
            (
                item.symbol,
                item.dataset_id,
                item.checksum,
                str(item.corporate_actions_dataset_id),
                str(item.corporate_actions_checksum),
            )
            for item in manifest.datasets
        )
        checks = {
            "run_id": summary.get("run_id") == manifest.run_id,
            "result_hash": summary.get("result_hash") == manifest.result_hash,
            "source_hash": summary.get("source_hash_sha256") == manifest.source_hash_sha256,
            "datasets": observed_datasets == expected_datasets,
            "risk_hash": (summary.get("risk") or {}).get("config_hash")
            == dict(manifest.config_hashes)["risk"],
            "regime_hash": (summary.get("regime") or {}).get("config_hash")
            == dict(manifest.config_hashes)["regime"],
            "policy_hash": (summary.get("regime") or {}).get("policy_config_hash")
            == dict(manifest.config_hashes)["policy"],
            "portfolio_hash": (summary.get("portfolio") or {}).get("config_hash")
            == dict(manifest.config_hashes)["portfolio"],
            "cost_hash": (summary.get("costs") or {}).get("config_hash")
            == dict(manifest.config_hashes)["cost"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise BaselineMismatchError(
                "frozen Lot 8.1 baseline mismatch: " + ", ".join(failed)
            )
        return True


class HoldoutAccessPolicy:
    """Keep the final holdout away from fitting, edge calibration, and tuning."""

    @staticmethod
    def authorize(record: HoldoutRecord, consumer: HoldoutConsumer) -> None:
        forbidden = {
            HoldoutConsumer.TRAINING_PIPELINE,
            HoldoutConsumer.EDGE_ESTIMATOR,
            HoldoutConsumer.CONFIG_SELECTION,
        }
        if consumer in forbidden:
            raise HoldoutGovernanceError(
                f"FINAL_HOLDOUT data is forbidden for {consumer.value}"
            )
        if record.status is HoldoutStatus.UNTOUCHED:
            if consumer is not HoldoutConsumer.FINAL_EVALUATION:
                raise HoldoutGovernanceError(
                    "untouched holdout may be read only by its one final evaluation"
                )
            return
        if record.status is HoldoutStatus.CONSUMED:
            if consumer not in {
                HoldoutConsumer.REPRODUCIBILITY,
                HoldoutConsumer.ROBUSTNESS_REPORTING,
            }:
                raise HoldoutGovernanceError(
                    "consumed holdout is available only for reproduction/reporting"
                )
            return
        raise HoldoutGovernanceError("invalidated holdout is ineligible as final evidence")


def make_untouched_holdout(
    plan: RobustnessResearchPlan,
    *,
    core_hash: str | None = None,
) -> HoldoutRecord:
    period = plan.period(classification=PeriodClassification.FINAL_HOLDOUT)
    core = core_hash or decision_core_hash()
    identity_payload = {
        "plan_hash": plan.plan_hash,
        "period": period,
    }
    expected_configs = tuple(
        item for item in plan.config_hashes if item[0] != "datasets"
    )
    payload = {
        **identity_payload,
        "expected_core_hash": core,
        "expected_config_hashes": expected_configs,
        "status": HoldoutStatus.UNTOUCHED,
    }
    digest = stable_hash(payload)
    identity_digest = stable_hash(identity_payload)
    return HoldoutRecord(
        # The registry identity is stable for one predeclared holdout.  Core or
        # configuration changes alter the immutable record hash, not the ID;
        # otherwise a changed core could silently create a fresh "untouched"
        # holdout after the original one had already been consumed.
        holdout_id=f"holdout-{identity_digest[:24]}",
        plan_hash=plan.plan_hash,
        period=period,
        status=HoldoutStatus.UNTOUCHED,
        expected_core_hash=core,
        expected_config_hashes=expected_configs,
        record_hash=digest,
    )


def consume_holdout(
    record: HoldoutRecord,
    *,
    result_hash: str,
    core_hash: str,
    config_hashes: tuple[tuple[str, str], ...],
    consumed_at: datetime | None = None,
) -> tuple[HoldoutRecord, bool]:
    """Consume once; an exact rerun is reproducible, a changed core invalidates."""

    if record.status is HoldoutStatus.INVALIDATED:
        raise HoldoutGovernanceError("invalidated holdout cannot be consumed")
    if record.status is HoldoutStatus.CONSUMED:
        exact = (
            record.result_hash == result_hash
            and record.expected_core_hash == core_hash
            and record.expected_config_hashes == config_hashes
        )
        if exact:
            return record, True
        now = (consumed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        reason = "CORE_OR_CONFIG_CHANGED_AFTER_HOLDOUT_CONSUMPTION"
        payload = {
            "holdout_id": record.holdout_id,
            "status": HoldoutStatus.INVALIDATED,
            "original_record_hash": record.record_hash,
            "attempted_result_hash": result_hash,
            "attempted_core_hash": core_hash,
            "attempted_config_hashes": config_hashes,
            "reason": reason,
        }
        return HoldoutRecord(
            holdout_id=record.holdout_id,
            plan_hash=record.plan_hash,
            period=record.period,
            status=HoldoutStatus.INVALIDATED,
            expected_core_hash=record.expected_core_hash,
            expected_config_hashes=record.expected_config_hashes,
            record_hash=stable_hash(payload),
            consumed_at=record.consumed_at,
            result_hash=record.result_hash,
            invalidated_at=now,
            invalidation_reason=reason,
        ), False
    if core_hash != record.expected_core_hash or config_hashes != record.expected_config_hashes:
        raise HoldoutGovernanceError(
            "core/config hash differs from the plan frozen before holdout evaluation"
        )
    now = (consumed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    payload = {
        "holdout_id": record.holdout_id,
        "plan_hash": record.plan_hash,
        "period": record.period,
        "status": HoldoutStatus.CONSUMED,
        "core_hash": core_hash,
        "config_hashes": config_hashes,
        "result_hash": result_hash,
    }
    return HoldoutRecord(
        holdout_id=record.holdout_id,
        plan_hash=record.plan_hash,
        period=record.period,
        status=HoldoutStatus.CONSUMED,
        expected_core_hash=core_hash,
        expected_config_hashes=config_hashes,
        record_hash=stable_hash(payload),
        consumed_at=now,
        result_hash=result_hash,
    ), False
