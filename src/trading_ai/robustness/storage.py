"""Tamper-evident local storage for frozen plans, holdouts, and diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from trading_ai.core.hashing import to_primitive
from trading_ai.robustness.exceptions import RobustnessStorageError
from trading_ai.robustness.models import (
    HoldoutRecord,
    PaperReadinessReport,
    ResearchBaselineManifest,
    ResearchPeriod,
    RobustnessReport,
    RobustnessResearchPlan,
)


ROBUSTNESS_EXPORT_SCHEMA_VERSION = "1.7"
EVIDENCE_EXPORT_SCHEMA_VERSION = "1.8"
ECONOMIC_RECOMPUTATION_SCHEMA_VERSION = "1.9"
_SUPPORTED_ROBUSTNESS_SCHEMAS = {"1.7", "1.8", "1.9"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class LocalRobustnessStore:
    """Store analytics under data_local without mutating backtest artifacts."""

    def __init__(self, root: Path | str = Path("data_local/robustness")) -> None:
        self.root = Path(root)

    def _safe_directory(self, category: str, artifact_id: str) -> Path:
        if not _SAFE_ID.fullmatch(category) or not _SAFE_ID.fullmatch(artifact_id):
            raise RobustnessStorageError("invalid robustness artifact identifier")
        root = self.root.resolve()
        directory = (self.root / category / artifact_id).resolve()
        if root not in directory.parents:
            raise RobustnessStorageError("robustness artifact path escapes local store")
        return directory

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(
            to_primitive(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def _save_single(
        self,
        category: str,
        artifact_id: str,
        filename: str,
        value: Any,
        *,
        technical_fields: tuple[str, ...] = (),
    ) -> Path:
        directory = self._safe_directory(category, artifact_id)
        payload = self._canonical(value)
        target = directory / filename
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            incoming = json.loads(payload)
            for field in technical_fields:
                existing.pop(field, None)
                incoming.pop(field, None)
            if existing != incoming:
                raise RobustnessStorageError(
                    f"{artifact_id} collides with a different immutable artifact"
                )
            self._verify_files(directory)
            return directory
        directory.mkdir(parents=True, exist_ok=False)
        self._atomic_write(target, payload)
        checksums = {
            "algorithm": "SHA-256",
            "schema_version": ROBUSTNESS_EXPORT_SCHEMA_VERSION,
            "files": {filename: _sha256(payload)},
        }
        self._atomic_write(directory / "checksums.json", self._canonical(checksums))
        return directory

    def save_plan(self, plan: RobustnessResearchPlan) -> Path:
        return self._save_single(
            "plans", plan.plan_id, "plan.json", plan, technical_fields=("frozen_at",)
        )

    def inspect_plan(self, plan_id: str) -> dict[str, Any]:
        directory = self._safe_directory("plans", plan_id)
        self._verify_files(directory)
        return json.loads((directory / "plan.json").read_text(encoding="utf-8"))

    def save_holdout(self, record: HoldoutRecord) -> Path:
        directory = self._safe_directory("holdouts", record.holdout_id)
        directory.mkdir(parents=True, exist_ok=True)
        latest_path = directory / "latest.json"
        if latest_path.is_file():
            self._verify_files(directory)
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            current = json.loads(
                (directory / str(latest["event"])).read_text(encoding="utf-8")
            )
            current_status = str(current.get("status"))
            if current_status in {"CONSUMED", "INVALIDATED"} and record.status.value == "UNTOUCHED":
                raise RobustnessStorageError(
                    "holdout lifecycle cannot regress to UNTOUCHED"
                )
            if current_status == "INVALIDATED" and current.get("record_hash") != record.record_hash:
                raise RobustnessStorageError(
                    "invalidated holdout lifecycle is terminal"
                )
        event_name = f"{record.status.value.lower()}-{record.record_hash[:16]}.json"
        event_path = directory / event_name
        payload = self._canonical(record)
        if event_path.is_file() and event_path.read_bytes() != payload:
            raise RobustnessStorageError("holdout event hash collision")
        if not event_path.is_file():
            self._atomic_write(event_path, payload)
        latest_payload = self._canonical(
            {
                "holdout_id": record.holdout_id,
                "status": record.status.value,
                "record_hash": record.record_hash,
                "event": event_name,
            }
        )
        self._atomic_write(directory / "latest.json", latest_payload)
        files = {
            path.name: _sha256(path.read_bytes())
            for path in sorted(directory.glob("*.json"))
            if path.name != "checksums.json"
        }
        self._atomic_write(
            directory / "checksums.json",
            self._canonical(
                {
                    "algorithm": "SHA-256",
                    "schema_version": ROBUSTNESS_EXPORT_SCHEMA_VERSION,
                    "files": files,
                }
            ),
        )
        return directory

    def inspect_holdout(self, holdout_id: str) -> dict[str, Any]:
        directory = self._safe_directory("holdouts", holdout_id)
        self._verify_files(directory)
        latest = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
        return json.loads((directory / str(latest["event"])).read_text(encoding="utf-8"))

    def find_holdout_for_period(self, period: ResearchPeriod) -> dict[str, Any] | None:
        """Find the one lifecycle for a period, independent of a changed plan hash."""

        root = self.root / "holdouts"
        if not root.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        expected_raw = to_primitive(period)
        expected = {
            name: expected_raw[name]
            for name in ("classification", "start", "end")
        }
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                continue
            try:
                candidate = self.inspect_holdout(directory.name)
            except RobustnessStorageError:
                continue
            observed_raw = candidate.get("period")
            observed = (
                {
                    name: observed_raw.get(name)
                    for name in ("classification", "start", "end")
                }
                if isinstance(observed_raw, dict)
                else None
            )
            if observed == expected:
                matches.append(candidate)
        if len(matches) > 1:
            raise RobustnessStorageError(
                "multiple holdout lifecycles claim the same frozen period"
            )
        return matches[0] if matches else None

    def save_report(
        self,
        report: RobustnessReport,
        *,
        baseline: ResearchBaselineManifest,
        plan: RobustnessResearchPlan,
        readiness: PaperReadinessReport,
    ) -> Path:
        directory = self._safe_directory("reports", report.report_id)
        records = {
            "research_baseline.json": baseline,
            "robustness_plan.json": plan,
            "robustness_report.json": report,
            "historical_coverage.json": report.coverage,
            "decision_funnel.json": report.decision_funnel,
            "drawdown_episodes.json": report.drawdown_episodes,
            "pnl_concentration.json": report.concentration,
            "temporal_robustness.json": report.temporal_rows,
            "regime_robustness.json": report.regime_rows,
            "cost_robustness.json": report.cost_robustness,
            "statistical_uncertainty.json": report.uncertainty,
            "leave_one_symbol_out.json": report.leave_one_symbol_out,
            "leave_one_strategy_out.json": report.leave_one_strategy_out,
            "single_strategy_runs.json": report.single_strategy_runs,
            "paper_readiness.json": readiness,
        }
        incoming = {name: self._canonical(value) for name, value in records.items()}
        if directory.is_dir():
            self._verify_files(directory)
            for name, payload in incoming.items():
                existing = json.loads((directory / name).read_text(encoding="utf-8"))
                candidate = json.loads(payload)
                for technical in ("created_at", "frozen_at"):
                    if isinstance(existing, dict):
                        existing.pop(technical, None)
                    if isinstance(candidate, dict):
                        candidate.pop(technical, None)
                if existing != candidate:
                    raise RobustnessStorageError(
                        "report_id collision with a different immutable diagnostic"
                    )
            return directory
        directory.mkdir(parents=True, exist_ok=False)
        for name, payload in incoming.items():
            self._atomic_write(directory / name, payload)
        checksums = {
            "algorithm": "SHA-256",
            "schema_version": ROBUSTNESS_EXPORT_SCHEMA_VERSION,
            "report_hash": report.report_hash,
            "files": {name: _sha256(payload) for name, payload in sorted(incoming.items())},
        }
        self._atomic_write(directory / "checksums.json", self._canonical(checksums))
        return directory

    def inspect_report(self, report_id: str) -> dict[str, Any]:
        directory = self._safe_directory("reports", report_id)
        manifest = self._verify_files(directory)
        return {
            "schema_version": manifest["schema_version"],
            "checksums_verified": True,
            **json.loads(
                (directory / "robustness_report.json").read_text(encoding="utf-8")
            ),
            "paper_readiness": json.loads(
                (directory / "paper_readiness.json").read_text(encoding="utf-8")
            ),
        }

    def inspect_report_bundle(self, report_id: str) -> dict[str, Any]:
        """Verify and return all governance records linked to one report."""

        directory = self._safe_directory("reports", report_id)
        manifest = self._verify_files(directory)
        report = json.loads(
            (directory / "robustness_report.json").read_text(encoding="utf-8")
        )
        return {
            "schema_version": manifest["schema_version"],
            "checksums_verified": True,
            **report,
            "research_baseline": json.loads(
                (directory / "research_baseline.json").read_text(encoding="utf-8")
            ),
            "robustness_plan": json.loads(
                (directory / "robustness_plan.json").read_text(encoding="utf-8")
            ),
            "paper_readiness": json.loads(
                (directory / "paper_readiness.json").read_text(encoding="utf-8")
            ),
        }

    def latest_report_bundle_for_run(self, run_id: str) -> dict[str, Any] | None:
        latest = self.latest_for_run(run_id)
        if latest is None:
            return None
        return self.inspect_report_bundle(str(latest["report_id"]))

    def latest_for_run(self, run_id: str) -> dict[str, Any] | None:
        root = self.root / "reports"
        if not root.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                continue
            try:
                report = self.inspect_report(directory.name)
            except RobustnessStorageError:
                continue
            if report.get("run_id") == run_id:
                matches.append(report)
        return max(matches, key=lambda item: str(item.get("created_at", "")), default=None)

    def save_evidence_bundle(
        self,
        *,
        reassessment: Any,
        registry: Any,
        completeness: Any,
        operating_scenario: Any,
        readiness: Any,
    ) -> Path:
        """Store Lot 8.3 evidence without rewriting a consumed backtest/report."""

        artifact_id = str(reassessment.reassessment_id)
        directory = self._safe_directory("evidence_reassessments", artifact_id)
        records = {
            "evidence_registry.json": registry,
            "evidence_reassessment.json": reassessment,
            "economic_completeness.json": completeness,
            "paper_operating_scenario.json": operating_scenario,
            "paper_readiness_v2.json": readiness,
        }
        incoming = {name: self._canonical(value) for name, value in records.items()}
        if directory.is_dir():
            self._verify_files(directory)
            for name, payload in incoming.items():
                existing = json.loads((directory / name).read_text(encoding="utf-8"))
                candidate = json.loads(payload)
                if isinstance(existing, dict):
                    existing.pop("created_at", None)
                if isinstance(candidate, dict):
                    candidate.pop("created_at", None)
                if existing != candidate:
                    raise RobustnessStorageError(
                        "evidence reassessment ID collides with different immutable facts"
                    )
            return directory
        directory.mkdir(parents=True, exist_ok=False)
        for name, payload in incoming.items():
            self._atomic_write(directory / name, payload)
        manifest = {
            "algorithm": "SHA-256",
            "schema_version": EVIDENCE_EXPORT_SCHEMA_VERSION,
            "reassessment_hash": str(reassessment.reassessment_hash),
            "files": {
                name: _sha256(payload) for name, payload in sorted(incoming.items())
            },
        }
        self._atomic_write(directory / "checksums.json", self._canonical(manifest))
        return directory

    def inspect_evidence_bundle(self, reassessment_id: str) -> dict[str, Any]:
        directory = self._safe_directory("evidence_reassessments", reassessment_id)
        manifest = self._verify_files(directory)
        reassessment = json.loads(
            (directory / "evidence_reassessment.json").read_text(encoding="utf-8")
        )
        return {
            "schema_version": manifest["schema_version"],
            "checksums_verified": True,
            **reassessment,
            "evidence_registry": json.loads(
                (directory / "evidence_registry.json").read_text(encoding="utf-8")
            ),
            "economic_completeness": json.loads(
                (directory / "economic_completeness.json").read_text(encoding="utf-8")
            ),
            "paper_operating_scenario": json.loads(
                (directory / "paper_operating_scenario.json").read_text(encoding="utf-8")
            ),
            "paper_readiness_v2": json.loads(
                (directory / "paper_readiness_v2.json").read_text(encoding="utf-8")
            ),
        }

    def latest_evidence_for_run(self, run_id: str) -> dict[str, Any] | None:
        root = self.root / "evidence_reassessments"
        if not root.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                continue
            try:
                item = self.inspect_evidence_bundle(directory.name)
            except RobustnessStorageError:
                continue
            if item.get("run_id") == run_id:
                matches.append(item)
        return max(matches, key=lambda item: str(item.get("created_at", "")), default=None)

    def save_recomputation_bundle(
        self,
        *,
        report: Any,
        readiness: Any,
        human_review: Any,
        evidence_registry: Any,
    ) -> Path:
        """Write schema 1.9 beside, never into, the original backtest export."""

        artifact_id = str(report.recomputation_id)
        directory = self._safe_directory("economic_recomputations", artifact_id)
        records = {
            "economic_recomputation.json": report,
            "decision_invariance.json": report.decision_invariance,
            "affected_fills.json": report.affected_fills,
            "recomputed_trades.json": report.affected_trades,
            "recomputed_equity.json": report.recomputed_equity,
            "economic_completeness.json": report.completeness,
            "paper_operating_scenario.json": report.operating,
            "paper_readiness_v3.json": readiness,
            "human_review.json": human_review,
            "evidence_registry.json": evidence_registry,
        }
        incoming = {name: self._canonical(value) for name, value in records.items()}
        if directory.is_dir():
            self._verify_files(directory)
            for name, payload in incoming.items():
                existing = json.loads((directory / name).read_text(encoding="utf-8"))
                candidate = json.loads(payload)
                for technical in ("created_at", "recorded_at", "acquired_at"):
                    if isinstance(existing, dict):
                        existing.pop(technical, None)
                    if isinstance(candidate, dict):
                        candidate.pop(technical, None)
                if existing != candidate:
                    raise RobustnessStorageError(
                        "recomputation ID collides with different immutable economics"
                    )
            return directory
        directory.mkdir(parents=True, exist_ok=False)
        for name, payload in incoming.items():
            self._atomic_write(directory / name, payload)
        manifest = {
            "algorithm": "SHA-256",
            "schema_version": ECONOMIC_RECOMPUTATION_SCHEMA_VERSION,
            "recomputation_hash": str(report.recomputation_hash),
            "files": {
                name: _sha256(payload) for name, payload in sorted(incoming.items())
            },
        }
        self._atomic_write(directory / "checksums.json", self._canonical(manifest))
        return directory

    def inspect_recomputation_bundle(self, recomputation_id: str) -> dict[str, Any]:
        directory = self._safe_directory("economic_recomputations", recomputation_id)
        manifest = self._verify_files(directory)
        report = json.loads(
            (directory / "economic_recomputation.json").read_text(encoding="utf-8")
        )
        readiness = json.loads(
            (directory / "paper_readiness_v3.json").read_text(encoding="utf-8")
        )
        latest_human = self.latest_human_review(str(readiness["readiness_id"]))
        if latest_human is None:
            latest_human = json.loads(
                (directory / "human_review.json").read_text(encoding="utf-8")
            )
        return {
            "schema_version": manifest["schema_version"],
            "checksums_verified": True,
            **report,
            "decision_invariance": json.loads(
                (directory / "decision_invariance.json").read_text(encoding="utf-8")
            ),
            "economic_completeness": json.loads(
                (directory / "economic_completeness.json").read_text(encoding="utf-8")
            ),
            "paper_operating_scenario": json.loads(
                (directory / "paper_operating_scenario.json").read_text(encoding="utf-8")
            ),
            "paper_readiness_v3": readiness,
            "human_review": latest_human,
            "evidence_registry": json.loads(
                (directory / "evidence_registry.json").read_text(encoding="utf-8")
            ),
        }

    def latest_recomputation_for_run(self, run_id: str) -> dict[str, Any] | None:
        root = self.root / "economic_recomputations"
        if not root.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                continue
            try:
                item = self.inspect_recomputation_bundle(directory.name)
            except RobustnessStorageError:
                continue
            if item.get("original_run_id") == run_id:
                matches.append(item)
        return max(matches, key=lambda item: str(item.get("created_at", "")), default=None)

    def find_recomputation_by_readiness(self, readiness_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(readiness_id):
            raise RobustnessStorageError("invalid Paper readiness identifier")
        root = self.root / "economic_recomputations"
        if root.is_dir():
            for directory in sorted(root.iterdir()):
                if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                    continue
                item = self.inspect_recomputation_bundle(directory.name)
                if item["paper_readiness_v3"].get("readiness_id") == readiness_id:
                    return item
        raise RobustnessStorageError("Paper readiness V3 artifact not found")

    def save_human_review(self, record: Any) -> Path:
        return self._save_single(
            "human_reviews",
            str(record.review_event_id),
            "human_review.json",
            record,
            technical_fields=("recorded_at",),
        )

    def latest_human_review(self, readiness_id: str) -> dict[str, Any] | None:
        if not _SAFE_ID.fullmatch(readiness_id):
            raise RobustnessStorageError("invalid Paper readiness identifier")
        root = self.root / "human_reviews"
        if not root.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                continue
            try:
                self._verify_files(directory)
                item = json.loads(
                    (directory / "human_review.json").read_text(encoding="utf-8")
                )
            except (RobustnessStorageError, OSError, ValueError, json.JSONDecodeError):
                continue
            if item.get("readiness_id") == readiness_id:
                matches.append(item)
        return max(matches, key=lambda item: str(item.get("recorded_at", "")), default=None)

    @staticmethod
    def _verify_files(directory: Path) -> dict[str, Any]:
        try:
            manifest = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
            if manifest.get("schema_version") not in _SUPPORTED_ROBUSTNESS_SCHEMAS:
                raise RobustnessStorageError("unsupported robustness export schema")
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                raise RobustnessStorageError("invalid robustness checksum manifest")
            for name, expected in files.items():
                if Path(name).name != name or not isinstance(expected, str) or len(expected) != 64:
                    raise RobustnessStorageError("invalid robustness checksum entry")
                path = directory / name
                if not path.is_file() or _sha256(path.read_bytes()) != expected:
                    raise RobustnessStorageError(
                        f"robustness artifact checksum mismatch: {name}"
                    )
            return manifest
        except RobustnessStorageError:
            raise
        except FileNotFoundError as exc:
            raise RobustnessStorageError("robustness artifact not found") from exc
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RobustnessStorageError("invalid robustness artifact") from exc
