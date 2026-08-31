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

    @staticmethod
    def _verify_files(directory: Path) -> dict[str, Any]:
        try:
            manifest = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
            if manifest.get("schema_version") != ROBUSTNESS_EXPORT_SCHEMA_VERSION:
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
