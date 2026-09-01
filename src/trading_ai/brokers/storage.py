"""Tamper-evident local Paper session store under ``data_local/paper``."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from trading_ai.brokers.exceptions import BrokerIntegrityError
from trading_ai.brokers.models import PaperSessionManifest
from trading_ai.core.hashing import stable_hash, to_primitive


PAPER_STORE_SCHEMA_VERSION = "1.0"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class LocalPaperStore:
    def __init__(self, root: Path | str = Path("data_local/paper")) -> None:
        self.root = Path(root)

    def _session(self, session_id: str) -> Path:
        if not _SAFE_ID.fullmatch(session_id):
            raise BrokerIntegrityError("invalid Paper session identifier")
        root = self.root.resolve()
        target = (self.root / session_id).resolve()
        if root not in target.parents:
            raise BrokerIntegrityError("Paper session path escapes local store")
        return target

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(
            to_primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @staticmethod
    def _atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def create_session(self, manifest: PaperSessionManifest) -> Path:
        directory = self._session(manifest.session_id)
        payload = self._canonical(manifest)
        target = directory / "session_manifest.json"
        if target.is_file():
            self.verify(manifest.session_id)
            if target.read_bytes() != payload:
                raise BrokerIntegrityError("Paper session manifest is immutable")
            return directory
        directory.mkdir(parents=True, exist_ok=False)
        self._atomic(target, payload)
        self._refresh_checksums(directory)
        return directory

    def append(self, session_id: str, category: str, value: Any, *, record_id: str) -> Path:
        if not _SAFE_ID.fullmatch(category) or not _SAFE_ID.fullmatch(record_id):
            raise BrokerIntegrityError("invalid Paper record identifier")
        directory = self._session(session_id)
        if not (directory / "session_manifest.json").is_file():
            raise BrokerIntegrityError("Paper session manifest must exist before records")
        self.verify(session_id)
        payload = self._canonical(value)
        target = directory / category / f"{record_id}.json"
        if target.is_file() and target.read_bytes() != payload:
            raise BrokerIntegrityError("immutable Paper record collision")
        if not target.is_file():
            self._atomic(target, payload)
            self._refresh_checksums(directory)
        return target

    def _refresh_checksums(self, directory: Path) -> None:
        files = {
            str(path.relative_to(directory)).replace("\\", "/"): _sha256(path.read_bytes())
            for path in sorted(directory.rglob("*.json"))
            if path.name != "checksums.json"
        }
        manifest = {
            "algorithm": "SHA-256",
            "schema_version": PAPER_STORE_SCHEMA_VERSION,
            "files": files,
        }
        self._atomic(directory / "checksums.json", self._canonical(manifest))

    def verify(self, session_id: str) -> dict[str, Any]:
        directory = self._session(session_id)
        try:
            manifest = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
            if manifest["schema_version"] != PAPER_STORE_SCHEMA_VERSION:
                raise BrokerIntegrityError("unsupported Paper store schema")
            for name, expected in manifest["files"].items():
                path = (directory / name).resolve()
                if directory.resolve() not in path.parents or not path.is_file():
                    raise BrokerIntegrityError("Paper checksum references an invalid path")
                if _sha256(path.read_bytes()) != expected:
                    raise BrokerIntegrityError(f"Paper artifact checksum mismatch: {name}")
            actual = {
                str(path.relative_to(directory)).replace("\\", "/")
                for path in directory.rglob("*.json")
                if path.name != "checksums.json"
            }
            if actual != set(manifest["files"]):
                raise BrokerIntegrityError("Paper store contains unmanifested JSON artifacts")
            return manifest
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, BrokerIntegrityError):
                raise
            raise BrokerIntegrityError(f"invalid Paper store: {exc}") from exc

    def list_sessions(self) -> tuple[dict[str, Any], ...]:
        if not self.root.is_dir():
            return ()
        rows = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or not _SAFE_ID.fullmatch(directory.name):
                continue
            try:
                self.verify(directory.name)
                manifest = json.loads((directory / "session_manifest.json").read_text(encoding="utf-8"))
                rows.append(
                    {
                        "session_id": directory.name,
                        "mode": manifest["mode"],
                        "account_masked": manifest["account_masked"],
                        "paper_execution_armed": manifest["paper_execution_armed"],
                        "integrity": "VERIFIED",
                    }
                )
            except BrokerIntegrityError:
                rows.append({"session_id": directory.name, "integrity": "ERROR"})
        return tuple(rows)

    def inspect(self, session_id: str) -> dict[str, Any]:
        self.verify(session_id)
        directory = self._session(session_id)
        manifest = json.loads((directory / "session_manifest.json").read_text(encoding="utf-8"))
        records: dict[str, list[dict[str, Any]]] = {}
        for category in (
            "events", "orders", "executions", "commissions", "snapshots",
            "reconciliation", "decisions", "outcomes", "audits",
        ):
            category_path = directory / category
            records[category] = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(category_path.glob("*.json"))
            ] if category_path.is_dir() else []
        return {
            "schema_version": PAPER_STORE_SCHEMA_VERSION,
            "integrity": "VERIFIED",
            "session": manifest,
            **records,
            "evidence_hash": stable_hash({"manifest": manifest, **records}),
        }
