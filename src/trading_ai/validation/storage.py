"""Tamper-evident local validation report storage below data_local/."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from trading_ai.core.hashing import to_primitive
from trading_ai.validation.exceptions import ValidationError
from trading_ai.validation.models import ValidationReport


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class LocalValidationStore:
    def __init__(self, root: Path | str = Path("data_local/validation")) -> None:
        self.root = Path(root)

    def _directory(self, validation_id: str) -> Path:
        if not _SAFE_ID.fullmatch(validation_id):
            raise ValidationError("invalid validation_id")
        return self.root / validation_id

    def save(self, report: ValidationReport) -> Path:
        directory = self._directory(report.validation_id)
        report_path = directory / "report.json"
        incoming = to_primitive(report)
        if report_path.is_file():
            existing = self.inspect(report.validation_id)
            # created_at is technical metadata excluded from the semantic
            # validation ID. Repeating the same evaluation is idempotent and
            # must not rewrite the first immutable artifact.
            comparable_existing = dict(existing)
            comparable_incoming = dict(incoming)
            comparable_existing.pop("created_at", None)
            comparable_incoming.pop("created_at", None)
            if comparable_existing != comparable_incoming:
                raise ValidationError(
                    "validation_id collision with a different immutable report"
                )
            return directory
        directory.mkdir(parents=True, exist_ok=False)
        payload = json.dumps(incoming, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        report_path.write_text(payload, encoding="utf-8")
        checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        (directory / "checksums.json").write_text(
            json.dumps({"algorithm": "SHA-256", "report.json": checksum}, sort_keys=True),
            encoding="utf-8",
        )
        return directory

    def inspect(self, validation_id: str) -> dict:
        directory = self._directory(validation_id)
        try:
            payload = (directory / "report.json").read_bytes()
            checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValidationError(f"validation report not found: {validation_id}") from exc
        if hashlib.sha256(payload).hexdigest() != checksums.get("report.json"):
            raise ValidationError("validation report checksum mismatch")
        return json.loads(payload)

    def latest_for_run(self, run_id: str) -> dict | None:
        if not self.root.is_dir():
            return None
        matches = []
        for path in self.root.iterdir():
            if path.is_dir() and _SAFE_ID.fullmatch(path.name):
                try:
                    payload = self.inspect(path.name)
                except ValidationError:
                    continue
                if payload.get("run_id") == run_id:
                    matches.append(payload)
        return max(matches, key=lambda item: item.get("created_at", ""), default=None)
