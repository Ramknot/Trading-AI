"""Canonical serialization, stable identifiers, and optional Git provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: to_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): to_primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        to_primitive(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_result_hash(value: Any) -> str:
    payload = dict(to_primitive(value))
    payload.pop("created_at", None)
    payload.pop("result_hash", None)
    return stable_hash(payload)


def source_tree_hash(project_root: Path) -> str:
    """Hash Python source bytes and relative paths for dirty-tree traceability."""

    digest = hashlib.sha256()
    source_root = project_root / "src" / "trading_ai"
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def detect_git_commit(project_root: Path) -> str | None:
    git_directory = project_root / ".git"
    try:
        if git_directory.is_file():
            pointer = git_directory.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir:"):
                git_directory = (project_root / pointer.split(":", 1)[1].strip()).resolve()
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
        if len(head) == 40 and all(character in "0123456789abcdef" for character in head.lower()):
            return head
        if head.startswith("ref: "):
            reference = head[5:]
            reference_path = git_directory / reference
            if reference_path.is_file():
                commit = reference_path.read_text(encoding="utf-8").strip()
                if len(commit) == 40:
                    return commit
            packed_refs = git_directory / "packed-refs"
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if not line.startswith(("#", "^")) and line.endswith(
                        f" {reference}"
                    ):
                        commit = line.split(" ", 1)[0]
                        if len(commit) == 40:
                            return commit
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None
