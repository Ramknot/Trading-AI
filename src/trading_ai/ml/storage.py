"""Safe local filesystem primitives for the Git-ignored ML registry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from trading_ai.backtesting.reproducibility import to_primitive
from trading_ai.ml.exceptions import MLRegistryError


SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_child(root: Path, *parts: str) -> Path:
    if any(not SAFE_MODEL_ID.fullmatch(part) for part in parts):
        raise MLRegistryError("invalid registry identifier")
    resolved_root = root.resolve()
    path = resolved_root.joinpath(*parts).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise MLRegistryError("registry path escapes its root")
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_primitive(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MLRegistryError(f"invalid registry metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise MLRegistryError("registry metadata must contain a JSON object")
    return value
