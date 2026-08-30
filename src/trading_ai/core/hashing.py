"""Dependency-neutral canonical serialization and SHA-256 helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping


def to_primitive(value: Any) -> Any:
    """Convert typed immutable domain values into canonical JSON primitives."""

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
