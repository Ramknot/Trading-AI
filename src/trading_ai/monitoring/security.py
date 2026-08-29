"""Defensive redaction for observability payloads rendered to local clients."""

from __future__ import annotations

from typing import Any


_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "credential",
    "api_key",
    "private_key",
    "authorization",
    "access_token",
    "refresh_token",
)


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            redacted[str(key)] = (
                "[REDACTED]"
                if any(part in normalized for part in _SENSITIVE_KEY_PARTS)
                else redact_sensitive(item)
            )
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value
