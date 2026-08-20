"""Small JSON logging setup ready for future log aggregation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format stable structured fields while keeping secrets out by design."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "environment": getattr(record, "environment", None),
            "profile": getattr(record, "profile", None),
            "message": record.getMessage(),
        }
        decision_id = getattr(record, "decision_id", None)
        if decision_id is not None:
            payload["decision_id"] = decision_id
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger with one JSON stream handler."""

    logger = logging.getLogger("trading_ai")
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(handler, "_trading_ai_json", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._trading_ai_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger
