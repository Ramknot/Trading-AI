import json
import logging

from trading_ai.core.logging import JsonFormatter


def test_json_log_contains_monitoring_fields() -> None:
    record = logging.LogRecord(
        name="trading_ai.risk",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="order rejected",
        args=(),
        exc_info=None,
    )
    record.component = "risk"  # type: ignore[attr-defined]
    record.environment = "PAPER"  # type: ignore[attr-defined]
    record.profile = "balanced"  # type: ignore[attr-defined]
    record.decision_id = "deny-all:order-1"  # type: ignore[attr-defined]

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "WARNING"
    assert payload["component"] == "risk"
    assert payload["environment"] == "PAPER"
    assert payload["profile"] == "balanced"
    assert payload["decision_id"] == "deny-all:order-1"
    assert payload["timestamp"].endswith("+00:00")
