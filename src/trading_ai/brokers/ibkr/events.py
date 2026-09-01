"""IBKR callback normalization helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading_ai.brokers.ibkr.errors import normalize_ibkr_error
from trading_ai.brokers.models import BrokerEvent, BrokerEventType
from trading_ai.core.hashing import stable_hash


_EVENT_MAP = {
    "CONNECTED": BrokerEventType.CONNECTED,
    "DISCONNECTED": BrokerEventType.DISCONNECTED,
    "RECONNECTING": BrokerEventType.RECONNECTING,
    "ACCOUNT_IDENTIFIERS": BrokerEventType.ACCOUNT,
    "ACCOUNT_SUMMARY": BrokerEventType.ACCOUNT,
    "POSITION": BrokerEventType.POSITIONS,
    "OPEN_ORDER": BrokerEventType.ORDER_ACK,
    "ORDER_SUBMITTED": BrokerEventType.ORDER_SUBMITTED,
    "COMPLETED_ORDER": BrokerEventType.ORDER_STATUS,
    "ORDER_STATUS": BrokerEventType.ORDER_STATUS,
    "EXECUTION": BrokerEventType.FILL,
    "COMMISSION_REPORT": BrokerEventType.COMMISSION_REPORT,
    "ERROR": BrokerEventType.ERROR,
}


def parse_ibkr_timestamp(value: object) -> datetime | None:
    """Parse documented execution timestamps without guessing an absent zone.

    TWS may return ``YYYYMMDD-HH:MM:SS`` followed by an IANA zone.  A timestamp
    without a zone remains unavailable rather than being silently interpreted
    in the workstation's local timezone.
    """

    raw = str(value or "").strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) != 2:
        return None
    try:
        local = datetime.strptime(parts[0], "%Y%m%d-%H:%M:%S")
        aware = local.replace(tzinfo=ZoneInfo(parts[1]))
    except (ValueError, ZoneInfoNotFoundError):
        return None
    return aware.astimezone(timezone.utc)


def normalize_callback_event(
    *,
    session_id: str,
    kind: str,
    payload: dict[str, Any],
    source_version: str,
    received_at: datetime | None = None,
) -> BrokerEvent:
    received = received_at or datetime.now(timezone.utc)
    clean = dict(payload)
    event_type = _EVENT_MAP.get(kind, BrokerEventType.WARNING)
    if kind == "ORDER_STATUS":
        status = str(clean.get("status", ""))
        try:
            filled = Decimal(str(clean.get("filled", "0")))
            remaining = Decimal(str(clean.get("remaining", "0")))
        except InvalidOperation:
            filled = Decimal("0")
            remaining = Decimal("0")
        if status in {"Cancelled", "ApiCancelled", "PendingCancel"}:
            event_type = BrokerEventType.ORDER_CANCEL
        elif status == "Inactive":
            event_type = BrokerEventType.ORDER_REJECT
        elif status == "Filled" or (filled > Decimal("0") and remaining == Decimal("0")):
            event_type = BrokerEventType.FILL
        elif filled > Decimal("0"):
            event_type = BrokerEventType.PARTIAL_FILL
        elif status in {"Submitted", "PreSubmitted"}:
            event_type = BrokerEventType.ORDER_ACK
    elif kind == "EXECUTION":
        if clean.get("correction_of"):
            event_type = BrokerEventType.EXECUTION_CORRECTION
        elif clean.get("is_partial") is True:
            event_type = BrokerEventType.PARTIAL_FILL
        else:
            event_type = BrokerEventType.FILL
    if kind == "ERROR":
        normalized = normalize_ibkr_error(int(clean.get("code", -1)))
        clean = {
            "code": normalized.code,
            "stable_code": normalized.stable_code,
            "severity": normalized.severity.value,
            "connectivity_lost": normalized.connectivity_lost,
            "reconciliation_required": normalized.reconciliation_required,
        }
        event_type = (
            BrokerEventType.ERROR
            if normalized.severity.value in {"REJECT", "CRITICAL", "CONNECTIVITY"}
            else BrokerEventType.WARNING
        )
    related = tuple(
        sorted(
            (key, str(clean[key]))
            for key in (
                "broker_order_id", "client_order_key", "exec_id", "perm_id", "stable_code"
            )
            if clean.get(key) not in (None, "")
        )
    )
    identity = {
        "session_id": session_id,
        "kind": kind,
        "payload": clean,
        "received_at": received,
    }
    return BrokerEvent(
        event_id="broker-event-" + stable_hash(identity)[:24],
        session_id=session_id,
        event_type=event_type,
        received_at=received,
        source="IBKR_TWS_API",
        source_version=source_version,
        broker_timestamp=parse_ibkr_timestamp(clean.get("time")),
        related_ids=related,
        payload_json=json.dumps(clean, sort_keys=True, separators=(",", ":")),
    )
