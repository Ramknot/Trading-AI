from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_ai.monitoring.costs import build_cost_snapshot
from trading_ai.monitoring.models import (
    CostComponent,
    CostCoverageStatus,
    CostKnowledge,
    HealthComponent,
    HealthSnapshot,
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSnapshot,
    SystemStatus,
)
from trading_ai.monitoring.store import SQLiteMonitoringStore


NOW = datetime(2026, 8, 29, 8, tzinfo=timezone.utc)


def _event() -> MonitoringEvent:
    return MonitoringEvent(
        event_id="evt-001",
        timestamp=NOW,
        event_type=MonitoringEventType.RISK_DECISION,
        run_id="bt-test",
        session_id="bt-test",
        source_component="BalancedRiskEngine",
        component_version="1.0",
        related_ids=(("order_id", "order-001"),),
        provenance=(("config_hash", "a" * 64),),
        payload_json='{"status":"REJECT"}',
        symbol="AAPL",
        strategy_name="trend",
        status="REJECT",
    )


def test_monitoring_event_is_immutable_utc_and_keeps_lineage() -> None:
    event = _event()
    assert event.payload == {"status": "REJECT"}
    assert event.related_ids == (("order_id", "order-001"),)
    with pytest.raises(FrozenInstanceError):
        event.status = "APPROVE"  # type: ignore[misc]
    with pytest.raises(ValueError, match="timezone-aware"):
        MonitoringEvent(
            event_id="evt-naive",
            timestamp=datetime(2026, 1, 1),
            event_type=MonitoringEventType.FILL,
            run_id="bt-test",
            session_id="bt-test",
            source_component="execution",
            component_version="1",
        )
    with pytest.raises(ValueError, match="normalized to UTC"):
        MonitoringEvent(
            event_id="evt-offset",
            timestamp=NOW.astimezone(timezone(timedelta(hours=2))),
            event_type=MonitoringEventType.FILL,
            run_id="bt-test",
            session_id="bt-test",
            source_component="execution",
            component_version="1",
        )


def test_unavailable_cost_can_never_be_silently_encoded_as_zero() -> None:
    unavailable = CostComponent.unavailable("not modeled")
    assert unavailable.status is CostKnowledge.UNAVAILABLE
    assert unavailable.amount is None
    with pytest.raises(ValueError, match="must not carry"):
        CostComponent(CostKnowledge.UNAVAILABLE, Decimal("0"))
    with pytest.raises(ValueError, match="need a non-negative amount"):
        CostComponent(CostKnowledge.KNOWN, None)


def test_backtest_cost_snapshot_reuses_modeled_costs_and_reports_incomplete() -> None:
    snapshot = build_cost_snapshot(
        {
            "run_id": "bt-test",
            "initial_cash": "1000",
            "final_equity": "1010",
            "metrics": {
                "total_commission": "2",
                "total_spread_cost": "1",
                "total_slippage_cost": "1",
            },
        },
        NOW,
    )
    assert snapshot.trading.commission.amount == Decimal("2")
    assert snapshot.trading.transaction_tax.status is CostKnowledge.UNAVAILABLE
    assert snapshot.trading.fx_cost.amount is None
    assert snapshot.trading.total_variable_cost.status is CostKnowledge.UNAVAILABLE
    assert snapshot.coverage_status is CostCoverageStatus.INCOMPLETE
    assert snapshot.gross_pnl == Decimal("14")
    assert snapshot.net_pnl_known == Decimal("10")
    assert snapshot.net_pnl_estimated is None


def test_explicit_estimated_costs_are_displayable_without_ui_tariffs() -> None:
    summary = {
        "run_id": "bt-costs",
        "initial_cash": "1000",
        "final_equity": "1010",
        "metrics": {
            "total_commission": "2",
            "total_spread_cost": "1",
            "total_slippage_cost": "1",
        },
        "trading_costs": {
            name: {"status": "ESTIMATED", "amount": "1", "source": "fixture"}
            for name in (
                "exchange_fees", "transaction_tax", "fx_cost",
                "financing_cost", "other_variable_cost",
            )
        },
        "operating_costs": {
            name: {"status": "KNOWN", "amount": "1", "source": "fixture"}
            for name in (
                "market_data_subscription", "server_vps",
                "software_subscriptions", "other_fixed_cost",
            )
        },
    }
    snapshot = build_cost_snapshot(summary, NOW)
    assert snapshot.coverage_status is CostCoverageStatus.COMPLETE
    assert snapshot.trading.transaction_tax.status is CostKnowledge.ESTIMATED
    assert snapshot.operating.server_vps.status is CostKnowledge.KNOWN
    assert snapshot.trading.total_variable_cost.amount == Decimal("9")
    assert snapshot.net_pnl_estimated == Decimal("1")


def test_health_snapshot_never_labels_absent_component_healthy() -> None:
    health = HealthSnapshot(
        run_id="bt-test",
        timestamp=NOW,
        status=SystemStatus.WARNING,
        components=(
            HealthComponent("ML", SystemStatus.UNAVAILABLE, "disabled", NOW),
            HealthComponent("Risk", SystemStatus.HEALTHY, "available", NOW),
        ),
    )
    assert health.components[0].status is SystemStatus.UNAVAILABLE
    with pytest.raises(ValueError, match="sorted"):
        HealthSnapshot(
            "bt-test",
            NOW,
            SystemStatus.WARNING,
            tuple(reversed(health.components)),
        )


def test_sqlite_monitoring_store_round_trip_is_local_and_deterministic(tmp_path) -> None:
    store = SQLiteMonitoringStore(tmp_path / "monitoring" / "events.db")
    event = _event()
    store.append_event(event)
    assert store.list_events("bt-test") == (event,)
    assert store.list_events("bt-test", status="REJECT") == (event,)
    assert store.list_events("bt-test", symbol="MSFT") == ()
    store.append_event(event)
    assert store.list_events("bt-test") == (event,)
    conflicting = MonitoringEvent(
        event_id=event.event_id,
        timestamp=event.timestamp,
        event_type=event.event_type,
        run_id=event.run_id,
        session_id=event.session_id,
        source_component=event.source_component,
        component_version=event.component_version,
        related_ids=event.related_ids,
        provenance=event.provenance,
        payload_json='{"status":"APPROVE"}',
        symbol=event.symbol,
        strategy_name=event.strategy_name,
        status="APPROVE",
    )
    with pytest.raises(Exception, match="immutable and unique"):
        store.append_event(conflicting)

    snapshot = MonitoringSnapshot(
        snapshot_id="mon-001",
        run_id="bt-test",
        timestamp=NOW,
        mode="BACKTEST",
        status=SystemStatus.WARNING,
        source_schema_version="1.5",
        source_fingerprint="b" * 64,
        sections_json='{"overview":{"mode":"BACKTEST"}}',
    )
    store.save_snapshot(snapshot)
    assert store.load_snapshot("mon-001") == snapshot
    assert store.load_snapshot("missing") is None
    assert store.is_healthy() is True
