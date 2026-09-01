from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trading_ai.brokers.exceptions import (
    BrokerConfigurationError,
    BrokerUnavailableError,
    PaperAccountGuardError,
    PaperExecutionLockedError,
    ReconciliationRequiredError,
)
from trading_ai.brokers.fake import FakeBroker, FakeBrokerScenario
from trading_ai.brokers.models import (
    BrokerAccountIdentity,
    BrokerConnectionState,
    BrokerEnvironment,
    BrokerExecution,
    BrokerOrderRecord,
    BrokerOrderState,
    BrokerPosition,
    CommissionKnowledge,
    DataFreshnessStatus,
    PaperEmergencyHalt,
    PaperMode,
    PaperSafetySnapshot,
    PaperSessionState,
    ReconciliationStatus,
)
from trading_ai.brokers.paper_guard import (
    PaperAccountGuard,
    PaperExecutionBoundary,
    PaperSubmissionAuthorization,
)
from trading_ai.brokers.reconciliation import BrokerReconciler, ReconciliationState
from trading_ai.brokers.recovery import PaperRecoveryService
from trading_ai.brokers.replay import PaperShadowAudit
from trading_ai.brokers.reporting import build_expected_observed_metrics
from trading_ai.brokers.session import PaperTradingSession
from trading_ai.brokers.ledger import PaperLedger
from trading_ai.brokers.state import BrokerOrderStateMachine
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.core.hashing import stable_hash
from trading_ai.core.models import (
    ExecutionEnvironment,
    ExecutionStatus,
    OrderRequest,
    OrderSide,
    PortfolioSnapshot,
    RiskDecision,
    RiskDecisionStatus,
    TradingContext,
    TradingProfileName,
)
from trading_ai.execution.base import ExecutionEngine
from trading_ai.risk.base import RiskEngine


NOW = datetime(2026, 8, 31, 12, tzinfo=timezone.utc)
ACCOUNT_HASH = "a" * 64


def paper_account(*, environment=BrokerEnvironment.PAPER, verified=True, digest=ACCOUNT_HASH):
    return BrokerAccountIdentity(
        broker="IBKR",
        account_hash=digest,
        account_masked="IBKR-****0001",
        environment=environment,
        base_currency="USD",
        capabilities=("ACCOUNT", "ORDERS", "POSITIONS"),
        environment_verified=verified,
        environment_evidence="TEST_ATTESTATION",
    )


def approved_order(quantity=Decimal("10"), *, side=OrderSide.BUY):
    order = OrderRequest(
        order_id="order-1",
        symbol="AAPL",
        side=side,
        quantity=quantity,
        created_at=NOW,
        cost_estimate_id="cost-1",
        economic_decision_id="economic-1",
        estimated_cash_requirement=Decimal("1001"),
        estimated_unit_cash_requirement=Decimal("100.1"),
    )
    decision = RiskDecision(
        decision_id="risk-1",
        order_id=order.order_id,
        status=RiskDecisionStatus.APPROVE,
        reason="fixture approval",
        risk_engine="BalancedRiskEngine",
        timestamp=NOW,
        engine_version="1.0",
        requested_quantity=quantity,
        approved_quantity=quantity,
    )
    from trading_ai.core.models import RiskApprovedOrder

    return RiskApprovedOrder(order, decision)


def safety(
    *,
    mode=PaperMode.PAPER_EXECUTION_ARMED,
    account=None,
    state=PaperSessionState.READY,
    reconciliation=ReconciliationStatus.IN_SYNC,
    freshness=DataFreshnessStatus.FRESH,
    connection=BrokerConnectionState.CONNECTED,
    halted=False,
):
    return PaperSafetySnapshot(
        session_id="paper-session-1",
        observed_at=NOW,
        mode=mode,
        session_state=state,
        account=account or paper_account(),
        connection_state=connection,
        reconciliation_status=reconciliation,
        reconciliation_hash="b" * 64,
        data_freshness=freshness,
        risk_health_ok=True,
        market_session_open=True,
        configs_frozen=True,
        emergency_halt=PaperEmergencyHalt(halted, "TEST", NOW),
    )


class TestAuthorization(PaperSubmissionAuthorization):
    __test__ = False

    def allows(self, snapshot, order) -> bool:
        del snapshot, order
        return True


class ApprovingRisk(RiskEngine):
    def evaluate(self, order, portfolio, context):
        del context
        return RiskDecision(
            decision_id="risk-1",
            order_id=order.order_id,
            status=RiskDecisionStatus.APPROVE,
            reason="fixture approval",
            risk_engine="BalancedRiskEngine",
            timestamp=portfolio.as_of,
            engine_version="1.0",
            requested_quantity=order.quantity,
            approved_quantity=order.quantity,
        )


def test_lot9_boundary_defaults_to_deny_even_with_ready_paper_state() -> None:
    fake = FakeBroker(session_id="paper-session-1", account=paper_account())
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
    )
    with pytest.raises(PaperExecutionLockedError, match="no authorization issuer"):
        boundary.submit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )
    assert fake.transmission_count == 0


@pytest.mark.parametrize(
    "snapshot, message",
    [
        (safety(mode=PaperMode.PAPER_READ_ONLY), "not armed"),
        (safety(state=PaperSessionState.DEGRADED), "not READY"),
        (safety(reconciliation=ReconciliationStatus.DRIFT), "not IN_SYNC"),
        (safety(freshness=DataFreshnessStatus.STALE), "stale"),
        (safety(connection=BrokerConnectionState.DISCONNECTED), "not connected"),
        (safety(halted=True), "emergency halt"),
    ],
)
def test_boundary_fails_closed_for_every_unsafe_session_condition(snapshot, message) -> None:
    fake = FakeBroker(session_id="paper-session-1", account=paper_account())
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=lambda: snapshot,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    with pytest.raises(PaperExecutionLockedError, match=message):
        boundary.submit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )


@pytest.mark.parametrize(
    "account",
    [
        paper_account(environment=BrokerEnvironment.LIVE, verified=False),
        paper_account(environment=BrokerEnvironment.UNKNOWN, verified=False),
        paper_account(verified=False),
        paper_account(digest="c" * 64),
    ],
)
def test_live_unknown_unverified_and_non_allowlisted_accounts_never_submit(account) -> None:
    fake = FakeBroker(session_id="paper-session-1", account=account)
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=lambda: safety(account=account),
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    with pytest.raises(PaperAccountGuardError):
        boundary.submit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )
    assert fake.transmission_count == 0


def test_fake_happy_path_preserves_risk_quantity_partial_fills_and_commissions() -> None:
    fake = FakeBroker(
        session_id="paper-session-1",
        account=paper_account(),
        scenario=FakeBrokerScenario(
            partial_fill_quantities=(Decimal("4"), Decimal("6")),
            duplicate_execution_callbacks=True,
            commission=Decimal("0.75"),
        ),
    )
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    order = approved_order().order
    engine = ExecutionEngine(boundary, ApprovingRisk())
    result = engine.submit_order(
        order,
        PortfolioSnapshot(NOW, Decimal("100000"), Decimal("100000")),
        TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
    )
    assert result.status is ExecutionStatus.SUBMITTED
    assert fake.transmission_count == 1
    assert [item.quantity for item in fake.executions()] == [Decimal("4"), Decimal("6")]
    assert sum(item.quantity for item in fake.executions()) == order.quantity
    assert len(fake.commission_reports) == 2
    assert fake.completed_orders()[0].state is BrokerOrderState.FILLED
    account = fake.account_snapshot()
    assert account.cash == Decimal("98998.50")
    assert account.positions[0].quantity == Decimal("10")
    local = fake.sync_state()
    reconciliation = BrokerReconciler().reconcile(
        session_id="paper-session-1",
        observed_at=NOW,
        local=local,
        broker=fake.sync_state(),
    )
    assert reconciliation.status is ReconciliationStatus.IN_SYNC
    metrics = build_expected_observed_metrics(
        order=fake.completed_orders()[0],
        executions=fake.executions(),
        commissions=fake.commission_reports,
        decision_at=NOW,
        submitted_at=NOW,
        acknowledged_at=NOW,
        expected_fill_price=Decimal("99"),
        estimated_slippage=Decimal("1"),
        estimated_commission=Decimal("1.40"),
    )
    assert metrics.observed_average_fill_price == Decimal("100")
    assert metrics.observed_slippage == Decimal("1")
    assert metrics.broker_commission == Decimal("1.50")
    assert metrics.partial_fills == 1


def test_fake_broker_event_and_execution_replay_is_deterministic() -> None:
    def run() -> str:
        fake = FakeBroker(
            session_id="deterministic-session",
            account=paper_account(),
            scenario=FakeBrokerScenario(
                partial_fill_quantities=(Decimal("4"), Decimal("6")),
                commission=Decimal("0.75"),
            ),
        )
        fake.connect()
        boundary = PaperExecutionBoundary(
            fake,
            safety_snapshot=lambda: replace(
                safety(), session_id="deterministic-session"
            ),
            account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
            authorization=TestAuthorization(),
        )
        boundary.submit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )
        return stable_hash(
            {
                "orders": fake.completed_orders(),
                "executions": fake.executions(),
                "commissions": fake.commission_reports,
                "events": fake.broker_events,
                "account": fake.account_snapshot(),
            }
        )

    assert run() == run()


def test_disconnect_during_submit_never_blindly_retries_same_order() -> None:
    fake = FakeBroker(
        session_id="paper-session-1",
        account=paper_account(),
        scenario=FakeBrokerScenario(disconnect_during_submit=True),
    )
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    context = TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED)
    with pytest.raises(ReconciliationRequiredError):
        boundary.submit_approved(approved_order(), context)
    fake.reconnect()
    with pytest.raises(ReconciliationRequiredError, match="reconcile"):
        boundary.submit_approved(approved_order(), context)
    assert fake.transmission_count == 1
    assert fake.open_orders()[0].state is BrokerOrderState.RECONCILIATION_REQUIRED


def test_fake_reject_creates_no_execution_or_economic_fill() -> None:
    fake = FakeBroker(
        session_id="paper-session-1",
        account=paper_account(),
        scenario=FakeBrokerScenario(reject=True),
    )
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    boundary.submit_approved(
        approved_order(),
        TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
    )
    assert fake.executions() == ()
    assert fake.commission_reports == ()
    assert fake.completed_orders()[0].state is BrokerOrderState.REJECTED


def test_disconnect_before_submit_is_deferred_without_transmission() -> None:
    fake = FakeBroker(
        session_id="paper-session-1",
        account=paper_account(),
        scenario=FakeBrokerScenario(disconnect_before_submit=True),
    )
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    with pytest.raises(BrokerUnavailableError, match="before submit"):
        boundary.submit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )
    assert fake.transmission_count == 0
    assert fake.open_orders() == ()


def test_disconnect_after_submit_preserves_broker_order_for_reconciliation() -> None:
    fake = FakeBroker(
        session_id="paper-session-1",
        account=paper_account(),
        scenario=FakeBrokerScenario(
            partial_fill_quantities=(Decimal("4"),),
            disconnect_after_submit=True,
        ),
    )
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    receipt = boundary.submit_approved(
        approved_order(),
        TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
    )
    assert receipt.broker_order_id == "fake-1"
    assert fake.connection_state is BrokerConnectionState.DISCONNECTED
    assert fake.open_orders()[0].state is BrokerOrderState.PARTIALLY_FILLED
    assert fake.transmission_count == 1


def test_system_owned_partial_order_can_cancel_but_external_order_cannot() -> None:
    fake = FakeBroker(
        session_id="paper-session-1",
        account=paper_account(),
        scenario=FakeBrokerScenario(partial_fill_quantities=(Decimal("4"),)),
    )
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    boundary.submit_approved(
        approved_order(),
        TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
    )
    fake.cancel_order("order-1")
    assert fake.completed_orders()[0].state is BrokerOrderState.CANCELLED

    external = replace(
        fake.completed_orders()[0],
        internal_order_id="external-order",
        client_order_key="external-key",
        broker_order_id="manual-1",
        state=BrokerOrderState.ACKNOWLEDGED,
        filled_quantity=Decimal("0"),
        risk_decision_id="EXTERNAL_BROKER_ACTIVITY",
        external=True,
    )
    fake.inject_external_order(external)
    with pytest.raises(ReconciliationRequiredError, match="external"):
        fake.cancel_order("external-order")


def test_correction_is_explicit_and_missing_commission_remains_unavailable() -> None:
    fake = FakeBroker(
        session_id="paper-session-1",
        account=paper_account(),
        scenario=FakeBrokerScenario(
            execution_correction_price=Decimal("101"), commission=None
        ),
    )
    fake.connect()
    boundary = PaperExecutionBoundary(
        fake,
        safety_snapshot=safety,
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    boundary.submit_approved(
        approved_order(),
        TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
    )
    executions = fake.executions()
    assert len(executions) == 2
    assert executions[1].correction_of == executions[0].exec_id
    assert all(
        report.status is CommissionKnowledge.UNAVAILABLE
        and report.amount is None
        for report in fake.commission_reports
    )


def test_read_only_session_reconciles_and_never_arms_execution(tmp_path) -> None:
    fake = FakeBroker(session_id="session-read-only", account=paper_account())
    local = ReconciliationState(Decimal("100000"), (), (), ())
    session = PaperTradingSession(
        fake,
        session_id="session-read-only",
        mode=PaperMode.PAPER_READ_ONLY,
        allowed_account_hashes=(ACCOUNT_HASH,),
        config_hashes=(("risk", "d" * 64),),
        code_sha="2b192ad38acf1a0c3a08f0173417597724c16ebd",
        store=LocalPaperStore(tmp_path / "paper"),
        data_freshness=DataFreshnessStatus.FRESH,
        risk_health_ok=True,
        market_session_open=True,
    )
    assert session.start(local_state=local) is PaperSessionState.READY
    session.set_data_freshness(DataFreshnessStatus.FRESH)
    assert session.safety_snapshot().mode is PaperMode.PAPER_READ_ONLY
    with pytest.raises(PaperExecutionLockedError, match="not armed"):
        session.boundary.submit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )
    inspected = session.store.inspect("session-read-only")
    assert inspected["session"]["paper_execution_armed"] is False
    assert inspected["integrity"] == "VERIFIED"


def test_decision_and_outcome_envelopes_persist_complete_read_only_lineage(
    tmp_path,
) -> None:
    from trading_ai.brokers.models import PaperDecisionEnvelope, PaperOutcomeEnvelope

    store = LocalPaperStore(tmp_path / "paper")
    fake = FakeBroker(session_id="session-envelope", account=paper_account())
    session = PaperTradingSession(
        fake,
        session_id="session-envelope",
        mode=PaperMode.PAPER_READ_ONLY,
        allowed_account_hashes=(ACCOUNT_HASH,),
        config_hashes=(("risk", "d" * 64),),
        code_sha="test-sha",
        store=store,
        data_freshness=DataFreshnessStatus.FRESH,
        risk_health_ok=True,
        market_session_open=True,
    )
    session.start(local_state=ReconciliationState(Decimal("100000"), (), (), ()))
    decision = PaperDecisionEnvelope(
        envelope_id="decision-envelope-1",
        session_id="session-envelope",
        timestamp=NOW,
        symbol="AAPL",
        order_id="order-1",
        risk_decision_id="risk-1",
        data_snapshot_id="data-1",
        feature_snapshot_id="feature-1",
        regime_snapshot_id="regime-1",
        signal_id="signal-1",
        ml_decision_id="ml-1",
        activation_decision_id="activation-1",
        portfolio_plan_id="portfolio-1",
        cost_estimate_id="cost-1",
        economic_decision_id="economic-1",
        account_snapshot_hash="a" * 64,
        broker_health_hash="b" * 64,
        reconciliation_hash="c" * 64,
    )
    outcome = PaperOutcomeEnvelope(
        envelope_id="outcome-envelope-1",
        decision_envelope_id=decision.envelope_id,
        session_id="session-envelope",
        completed_at=NOW,
        order_id="order-1",
        broker_order_id="fake-1",
        final_state=BrokerOrderState.FILLED,
        execution_ids=("exec-1",),
        event_ids=("event-1",),
        estimated_cost=Decimal("1.25"),
        broker_reported_cost=Decimal("1.30"),
        resulting_cash=Decimal("98998.70"),
        resulting_positions=(
            BrokerPosition("AAPL", Decimal("10"), Decimal("100"), "USD"),
        ),
        resulting_equity=Decimal("99998.70"),
        reconciliation_hash="c" * 64,
    )
    session.record_decision_envelope(decision)
    session.record_outcome_envelope(outcome)
    payload = store.inspect("session-envelope")
    assert payload["decisions"][0]["risk_decision_id"] == "risk-1"
    assert payload["outcomes"][0]["decision_envelope_id"] == decision.envelope_id
    audit = PaperShadowAudit(store).audit("session-envelope")
    assert audit.status == "UNAVAILABLE"
    assert audit.decision_envelopes == 1
    assert audit.outcome_envelopes == 1
    observed_hash = stable_hash(payload["decisions"][0])
    compared = PaperShadowAudit(store).audit(
        "session-envelope",
        recalculated_decision_hashes={decision.envelope_id: observed_hash},
    )
    assert compared.status == "IN_SYNC"
    assert compared.audit_hash != audit.audit_hash
    mismatched = PaperShadowAudit(store).audit(
        "session-envelope",
        recalculated_decision_hashes={decision.envelope_id: "0" * 64},
    )
    assert mismatched.status == "DRIFT"
    assert mismatched.divergences == (f"HASH_MISMATCH:{decision.envelope_id}",)


def test_session_config_change_halts_and_requires_restart(tmp_path) -> None:
    fake = FakeBroker(session_id="session-config", account=paper_account())
    session = PaperTradingSession(
        fake,
        session_id="session-config",
        mode=PaperMode.PAPER_READ_ONLY,
        allowed_account_hashes=(ACCOUNT_HASH,),
        config_hashes=(("risk", "d" * 64),),
        code_sha="test-sha",
        store=LocalPaperStore(tmp_path / "paper"),
        data_freshness=DataFreshnessStatus.FRESH,
        risk_health_ok=True,
        market_session_open=True,
    )
    session.start(local_state=ReconciliationState(Decimal("100000"), (), (), ()))
    with pytest.raises(BrokerConfigurationError, match="restart"):
        session.assert_frozen_configs((("risk", "e" * 64),))
    assert session.state is PaperSessionState.HALTED
    assert session.safety_snapshot().emergency_halt.active is True


def test_stale_broker_session_never_reaches_ready(tmp_path) -> None:
    fake = FakeBroker(
        session_id="session-stale",
        account=paper_account(),
        scenario=FakeBrokerScenario(stale=True),
    )
    session = PaperTradingSession(
        fake,
        session_id="session-stale",
        mode=PaperMode.PAPER_READ_ONLY,
        allowed_account_hashes=(ACCOUNT_HASH,),
        config_hashes=(("risk", "d" * 64),),
        code_sha="test-sha",
        store=LocalPaperStore(tmp_path / "paper"),
        data_freshness=DataFreshnessStatus.FRESH,
        risk_health_ok=True,
        market_session_open=True,
    )
    assert session.start(
        local_state=ReconciliationState(Decimal("100000"), (), (), ())
    ) is PaperSessionState.DEGRADED
    assert session.safety_snapshot().connection_state is BrokerConnectionState.STALE


def test_connectivity_check_reads_identity_then_disconnects_without_orders(tmp_path) -> None:
    fake = FakeBroker(session_id="session-connect", account=paper_account())
    session = PaperTradingSession(
        fake,
        session_id="session-connect",
        mode=PaperMode.CONNECTIVITY_CHECK,
        allowed_account_hashes=(ACCOUNT_HASH,),
        config_hashes=(("broker", "f" * 64),),
        code_sha="test-sha",
        store=LocalPaperStore(tmp_path / "paper"),
    )
    assert session.start() is PaperSessionState.DISCONNECTED
    assert fake.connection_state is BrokerConnectionState.DISCONNECTED
    assert fake.transmission_count == 0


def test_reconciliation_detects_external_activity_as_critical_drift() -> None:
    external = BrokerOrderRecord(
        internal_order_id="external-1",
        client_order_key="external-key",
        session_id="session-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=approved_order().order.order_type,
        quantity=Decimal("1"),
        filled_quantity=Decimal("0"),
        state=BrokerOrderState.ACKNOWLEDGED,
        risk_decision_id="EXTERNAL_BROKER_ACTIVITY",
        created_at=NOW,
        updated_at=NOW,
        broker_order_id="manual-1",
        external=True,
    )
    empty = ReconciliationState(Decimal("1000"), (), (), ())
    broker = ReconciliationState(Decimal("1000"), (), (external,), ())
    result = BrokerReconciler().reconcile(
        session_id="session-1", observed_at=NOW, local=empty, broker=broker
    )
    assert result.status is ReconciliationStatus.CRITICAL_DRIFT
    assert result.external_activity == ("ORDER:manual-1",)


def test_reconciliation_treats_cash_and_missing_fill_drift_as_critical() -> None:
    execution = BrokerExecution(
        exec_id="exec-missing",
        internal_order_id="order-1",
        client_order_key="key-1",
        broker_order_id="broker-1",
        perm_id="perm-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        price=Decimal("100"),
        broker_timestamp=NOW,
        received_at=NOW,
    )
    local = ReconciliationState(Decimal("900"), (), (), (execution,))
    broker = ReconciliationState(Decimal("899"), (), (), ())
    result = BrokerReconciler().reconcile(
        session_id="session-1", observed_at=NOW, local=local, broker=broker
    )
    assert result.status is ReconciliationStatus.CRITICAL_DRIFT
    assert result.differences == (
        "BROKER_EXECUTION_MISSING",
        "CASH_MISMATCH",
    )


def test_invalid_order_lifecycle_transition_is_refused() -> None:
    order = BrokerOrderRecord(
        internal_order_id="order",
        client_order_key="key",
        session_id="session",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=approved_order().order.order_type,
        quantity=Decimal("1"),
        filled_quantity=Decimal("1"),
        state=BrokerOrderState.FILLED,
        risk_decision_id="risk",
        created_at=NOW,
        updated_at=NOW,
    )
    from trading_ai.brokers.exceptions import BrokerStateTransitionError

    with pytest.raises(BrokerStateTransitionError):
        BrokerOrderStateMachine.transition(
            order, BrokerOrderState.SUBMITTED, updated_at=NOW
        )


def test_restart_restores_partial_fill_mapping_without_duplicate_submission(tmp_path) -> None:
    store = LocalPaperStore(tmp_path / "paper")
    from trading_ai.brokers.models import BrokerExecution, PaperSessionManifest

    store.create_session(
        PaperSessionManifest(
            session_id="restart-session",
            created_at=NOW,
            code_sha="sha",
            mode=PaperMode.PAPER_READ_ONLY,
            broker_adapter_name="fake-paper-broker",
            broker_adapter_version="1.0",
            official_sdk_version=None,
            server_version=None,
            account_hash=ACCOUNT_HASH,
            account_masked="IBKR-****0001",
            config_hashes=(("risk", "d" * 64),),
            ml_model_ids=(),
        )
    )
    approved = approved_order()
    from trading_ai.brokers.idempotency import client_order_key

    key = client_order_key(
        "restart-session", approved.order.order_id, approved.risk_decision.decision_id
    )
    partial = BrokerOrderRecord(
        internal_order_id=approved.order.order_id,
        client_order_key=key,
        session_id="restart-session",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=approved.order.order_type,
        quantity=Decimal("10"),
        filled_quantity=Decimal("4"),
        state=BrokerOrderState.PARTIALLY_FILLED,
        risk_decision_id=approved.risk_decision.decision_id,
        created_at=NOW,
        updated_at=NOW,
        broker_order_id="fake-1",
    )
    execution = BrokerExecution(
        exec_id="exec-fake-1-1",
        internal_order_id=approved.order.order_id,
        client_order_key=key,
        broker_order_id="fake-1",
        perm_id="perm-fake-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("4"),
        price=Decimal("100"),
        broker_timestamp=NOW,
        received_at=NOW,
    )
    store.append("restart-session", "orders", partial, record_id="partial-order")
    store.append("restart-session", "executions", execution, record_id="partial-execution")
    store.append(
        "restart-session",
        "snapshots",
        {
            "observed_at": NOW,
            "cash": "100000",
            "net_liquidation": "100000",
            "positions": [],
        },
        record_id="account-snapshot",
    )
    restored = FakeBroker(session_id="restart-session", account=paper_account())
    local = PaperRecoveryService(store).restore("restart-session", restored)
    restored.connect()
    assert local.orders[0].state is BrokerOrderState.PARTIALLY_FILLED
    boundary = PaperExecutionBoundary(
        restored,
        safety_snapshot=lambda: replace(safety(), session_id="restart-session"),
        account_guard=PaperAccountGuard((ACCOUNT_HASH,)),
        authorization=TestAuthorization(),
    )
    receipt = boundary.submit_approved(
        approved,
        TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
    )
    assert receipt.broker_order_id == "fake-1"
    assert restored.transmission_count == 0


def test_paper_ledger_applies_partial_fills_once_and_keeps_corrections_explicit() -> None:
    from trading_ai.brokers.models import (
        BrokerCommissionReport,
        BrokerExecution,
        CommissionKnowledge,
    )

    ledger = PaperLedger(cash=Decimal("1000"), base_currency="USD")
    first = BrokerExecution(
        exec_id="execution.01",
        internal_order_id="order-1",
        client_order_key="key-1",
        broker_order_id="broker-1",
        perm_id="perm-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("4"),
        price=Decimal("100"),
        broker_timestamp=NOW,
        received_at=NOW,
    )
    assert ledger.apply_execution(first) is True
    assert ledger.apply_execution(first) is False
    assert ledger.snapshot.cash == Decimal("600")
    corrected = replace(
        first,
        exec_id="execution.02",
        price=Decimal("101"),
        correction_of=first.exec_id,
    )
    assert ledger.apply_execution(corrected) is True
    assert ledger.snapshot.cash == Decimal("596")
    commission = BrokerCommissionReport(
        exec_id=corrected.exec_id,
        status=CommissionKnowledge.KNOWN,
        received_at=NOW,
        amount=Decimal("1"),
        currency="USD",
    )
    assert ledger.apply_commission(commission) is True
    assert ledger.apply_commission(commission) is False
    assert ledger.snapshot.cash == Decimal("595")
    unavailable = BrokerCommissionReport(
        exec_id=corrected.exec_id,
        status=CommissionKnowledge.UNAVAILABLE,
        received_at=NOW,
    )
    # Already applied is idempotent; unavailable was never interpreted as zero.
    assert ledger.apply_commission(unavailable) is False
