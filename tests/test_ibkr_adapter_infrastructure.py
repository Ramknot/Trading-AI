from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from trading_ai.brokers.config import IBKRPaperConfig, load_ibkr_paper_config
from trading_ai.brokers.exceptions import (
    BrokerConfigurationError,
    BrokerIntegrityError,
    ContractResolutionError,
    PaperExecutionLockedError,
)
from trading_ai.brokers.ibkr.adapter import IBKRPaperAdapter
from trading_ai.brokers.ibkr.client import IBKRClientPort, OfficialIBAPIClient
from trading_ai.brokers.ibkr.contracts import (
    IBKRContractCandidate,
    IBKRContractResolver,
    IBKRContractSpec,
    load_contract_specs,
)
from trading_ai.brokers.ibkr.errors import normalize_ibkr_error
from trading_ai.brokers.ibkr.orders import IBKROrderSpec
from trading_ai.brokers.ibkr.versioning import validate_sdk_version
from trading_ai.brokers.models import (
    BrokerEnvironment,
    BrokerErrorSeverity,
    BrokerEventType,
    BrokerOrderState,
    PaperMode,
    PaperSessionManifest,
)
from trading_ai.brokers.replay import PaperEventReplay, PaperShadowAudit
from trading_ai.brokers.storage import LocalPaperStore
from trading_ai.core.hashing import stable_hash


class StubIBKRClient(IBKRClientPort):
    def __init__(self, account_id="DU123456", sdk="10.50.0") -> None:
        self._account_ids = (account_id,)
        self._sdk = sdk
        self._connected = False
        self.placed = []

    def connect(self, host, port, client_id, timeout):
        del host, port, client_id, timeout
        self._connected = True

    def disconnect(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    @property
    def account_ids(self):
        return self._account_ids

    @property
    def next_order_id(self):
        return 1

    @property
    def sdk_version(self):
        return self._sdk

    @property
    def server_version(self):
        return "190"

    def request_state(self):
        return None

    def request_contract_details(self, request_id, contract):
        del request_id, contract

    def place_order(self, order_id, contract, order, *, account_id, client_order_key):
        self.placed.append((order_id, contract, order, account_id, client_order_key))

    def cancel_order(self, order_id):
        del order_id

    def request_current_time(self):
        return None


def test_example_config_is_non_connectable_and_real_loader_fails_closed() -> None:
    config = load_ibkr_paper_config(
        Path("config/brokers/ibkr_paper.example.toml"), allow_example=True
    )
    assert config.host == "127.0.0.1"
    assert config.mode is PaperMode.PAPER_READ_ONLY
    assert config.paper_execution_armed is False
    assert config.connectable is False
    assert len(config.config_hash) == 64
    with pytest.raises(BrokerConfigurationError, match="example"):
        load_ibkr_paper_config(Path("config/brokers/ibkr_paper.example.toml"))


@pytest.mark.parametrize("host", ("0.0.0.0", "192.168.1.4", "broker.example"))
def test_ibkr_config_rejects_non_loopback_hosts(host) -> None:
    with pytest.raises(BrokerConfigurationError, match="loopback"):
        IBKRPaperConfig(
            host=host,
            port=7497,
            client_id=1,
            mode=PaperMode.PAPER_READ_ONLY,
            expected_environment=BrokerEnvironment.PAPER,
            allowed_account_hashes=("a" * 64,),
            account_hash_salt_env="SALT",
            request_timeout_seconds=1,
            heartbeat_timeout_seconds=1,
            max_clock_drift_seconds=5,
            max_messages_per_second=20,
            tif="DAY",
            official_sdk_version="10.50",
            contract_config="contracts.toml",
        )


def test_lot9_config_rejects_execution_armed_and_non_day_tif() -> None:
    base = dict(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        expected_environment=BrokerEnvironment.PAPER,
        allowed_account_hashes=("a" * 64,),
        account_hash_salt_env="SALT",
        request_timeout_seconds=1,
        heartbeat_timeout_seconds=1,
        max_clock_drift_seconds=5,
        max_messages_per_second=20,
        official_sdk_version="10.50",
        contract_config="contracts.toml",
    )
    with pytest.raises(BrokerConfigurationError, match="does not permit"):
        IBKRPaperConfig(mode=PaperMode.PAPER_EXECUTION_ARMED, tif="DAY", **base)
    with pytest.raises(BrokerConfigurationError, match="DAY"):
        IBKRPaperConfig(mode=PaperMode.PAPER_READ_ONLY, tif="GTC", **base)


def test_balanced_contracts_cover_profile_and_require_exact_resolution() -> None:
    specs = load_contract_specs("config/brokers/ibkr_contracts_balanced.toml")
    assert len(specs) == 13
    assert {item.symbol for item in specs} >= {"AAPL", "MC.PA", "AIR.PA"}
    assert all(item.con_id is None for item in specs)
    assert all(item.currency in {"USD", "EUR"} for item in specs)


def test_contract_resolver_refuses_zero_or_ambiguous_candidates(tmp_path) -> None:
    spec = IBKRContractSpec("AAPL", "AAPL", "STK", "SMART", "NASDAQ", "USD")
    resolver = IBKRContractResolver((spec,), cache_path=tmp_path / "contracts.json")
    candidate = IBKRContractCandidate(265598, "AAPL", "STK", "SMART", "NASDAQ", "USD", "AAPL")
    with pytest.raises(ContractResolutionError, match="found 0"):
        resolver.resolve("AAPL", ())
    with pytest.raises(ContractResolutionError, match="found 2"):
        resolver.resolve("AAPL", (candidate, candidate))
    resolved = resolver.resolve("AAPL", (candidate,))
    assert resolved.con_id == 265598
    reloaded = IBKRContractResolver((spec,), cache_path=tmp_path / "contracts.json")
    assert reloaded.resolve("AAPL").con_id == 265598


def test_contract_cache_tampering_is_detected(tmp_path) -> None:
    spec = IBKRContractSpec("AAPL", "AAPL", "STK", "SMART", "NASDAQ", "USD")
    path = tmp_path / "contracts.json"
    resolver = IBKRContractResolver((spec,), cache_path=path)
    resolver.resolve(
        "AAPL",
        (IBKRContractCandidate(1, "AAPL", "STK", "SMART", "NASDAQ", "USD", "AAPL"),),
    )
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    # Whitespace does not change JSON facts, so alter a value as a real tamper.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contracts"][0]["currency"] = "EUR"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BrokerIntegrityError):
        IBKRContractResolver((spec,), cache_path=path)


def test_ibkr_adapter_masks_and_hashes_account_and_blocks_direct_submit(monkeypatch) -> None:
    salt = "local-test-salt"
    account_id = "DU123456"
    digest = hashlib.sha256(f"{salt}:{account_id}".encode()).hexdigest()
    monkeypatch.setenv("TEST_IBKR_SALT", salt)
    config = IBKRPaperConfig(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        mode=PaperMode.PAPER_READ_ONLY,
        expected_environment=BrokerEnvironment.PAPER,
        allowed_account_hashes=(digest,),
        account_hash_salt_env="TEST_IBKR_SALT",
        request_timeout_seconds=1,
        heartbeat_timeout_seconds=30,
        max_clock_drift_seconds=5,
        max_messages_per_second=20,
        tif="DAY",
        official_sdk_version="10.50",
        contract_config="config/brokers/ibkr_contracts_balanced.toml",
    )
    resolver = IBKRContractResolver(load_contract_specs(config.contract_config))
    adapter = IBKRPaperAdapter(
        config,
        resolver,
        session_id="ibkr-test",
        client=StubIBKRClient(account_id),
    )
    adapter.connect()
    assert adapter.account_identity is not None
    assert adapter.account_identity.account_hash == digest
    assert account_id not in adapter.account_identity.account_masked
    from test_broker_paper_guards import approved_order
    from trading_ai.core.models import ExecutionEnvironment, TradingContext, TradingProfileName

    with pytest.raises(PaperExecutionLockedError, match="PaperExecutionBoundary"):
        adapter.submit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )
    with pytest.raises(PaperExecutionLockedError, match="not armed"):
        adapter.transmit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.PAPER, TradingProfileName.BALANCED),
        )
    with pytest.raises(PaperExecutionLockedError, match="PAPER context"):
        adapter.transmit_approved(
            approved_order(),
            TradingContext(ExecutionEnvironment.LIVE, TradingProfileName.BALANCED),
        )
    with pytest.raises(PaperExecutionLockedError, match="not armed"):
        adapter.cancel_order("order-1")
    assert adapter._client.placed == []

    adapter._on_callback(
        "ACCOUNT_SUMMARY",
        {"tag": "NetLiquidation", "value": "1000", "currency": "USD"},
    )
    adapter._on_callback(
        "ACCOUNT_SUMMARY",
        {"tag": "TotalCashValue", "value": "750", "currency": "USD"},
    )
    assert adapter.account_snapshot().account.base_currency == "USD"


@pytest.mark.parametrize(
    "account_id, expected",
    (("U123456", BrokerEnvironment.LIVE), ("OTHER123", BrokerEnvironment.UNKNOWN)),
)
def test_adapter_classifies_non_paper_accounts_fail_closed(
    monkeypatch, account_id, expected
) -> None:
    salt = "local-test-salt"
    digest = hashlib.sha256(f"{salt}:{account_id}".encode()).hexdigest()
    monkeypatch.setenv("TEST_IBKR_SALT", salt)
    config = IBKRPaperConfig(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        mode=PaperMode.PAPER_READ_ONLY,
        expected_environment=BrokerEnvironment.PAPER,
        allowed_account_hashes=(digest,),
        account_hash_salt_env="TEST_IBKR_SALT",
        request_timeout_seconds=1,
        heartbeat_timeout_seconds=30,
        max_clock_drift_seconds=5,
        max_messages_per_second=20,
        tif="DAY",
        official_sdk_version="10.50",
        contract_config="config/brokers/ibkr_contracts_balanced.toml",
    )
    adapter = IBKRPaperAdapter(
        config,
        IBKRContractResolver(load_contract_specs(config.contract_config)),
        session_id="ibkr-non-paper",
        client=StubIBKRClient(account_id),
    )
    adapter.connect()
    assert adapter.account_identity is not None
    assert adapter.account_identity.environment is expected
    assert adapter.account_identity.environment_verified is False


def test_adapter_preserves_external_limit_order_and_broker_timestamp(monkeypatch) -> None:
    salt = "local-test-salt"
    account_id = "DU123456"
    digest = hashlib.sha256(f"{salt}:{account_id}".encode()).hexdigest()
    monkeypatch.setenv("TEST_IBKR_SALT", salt)
    config = IBKRPaperConfig(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        mode=PaperMode.PAPER_READ_ONLY,
        expected_environment=BrokerEnvironment.PAPER,
        allowed_account_hashes=(digest,),
        account_hash_salt_env="TEST_IBKR_SALT",
        request_timeout_seconds=1,
        heartbeat_timeout_seconds=30,
        max_clock_drift_seconds=5,
        max_messages_per_second=20,
        tif="DAY",
        official_sdk_version="10.50",
        contract_config="config/brokers/ibkr_contracts_balanced.toml",
    )
    adapter = IBKRPaperAdapter(
        config,
        IBKRContractResolver(load_contract_specs(config.contract_config)),
        session_id="ibkr-callback-test",
        client=StubIBKRClient(account_id),
    )
    adapter.connect()
    adapter._on_callback(
        "OPEN_ORDER",
        {
            "broker_order_id": "77",
            "perm_id": "7007",
            "client_order_key": "",
            "symbol": "AAPL",
            "currency": "USD",
            "side": "BUY",
            "order_type": "LMT",
            "quantity": "3",
            "limit_price": "189.50",
            "tif": "DAY",
        },
    )
    order = adapter.open_orders()[0]
    assert order.external is True
    assert order.state is BrokerOrderState.ACKNOWLEDGED
    assert order.limit_price == Decimal("189.50")

    adapter._on_callback(
        "EXECUTION",
        {
            "exec_id": "external-exec-1",
            "broker_order_id": "77",
            "perm_id": "7007",
            "client_order_key": order.client_order_key,
            "symbol": "AAPL",
            "currency": "USD",
            "side": "BUY",
            "quantity": "1",
            "price": "189.50",
            "time": "20260831-16:30:00 Europe/Paris",
        },
    )
    execution = adapter.executions()[0]
    assert execution.broker_timestamp == datetime(
        2026, 8, 31, 14, 30, tzinfo=timezone.utc
    )
    assert execution.received_at.tzinfo is timezone.utc
    fill_event = next(
        event
        for event in adapter.broker_events
        if event.event_type is BrokerEventType.PARTIAL_FILL
    )
    assert fill_event.broker_timestamp == execution.broker_timestamp
    adapter._on_callback(
        "EXECUTION",
        {
            "exec_id": "external-exec-1.1",
            "broker_order_id": "77",
            "perm_id": "7007",
            "client_order_key": order.client_order_key,
            "symbol": "AAPL",
            "currency": "USD",
            "side": "BUY",
            "quantity": "1",
            "price": "189.25",
            "time": "20260831-16:31:00 Europe/Paris",
        },
    )
    correction = next(
        item for item in adapter.executions() if item.exec_id == "external-exec-1.1"
    )
    assert correction.correction_of == "external-exec-1"
    assert (
        "EXECUTION_CORRECTION_REQUIRES_RECONCILIATION"
        in adapter.health().critical_errors
    )
    adapter._on_callback(
        "COMMISSION_REPORT",
        {"exec_id": "external-exec-1", "commission": "0.75", "currency": "USD"},
    )
    adapter._on_callback(
        "COMMISSION_REPORT",
        {"exec_id": "external-exec-1", "commission": "0.75", "currency": "USD"},
    )
    assert adapter.commission_reports[0].amount == Decimal("0.75")
    assert "COMMISSION_CORRECTION_REQUIRES_REVIEW" not in adapter.health().critical_errors
    adapter._on_callback(
        "COMMISSION_REPORT",
        {"exec_id": "external-exec-1", "commission": "0.80", "currency": "USD"},
    )
    assert "COMMISSION_CORRECTION_REQUIRES_REVIEW" in adapter.health().critical_errors
    assert adapter.commission_reports[0].amount == Decimal("0.75")

    adapter._on_callback(
        "POSITION",
        {
            "symbol": "AAPL",
            "currency": "USD",
            "quantity": "2",
            "average_cost": "180",
        },
    )
    assert adapter.positions()[0].quantity == Decimal("2")
    adapter._on_callback(
        "POSITION",
        {
            "symbol": "AAPL",
            "currency": "USD",
            "quantity": "0",
            "average_cost": "0",
        },
    )
    assert adapter.positions() == ()

    adapter._on_callback(
        "OPEN_ORDER",
        {
            "broker_order_id": "78",
            "perm_id": "7008",
            "client_order_key": "",
            "symbol": "AAPL",
            "currency": "USD",
            "side": "SELL",
            "order_type": "STP",
            "quantity": "1",
            "limit_price": "0",
            "tif": "DAY",
        },
    )
    assert "UNSUPPORTED_BROKER_ORDER_TYPE" in adapter.health().critical_errors

    adapter._on_callback(
        "EXECUTION",
        {
            "exec_id": "external-exec-without-order",
            "broker_order_id": "99",
            "perm_id": "9009",
            "client_order_key": "",
            "symbol": "AAPL",
            "currency": "USD",
            "side": "SLD",
            "quantity": "1",
            "price": "190",
            "time": "20260831-17:00:00 Europe/Paris",
        },
    )
    assert {item.exec_id for item in adapter.executions()} == {
        "external-exec-1",
        "external-exec-1.1",
        "external-exec-without-order",
    }
    assert "EXTERNAL_BROKER_ACTIVITY" in adapter.health().critical_errors


def test_adapter_sync_uses_only_fresh_broker_snapshot(monkeypatch) -> None:
    salt = "local-test-salt"
    account_id = "DU123456"
    digest = hashlib.sha256(f"{salt}:{account_id}".encode()).hexdigest()
    monkeypatch.setenv("TEST_IBKR_SALT", salt)

    class SyncingClient(StubIBKRClient):
        sink = None

        def request_state(self):
            assert self.sink is not None
            self.sink(
                "ACCOUNT_SUMMARY",
                {"tag": "NetLiquidation", "value": "1000", "currency": "USD"},
            )
            self.sink(
                "ACCOUNT_SUMMARY",
                {"tag": "TotalCashValue", "value": "1000", "currency": "USD"},
            )
            for kind in (
                "ACCOUNT_SUMMARY_END",
                "POSITIONS_END",
                "OPEN_ORDERS_END",
                "COMPLETED_ORDERS_END",
                "EXECUTIONS_END",
            ):
                self.sink(kind, {})

    config = IBKRPaperConfig(
        host="127.0.0.1",
        port=7497,
        client_id=1,
        mode=PaperMode.PAPER_READ_ONLY,
        expected_environment=BrokerEnvironment.PAPER,
        allowed_account_hashes=(digest,),
        account_hash_salt_env="TEST_IBKR_SALT",
        request_timeout_seconds=1,
        heartbeat_timeout_seconds=30,
        max_clock_drift_seconds=5,
        max_messages_per_second=20,
        tif="DAY",
        official_sdk_version="10.50",
        contract_config="config/brokers/ibkr_contracts_balanced.toml",
    )
    client = SyncingClient(account_id)
    adapter = IBKRPaperAdapter(
        config,
        IBKRContractResolver(load_contract_specs(config.contract_config)),
        session_id="ibkr-sync-test",
        client=client,
    )
    client.sink = adapter._on_callback
    adapter.connect()
    adapter._on_callback(
        "POSITION",
        {
            "symbol": "AAPL",
            "currency": "USD",
            "quantity": "4",
            "average_cost": "170",
        },
    )
    adapter._on_callback("ERROR", {"code": 1101})
    assert adapter.positions()
    assert "IBKR_CONNECTIVITY_RESTORED_DATA_LOST" in adapter.health().critical_errors

    snapshot = adapter.sync_state()
    assert snapshot.cash == Decimal("1000")
    assert snapshot.positions == ()
    assert snapshot.orders == ()
    assert snapshot.executions == ()
    assert "IBKR_CONNECTIVITY_RESTORED_DATA_LOST" not in adapter.health().critical_errors


@pytest.mark.parametrize(
    "code, stable, severity",
    [
        (1100, "IBKR_CONNECTIVITY_LOST", BrokerErrorSeverity.CONNECTIVITY),
        (1101, "IBKR_CONNECTIVITY_RESTORED_DATA_LOST", BrokerErrorSeverity.WARNING),
        (1102, "IBKR_CONNECTIVITY_RESTORED_DATA_MAINTAINED", BrokerErrorSeverity.INFORMATIONAL),
        (100, "IBKR_PACING_LIMIT", BrokerErrorSeverity.CRITICAL),
        (103, "IBKR_DUPLICATE_ORDER_ID", BrokerErrorSeverity.REJECT),
    ],
)
def test_documented_ibkr_error_codes_are_normalized(code, stable, severity) -> None:
    result = normalize_ibkr_error(code)
    assert result.stable_code == stable
    assert result.severity is severity


def test_sdk_version_is_explicit_and_mismatch_fails_closed() -> None:
    validate_sdk_version("10.50", "10.50.0")
    with pytest.raises(BrokerConfigurationError, match="mismatch"):
        validate_sdk_version("10.50", "10.45")


def test_official_client_dispatches_callbacks_through_controlled_queue() -> None:
    observed = []
    client = OfficialIBAPIClient(
        lambda kind, payload: observed.append((kind, payload)),
        expected_sdk_version="10.50",
    )
    client._start_dispatcher()
    client._enqueue_event("TEST_EVENT", {"sequence": 1})
    client._event_queue.join()
    client.disconnect()
    assert observed == [("TEST_EVENT", {"sequence": 1})]
    with pytest.raises(BrokerConfigurationError, match="unsupported"):
        validate_sdk_version("9.76", "9.76")


def test_official_client_uses_socket_state_not_connect_return_value(monkeypatch) -> None:
    observed = []
    client = OfficialIBAPIClient(
        lambda kind, payload: observed.append((kind, payload)),
        expected_sdk_version="10.50",
    )

    class FakeOfficialApp:
        def __init__(self) -> None:
            self.connected = False

        def connect(self, host, port, clientId):
            del host, port, clientId
            self.connected = True
            client._account_ids = ("DU654321",)
            client._next_order_id = 42
            client._ready.set()
            return None

        def isConnected(self):
            return self.connected

        def run(self):
            return None

        def disconnect(self):
            self.connected = False

    monkeypatch.setattr(client, "_build_app", lambda: FakeOfficialApp())
    client._account_ids = ("STALE_ACCOUNT",)
    client.connect("127.0.0.1", 7497, 17, 1.0)
    assert client.connected is True
    assert client.account_ids == ("DU654321",)
    assert client.next_order_id == 42
    client.disconnect()


def test_paper_store_replay_shadow_and_tamper_evidence(tmp_path) -> None:
    store = LocalPaperStore(tmp_path / "paper")
    manifest = PaperSessionManifest(
        session_id="paper-1",
        created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        code_sha="2b192ad38acf1a0c3a08f0173417597724c16ebd",
        mode=PaperMode.PAPER_READ_ONLY,
        broker_adapter_name="fake-paper-broker",
        broker_adapter_version="1.0",
        official_sdk_version=None,
        server_version=None,
        account_hash="a" * 64,
        account_masked="IBKR-****0001",
        config_hashes=(("risk", "b" * 64),),
        ml_model_ids=(),
    )
    store.create_session(manifest)
    store.append("paper-1", "events", {"event_id": "event-1"}, record_id="event-1")
    store.append(
        "paper-1", "decisions", {"envelope_id": "decision-1"}, record_id="decision-1"
    )
    assert store.inspect("paper-1")["integrity"] == "VERIFIED"
    replay = PaperEventReplay(store).replay("paper-1")
    audit = PaperShadowAudit(store).audit("paper-1")
    assert replay.event_count == 1
    assert replay.broker_fills_reproduced is False
    assert audit.status == "UNAVAILABLE"
    assert audit.decision_envelopes == 1
    compared = PaperShadowAudit(store).audit(
        "paper-1",
        recalculated_decision_hashes={
            "decision-1": stable_hash({"envelope_id": "decision-1"})
        },
    )
    assert compared.status == "IN_SYNC"
    event_path = tmp_path / "paper" / "paper-1" / "events" / "event-1.json"
    event_path.write_text('{"event_id":"tampered"}', encoding="utf-8")
    with pytest.raises(BrokerIntegrityError, match="checksum"):
        store.inspect("paper-1")


def test_session_manifest_never_accepts_armed_execution() -> None:
    with pytest.raises(ValueError, match="unarmed"):
        PaperSessionManifest(
            session_id="bad",
            created_at=datetime.now(timezone.utc),
            code_sha="sha",
            mode=PaperMode.PAPER_READ_ONLY,
            broker_adapter_name="adapter",
            broker_adapter_version="1",
            official_sdk_version=None,
            server_version=None,
            account_hash="a" * 64,
            account_masked="masked",
            config_hashes=(("risk", "b" * 64),),
            ml_model_ids=(),
            paper_execution_armed=True,
        )
