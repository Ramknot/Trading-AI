"""Offline regression fixtures for the official SDK callback ABI."""

from types import SimpleNamespace
import sys

import pytest

from trading_ai.brokers.ibkr.client import OfficialIBAPIClient


@pytest.fixture
def sdk_app(monkeypatch):
    class Wrapper:
        pass

    class Client:
        def __init__(self, wrapper):
            self.wrapper = wrapper

    monkeypatch.setitem(sys.modules, "ibapi", SimpleNamespace(__version__="10.50.2"))
    monkeypatch.setitem(sys.modules, "ibapi.client", SimpleNamespace(EClient=Client))
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", SimpleNamespace(EWrapper=Wrapper))
    observed = []
    client = OfficialIBAPIClient(lambda kind, payload: observed.append((kind, payload)))
    client._start_dispatcher()
    try:
        yield client, client._build_app(), observed
    finally:
        client.disconnect()


@pytest.mark.parametrize("args", [
    (2104, "sensitive account text"),
    (2104, "sensitive account text", "sensitive reject JSON"),
    (1780000000000, 2104, "sensitive account text"),
    (1780000000000, 2104, "sensitive account text", "sensitive reject JSON"),
])
def test_error_callback_accepts_legacy_and_timestamped_sdk_without_leaking_text(sdk_app, args):
    client, app, observed = sdk_app
    app.error(-1, *args)
    client._event_queue.join()
    kind, payload = observed[0]
    assert kind == "ERROR"
    assert payload["code"] == 2104
    assert payload["request_id"] == -1
    assert "sensitive" not in repr(observed)
    assert payload.get("error_time_ms") == (1780000000000 if len(args) >= 3 and isinstance(args[1], int) else None)


def test_combined_commission_callback_preserves_single_broker_amount(sdk_app):
    client, app, observed = sdk_app
    app.commissionAndFeesReport(SimpleNamespace(execId="exec-1", commissionAndFees=1.25, currency="USD"))
    client._event_queue.join()
    assert observed == [("COMMISSION_REPORT", {"exec_id": "exec-1", "commission": "1.25", "currency": "USD"})]


def test_legacy_commission_callback_is_still_supported(sdk_app):
    client, app, observed = sdk_app
    app.commissionReport(SimpleNamespace(execId="exec-1", commission=1.25, currency="USD"))
    client._event_queue.join()
    assert observed[0][1]["commission"] == "1.25"


def test_reader_failure_invalidates_socket_without_exposing_exception():
    observed = []
    client = OfficialIBAPIClient(lambda kind, payload: observed.append((kind, payload)))

    def fail():
        raise TypeError("private broker message")

    client._app = SimpleNamespace(run=fail, isConnected=lambda: True, disconnect=lambda: None)
    client._start_dispatcher()
    client._ready.set()
    try:
        assert client.connected
        client._run_reader()
        client._event_queue.join()
        assert not client.connected
        assert [kind for kind, _ in observed] == ["ERROR", "DISCONNECTED"]
        assert "private" not in repr(observed)
        assert "private" not in str(client._dispatcher_error)
    finally:
        client.disconnect()


def test_repeated_state_reads_release_only_data_subscriptions(monkeypatch):
    """Account-wide reads must not bind, transmit or cancel any order."""
    calls = []
    names = ("reqAccountSummary", "reqPositions", "reqAllOpenOrders", "reqCompletedOrders",
             "reqExecutions", "cancelAccountSummary", "cancelPositions")
    app = SimpleNamespace(**{name: (lambda *args, name=name: calls.append((name, args))) for name in names})
    client = OfficialIBAPIClient(lambda kind, payload: None)
    monkeypatch.setattr(client, "_require", lambda: app)
    monkeypatch.setattr(client._pacer, "wait", lambda: None)
    monkeypatch.setitem(sys.modules, "ibapi.execution", SimpleNamespace(ExecutionFilter=lambda: "FILTER"))
    client.request_state()
    client.request_state()
    assert [name for name, args in calls] == [
        "reqAccountSummary", "reqPositions", "reqAllOpenOrders", "reqCompletedOrders", "reqExecutions",
        "cancelAccountSummary", "cancelPositions",
        "reqAccountSummary", "reqPositions", "reqAllOpenOrders", "reqCompletedOrders", "reqExecutions",
    ]
    assert all(args == (False,) for name, args in calls if name == "reqCompletedOrders")
