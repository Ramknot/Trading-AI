"""Optional official ``ibapi`` bridge with controlled thread and event queue.

The IBKR SDK is deliberately not a project dependency or vendored source.  Users
install it from IBKR after accepting the applicable official license.  Import is
lazy, so all default tests and research remain offline.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from trading_ai.brokers.exceptions import BrokerUnavailableError, IBKRSDKUnavailableError
from trading_ai.brokers.ibkr.orders import IBKROrderSpec


IBKREventSink = Callable[[str, dict[str, Any]], None]


class IBKRClientPort(ABC):
    """Narrow asynchronous port implemented by the official SDK and CI fakes."""

    @abstractmethod
    def connect(self, host: str, port: int, client_id: int, timeout: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def connected(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def account_ids(self) -> tuple[str, ...]:
        """Transient raw IDs; callers must hash immediately and never persist them."""

    @property
    @abstractmethod
    def next_order_id(self) -> int | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def sdk_version(self) -> str | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def server_version(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def request_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def request_contract_details(self, request_id: int, contract: dict[str, object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def place_order(
        self,
        order_id: int,
        contract: dict[str, object],
        order: IBKROrderSpec,
        *,
        account_id: str,
        client_order_key: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def request_current_time(self) -> None:
        raise NotImplementedError


class _Pacer:
    def __init__(self, max_messages_per_second: int) -> None:
        self.interval = 1.0 / max_messages_per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class OfficialIBAPIClient(IBKRClientPort):
    """Thin adapter over IBKR's callback API; no login or 2FA automation."""

    def __init__(
        self,
        event_sink: IBKREventSink,
        *,
        max_messages_per_second: int = 20,
        expected_sdk_version: str = "10.50",
    ) -> None:
        self._event_sink = event_sink
        self._event_queue: queue.Queue[tuple[str, dict[str, Any]] | None] = (
            queue.Queue(maxsize=10_000)
        )
        self._dispatch_thread: threading.Thread | None = None
        self._dispatcher_error: Exception | None = None
        self._pacer = _Pacer(max_messages_per_second)
        self._expected_sdk_version = expected_sdk_version
        self._app: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._account_ids: tuple[str, ...] = ()
        self._next_order_id: int | None = None
        self._sdk_version: str | None = None
        self._server_version: str | None = None

    def _enqueue_event(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._event_queue.put_nowait((kind, payload))
        except queue.Full:
            self._dispatcher_error = BrokerUnavailableError(
                "IBKR callback queue capacity was exceeded"
            )
            self._ready.clear()

    def _dispatch_events(self) -> None:
        while True:
            item = self._event_queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                self._event_sink(kind, payload)
            except Exception as exc:  # fail closed; never kill the callback reader silently
                self._dispatcher_error = exc
                self._ready.clear()
            finally:
                self._event_queue.task_done()

    def _start_dispatcher(self) -> None:
        self._dispatcher_error = None
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_events,
            name="trading-ai-ibkr-events",
            daemon=True,
        )
        self._dispatch_thread.start()

    def _build_app(self) -> Any:
        try:
            package = importlib.import_module("ibapi")
            client_module = importlib.import_module("ibapi.client")
            wrapper_module = importlib.import_module("ibapi.wrapper")
        except ImportError as exc:
            raise IBKRSDKUnavailableError(
                "official IBKR TWS API is not installed; install it separately from IBKR after license acceptance"
            ) from exc
        try:
            self._sdk_version = importlib.metadata.version("ibapi")
        except importlib.metadata.PackageNotFoundError:
            self._sdk_version = str(getattr(package, "__version__", self._expected_sdk_version))
        outer = self
        EClient = client_module.EClient
        EWrapper = wrapper_module.EWrapper

        class Application(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                EWrapper.__init__(self)
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                outer._next_order_id = int(orderId)
                outer._server_version = str(self.serverVersion())
                outer._ready.set()
                outer._enqueue_event("CONNECTED", {"next_order_id": int(orderId)})

            def managedAccounts(self, accountsList: str) -> None:  # noqa: N802
                outer._account_ids = tuple(
                    sorted(value.strip() for value in accountsList.split(",") if value.strip())
                )
                outer._enqueue_event("ACCOUNT_IDENTIFIERS", {"count": len(outer._account_ids)})

            def accountSummary(self, reqId, account, tag, value, currency) -> None:  # noqa: N802
                del account
                outer._enqueue_event(
                    "ACCOUNT_SUMMARY",
                    {"request_id": int(reqId), "tag": str(tag), "value": str(value), "currency": str(currency)},
                )

            def accountSummaryEnd(self, reqId) -> None:  # noqa: N802
                outer._enqueue_event("ACCOUNT_SUMMARY_END", {"request_id": int(reqId)})

            def position(self, account, contract, position, avgCost) -> None:
                del account
                outer._enqueue_event(
                    "POSITION",
                    {
                        "symbol": str(contract.symbol),
                        "local_symbol": str(contract.localSymbol),
                        "sec_type": str(contract.secType),
                        "exchange": str(contract.exchange),
                        "currency": str(contract.currency),
                        "con_id": int(contract.conId),
                        "quantity": str(position),
                        "average_cost": str(avgCost),
                    },
                )

            def positionEnd(self) -> None:  # noqa: N802
                outer._enqueue_event("POSITIONS_END", {})

            def openOrder(self, orderId, contract, order, orderState) -> None:  # noqa: N802
                outer._enqueue_event(
                    "OPEN_ORDER",
                    {
                        "broker_order_id": str(orderId),
                        "perm_id": str(order.permId),
                        "client_order_key": str(order.orderRef or ""),
                        "symbol": str(contract.symbol),
                        "currency": str(contract.currency),
                        "side": str(order.action),
                        "order_type": str(order.orderType),
                        "quantity": str(order.totalQuantity),
                        "limit_price": str(order.lmtPrice),
                        "tif": str(order.tif),
                        "state": str(orderState.status),
                    },
                )

            def openOrderEnd(self) -> None:  # noqa: N802
                outer._enqueue_event("OPEN_ORDERS_END", {})

            def completedOrder(self, contract, order, orderState) -> None:  # noqa: N802
                outer._enqueue_event(
                    "COMPLETED_ORDER",
                    {
                        "broker_order_id": str(order.orderId),
                        "perm_id": str(order.permId),
                        "client_order_key": str(order.orderRef or ""),
                        "symbol": str(contract.symbol),
                        "currency": str(contract.currency),
                        "side": str(order.action),
                        "order_type": str(order.orderType),
                        "quantity": str(order.totalQuantity),
                        "limit_price": str(order.lmtPrice),
                        "tif": str(order.tif),
                        "state": str(orderState.status),
                    },
                )

            def completedOrdersEnd(self) -> None:  # noqa: N802
                outer._enqueue_event("COMPLETED_ORDERS_END", {})

            def orderStatus(
                self, orderId, status, filled, remaining, avgFillPrice, permId,
                parentId, lastFillPrice, clientId, whyHeld, mktCapPrice,
            ) -> None:  # noqa: N802
                del parentId, clientId, whyHeld, mktCapPrice
                outer._enqueue_event(
                    "ORDER_STATUS",
                    {
                        "broker_order_id": str(orderId), "status": str(status),
                        "filled": str(filled), "remaining": str(remaining),
                        "average_fill_price": str(avgFillPrice), "last_fill_price": str(lastFillPrice),
                        "perm_id": str(permId),
                    },
                )

            def execDetails(self, reqId, contract, execution) -> None:  # noqa: N802
                outer._enqueue_event(
                    "EXECUTION",
                    {
                        "request_id": int(reqId), "exec_id": str(execution.execId),
                        "broker_order_id": str(execution.orderId), "perm_id": str(execution.permId),
                        "client_order_key": str(execution.orderRef or ""),
                        "symbol": str(contract.symbol), "currency": str(contract.currency),
                        "side": str(execution.side),
                        "quantity": str(execution.shares), "cumulative_quantity": str(execution.cumQty),
                        "price": str(execution.price), "exchange": str(execution.exchange),
                        "time": str(execution.time),
                    },
                )

            def execDetailsEnd(self, reqId) -> None:  # noqa: N802
                outer._enqueue_event("EXECUTIONS_END", {"request_id": int(reqId)})

            def commissionReport(self, report) -> None:  # noqa: N802
                outer._enqueue_event(
                    "COMMISSION_REPORT",
                    {"exec_id": str(report.execId), "commission": str(report.commission), "currency": str(report.currency)},
                )

            def contractDetails(self, reqId, details) -> None:  # noqa: N802
                contract = details.contract
                outer._enqueue_event(
                    "CONTRACT_DETAILS",
                    {
                        "request_id": int(reqId), "con_id": int(contract.conId),
                        "symbol": str(contract.symbol), "local_symbol": str(contract.localSymbol),
                        "sec_type": str(contract.secType), "exchange": str(contract.exchange),
                        "primary_exchange": str(contract.primaryExchange), "currency": str(contract.currency),
                    },
                )

            def contractDetailsEnd(self, reqId) -> None:  # noqa: N802
                outer._enqueue_event("CONTRACT_DETAILS_END", {"request_id": int(reqId)})

            def currentTime(self, epoch: int) -> None:  # noqa: N802
                outer._enqueue_event("CURRENT_TIME", {"epoch": int(epoch)})

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:
                del errorString, advancedOrderRejectJson
                outer._enqueue_event("ERROR", {"request_id": int(reqId), "code": int(errorCode)})

            def connectionClosed(self) -> None:  # noqa: N802
                outer._enqueue_event("DISCONNECTED", {})

        return Application()

    def connect(self, host: str, port: int, client_id: int, timeout: float) -> None:
        if self.connected:
            raise BrokerUnavailableError("IBKR TWS API client is already connected")
        self._ready.clear()
        # Never reuse identity or order-sequence evidence from a previous socket.
        self._account_ids = ()
        self._next_order_id = None
        self._server_version = None
        self._start_dispatcher()
        self._app = self._build_app()
        # The official Python EClient.connect contract does not promise a
        # truthy return value; socket state and the nextValidId handshake are
        # the authoritative checks.
        self._app.connect(host, port, clientId=client_id)
        if not self._app.isConnected():
            raise BrokerUnavailableError("IBKR TWS API socket connection was refused")
        self._thread = threading.Thread(
            target=self._app.run,
            name="trading-ai-ibkr-callbacks",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.disconnect()
            raise BrokerUnavailableError("IBKR handshake timed out before nextValidId")

    def disconnect(self) -> None:
        app, thread = self._app, self._thread
        if app is not None:
            app.disconnect()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        dispatch = self._dispatch_thread
        if dispatch is not None:
            self._event_queue.put(None)
            if dispatch.is_alive() and dispatch is not threading.current_thread():
                dispatch.join(timeout=5.0)
        self._app = None
        self._thread = None
        self._dispatch_thread = None
        self._ready.clear()

    @property
    def connected(self) -> bool:
        return bool(
            self._app is not None
            and self._app.isConnected()
            and self._ready.is_set()
            and self._dispatcher_error is None
        )

    @property
    def account_ids(self) -> tuple[str, ...]:
        return self._account_ids

    @property
    def next_order_id(self) -> int | None:
        return self._next_order_id

    @property
    def sdk_version(self) -> str | None:
        return self._sdk_version

    @property
    def server_version(self) -> str | None:
        return self._server_version

    def _require(self) -> Any:
        if not self.connected:
            raise BrokerUnavailableError("IBKR TWS API is not connected and handshaken")
        self._pacer.wait()
        return self._app

    def request_state(self) -> None:
        app = self._require()
        app.reqAccountSummary(9001, "All", "AccountType,NetLiquidation,TotalCashValue")
        self._pacer.wait(); app.reqPositions()
        self._pacer.wait(); app.reqOpenOrders()
        self._pacer.wait(); app.reqCompletedOrders(True)
        execution_module = importlib.import_module("ibapi.execution")
        self._pacer.wait(); app.reqExecutions(9002, execution_module.ExecutionFilter())

    def request_contract_details(self, request_id: int, contract: dict[str, object]) -> None:
        app = self._require()
        contract_module = importlib.import_module("ibapi.contract")
        value = contract_module.Contract()
        for key, item in contract.items():
            setattr(value, key, item)
        app.reqContractDetails(request_id, value)

    def place_order(
        self,
        order_id: int,
        contract: dict[str, object],
        order: IBKROrderSpec,
        *,
        account_id: str,
        client_order_key: str,
    ) -> None:
        app = self._require()
        contract_module = importlib.import_module("ibapi.contract")
        order_module = importlib.import_module("ibapi.order")
        ib_contract = contract_module.Contract()
        for key, item in contract.items():
            setattr(ib_contract, key, item)
        ib_order = order_module.Order()
        ib_order.action = order.action
        ib_order.orderType = order.order_type
        ib_order.totalQuantity = order.total_quantity
        ib_order.tif = order.tif
        ib_order.transmit = order.transmit
        ib_order.orderRef = client_order_key
        ib_order.account = account_id
        if order.limit_price is not None:
            ib_order.lmtPrice = float(order.limit_price)
        app.placeOrder(order_id, ib_contract, ib_order)
        self._next_order_id = order_id + 1

    def cancel_order(self, order_id: int) -> None:
        app = self._require()
        app.cancelOrder(order_id, "")

    def request_current_time(self) -> None:
        self._require().reqCurrentTime()
