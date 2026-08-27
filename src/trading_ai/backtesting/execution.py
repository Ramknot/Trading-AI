"""Deterministic bar-based simulated execution and commission models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from trading_ai.backtesting.exceptions import BacktestExecutionError
from trading_ai.backtesting.models import (
    BacktestConfig,
    BacktestOrder,
    CommissionConfig,
    Fill,
    OrderStatus,
)
from trading_ai.core.models import MarketBar, OrderSide, OrderType


BPS = Decimal("10000")
QUANTUM = Decimal("0.00000001")


def _q(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


class CommissionModel(ABC):
    @abstractmethod
    def calculate(self, notional: Decimal) -> Decimal:
        """Return a non-negative commission for one complete fill."""


class ConfigurableCommissionModel(CommissionModel):
    def __init__(self, config: CommissionConfig) -> None:
        self.config = config

    def calculate(self, notional: Decimal) -> Decimal:
        if notional <= Decimal("0"):
            raise BacktestExecutionError("commission notional must be positive")
        variable = notional * self.config.percentage_bps / BPS
        return _q(max(self.config.fixed + variable, self.config.minimum))


class ExecutionModel(ABC):
    @abstractmethod
    def try_fill(
        self, order: BacktestOrder, bar: MarketBar, fill_id: str
    ) -> Fill | None:
        """Return one complete fill or None when the order is not reachable."""


class BarExecutionModel(ExecutionModel):
    """Next-bar-open MARKET fills and conservative touch-triggered LIMIT fills."""

    def __init__(
        self,
        config: BacktestConfig,
        commission_model: CommissionModel | None = None,
    ) -> None:
        self.config = config
        self.commission_model = commission_model or ConfigurableCommissionModel(
            config.commission
        )

    def try_fill(
        self, order: BacktestOrder, bar: MarketBar, fill_id: str
    ) -> Fill | None:
        if order.status is not OrderStatus.PENDING:
            raise BacktestExecutionError("only pending orders can be filled")
        if order.created_at >= bar.timestamp:
            return None
        if order.symbol != bar.symbol or order.timeframe != bar.timeframe:
            return None
        total_rate = (self.config.spread_bps + self.config.slippage_bps) / BPS
        if order.order_type is OrderType.MARKET:
            reference_price = bar.open
            if order.side is OrderSide.BUY:
                execution_price = reference_price * (Decimal("1") + total_rate)
            else:
                execution_price = reference_price * (Decimal("1") - total_rate)
        elif order.order_type is OrderType.LIMIT:
            if order.limit_price is None:
                raise BacktestExecutionError("LIMIT order has no limit_price")
            if order.side is OrderSide.BUY:
                reference_price = order.limit_price / (Decimal("1") + total_rate)
                if bar.low > reference_price:
                    return None
            else:
                reference_price = order.limit_price / (Decimal("1") - total_rate)
                if bar.high < reference_price:
                    return None
            execution_price = order.limit_price
        else:
            raise BacktestExecutionError(
                f"unsupported simulated order type {order.order_type.value}"
            )
        reference_price = _q(reference_price)
        execution_price = _q(execution_price)
        reference_notional = reference_price * order.quantity
        spread_cost = _q(
            reference_notional * self.config.spread_bps / BPS
        )
        slippage_cost = _q(
            reference_notional * self.config.slippage_bps / BPS
        )
        commission = self.commission_model.calculate(
            execution_price * order.quantity
        )
        return Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference_price,
            price=execution_price,
            timestamp=bar.timestamp,
            commission=commission,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
        )
