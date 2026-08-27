"""Look-ahead-safe strategy boundary used only by historical simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Sequence

from trading_ai.backtesting.models import OrderIntent, StrategyContext, StrategySignal
from trading_ai.core.models import OrderSide, OrderType

if TYPE_CHECKING:
    from trading_ai.regimes.models import ActivationDecision


class BacktestStrategy(ABC):
    """Consume a read-only past-and-present context and emit order intents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable strategy name recorded in results."""

    @property
    def version(self) -> str:
        return "1"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        """Stable, serializable parameters included in the run identity."""

        return ()

    def reset(self) -> None:
        """Reset deterministic run state before the first market event."""

    @property
    def signals(self) -> tuple[StrategySignal, ...]:
        """Explainable signals emitted during the current deterministic run."""

        return ()

    def on_activation_decision(self, decision: ActivationDecision) -> None:
        """Observe policy outcome only to maintain deterministic proposal state."""

        del decision

    @abstractmethod
    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        """Return intents using only context.history, which never contains future bars."""


class BuyAndHoldDemoStrategy(BacktestStrategy):
    """Technical CLI demo; not a Lot 3 investment strategy."""

    def __init__(self, symbol: str, quantity: Decimal) -> None:
        if not symbol.strip():
            raise ValueError("symbol must not be empty")
        if quantity <= Decimal("0"):
            raise ValueError("quantity must be positive")
        self.symbol = symbol
        self.quantity = quantity
        self._submitted = False

    @property
    def name(self) -> str:
        return "buy-and-hold-demo"

    def reset(self) -> None:
        self._submitted = False

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return (("quantity", str(self.quantity)), ("symbol", self.symbol))

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        if self._submitted or context.current_bar.symbol != self.symbol:
            return ()
        self._submitted = True
        return (
            OrderIntent(
                symbol=self.symbol,
                side=OrderSide.BUY,
                quantity=self.quantity,
                order_type=OrderType.MARKET,
                timeframe=context.current_bar.timeframe,
            ),
        )
