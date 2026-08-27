"""Central registry for Lot 3 baseline selection and introspection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from trading_ai.backtesting.strategy import BacktestStrategy
from trading_ai.strategies.baselines import (
    BreakoutStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from trading_ai.strategies.config import (
    BreakoutConfig,
    MomentumConfig,
    TrendConfig,
)


BaselineConfig: TypeAlias = TrendConfig | MomentumConfig | BreakoutConfig
StrategyFactory: TypeAlias = Callable[
    [Sequence[str], str, BaselineConfig | None], BacktestStrategy
]


@dataclass(frozen=True, slots=True)
class StrategyDescriptor:
    name: str
    version: str
    description: str
    default_parameters: tuple[tuple[str, str], ...]


def _trend_factory(
    symbols: Sequence[str], timeframe: str, config: BaselineConfig | None
) -> BacktestStrategy:
    if config is not None and not isinstance(config, TrendConfig):
        raise TypeError("trend requires TrendConfig")
    return TrendFollowingStrategy(symbols, timeframe, config)


def _momentum_factory(
    symbols: Sequence[str], timeframe: str, config: BaselineConfig | None
) -> BacktestStrategy:
    if config is not None and not isinstance(config, MomentumConfig):
        raise TypeError("momentum requires MomentumConfig")
    return MomentumStrategy(symbols, timeframe, config)


def _breakout_factory(
    symbols: Sequence[str], timeframe: str, config: BaselineConfig | None
) -> BacktestStrategy:
    if config is not None and not isinstance(config, BreakoutConfig):
        raise TypeError("breakout requires BreakoutConfig")
    return BreakoutStrategy(symbols, timeframe, config)


class StrategyRegistry:
    """Small closed registry; selection does not leak CLI branching into strategies."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[StrategyDescriptor, StrategyFactory]] = {}

    def register(
        self, descriptor: StrategyDescriptor, factory: StrategyFactory
    ) -> None:
        if descriptor.name in self._entries:
            raise ValueError(f"strategy already registered: {descriptor.name}")
        self._entries[descriptor.name] = (descriptor, factory)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    @property
    def descriptors(self) -> tuple[StrategyDescriptor, ...]:
        return tuple(self._entries[name][0] for name in self.names)

    def create(
        self,
        name: str,
        *,
        symbols: Sequence[str],
        timeframe: str,
        config: BaselineConfig | None = None,
    ) -> BacktestStrategy:
        try:
            factory = self._entries[name][1]
        except KeyError as exc:
            raise ValueError(f"unknown baseline strategy: {name}") from exc
        return factory(symbols, timeframe, config)


def _default_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(
        StrategyDescriptor(
            name="trend",
            version=TrendFollowingStrategy.version,
            description="Long-only fast/slow EMA trend following",
            default_parameters=TrendConfig().to_parameters(),
        ),
        _trend_factory,
    )
    registry.register(
        StrategyDescriptor(
            name="momentum",
            version=MomentumStrategy.version,
            description="Exact-timestamp top-K cross-sectional momentum",
            default_parameters=MomentumConfig().to_parameters(),
        ),
        _momentum_factory,
    )
    registry.register(
        StrategyDescriptor(
            name="breakout",
            version=BreakoutStrategy.version,
            description="Previous-range long-only breakout",
            default_parameters=BreakoutConfig().to_parameters(),
        ),
        _breakout_factory,
    )
    return registry


BASELINE_STRATEGIES = _default_registry()
