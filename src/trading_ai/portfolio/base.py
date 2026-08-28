"""Provider- and broker-neutral target portfolio construction contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from trading_ai.portfolio.models import (
    PortfolioContext,
    PortfolioDecisionBatch,
    PortfolioPlanResult,
    StrategySleeveState,
)


class PortfolioEngine(ABC):
    """Propose allocations; it never authorizes or transmits an order."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Stable engine identity."""

    @property
    @abstractmethod
    def engine_version(self) -> str:
        """Stable implementation version."""

    @property
    @abstractmethod
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        """Deterministically sorted configuration provenance."""

    @property
    @abstractmethod
    def config_hash(self) -> str:
        """SHA-256 of normalized portfolio configuration and metadata."""

    @property
    @abstractmethod
    def sleeve_state(self) -> tuple[StrategySleeveState, ...]:
        """Current logical sleeve attribution, aggregated physically by symbol."""

    @abstractmethod
    def reset(self, timestamp: datetime, equity: Decimal) -> None:
        """Start a deterministic run with no implicit prior allocation."""

    @abstractmethod
    def plan(
        self,
        batch: PortfolioDecisionBatch,
        context: PortfolioContext,
    ) -> PortfolioPlanResult:
        """Build one point-in-time proposal for a complete UTC opportunity batch."""
