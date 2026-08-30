"""Provider- and broker-neutral transaction-cost engine contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from trading_ai.costs.models import (
    ActualTradingCost,
    CostReconciliation,
    OperatingCostBreakdown,
    PreTradeCostEstimate,
    PreTradeCostRequest,
    TariffStatus,
)

if TYPE_CHECKING:
    from trading_ai.backtesting.models import Fill


class TransactionCostEngine(ABC):
    """Estimate at decision time, record at fill time, and never execute trades."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Stable engine identity."""

    @property
    @abstractmethod
    def engine_version(self) -> str:
        """Stable semantic version."""

    @property
    @abstractmethod
    def config_hash(self) -> str:
        """SHA-256 of normalized configuration and referenced schedules."""

    @property
    @abstractmethod
    def config_parameters(self) -> tuple[tuple[str, str], ...]:
        """Deterministic human-readable configuration provenance."""

    @property
    @abstractmethod
    def tariff_profile_id(self) -> str:
        """Explicit dated tariff profile used for the estimates."""

    @property
    @abstractmethod
    def tariff_status(self) -> TariffStatus:
        """Verification lifecycle of the selected tariff profile."""

    @abstractmethod
    def operating_costs(
        self, period_start: datetime, period_end: datetime
    ) -> OperatingCostBreakdown:
        """Return period-level fixed costs without allocating them to fills."""

    @abstractmethod
    def estimate(self, request: PreTradeCostRequest) -> PreTradeCostEstimate:
        """Estimate costs using only information observable at request.timestamp."""

    @abstractmethod
    def actualize(
        self, fill: Fill, estimate: PreTradeCostEstimate
    ) -> ActualTradingCost:
        """Calculate actual modeled costs without mutating the historical estimate."""

    @abstractmethod
    def reconcile(
        self, estimate: PreTradeCostEstimate, actual: ActualTradingCost
    ) -> CostReconciliation:
        """Compare immutable estimate and actual records component by component."""
