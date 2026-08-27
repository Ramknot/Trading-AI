"""Configuration-driven asset-group concentration guard."""

from __future__ import annotations

from decimal import Decimal

from trading_ai.risk.config import RiskAssetGroups


ZERO = Decimal("0")


class RiskGroupResolver:
    def __init__(self, groups: RiskAssetGroups) -> None:
        self._groups = groups

    def resolve(self, symbol: str) -> str | None:
        return self._groups.group_for(symbol)

    def exposure_value(
        self,
        group: str,
        position_values: dict[str, Decimal],
    ) -> Decimal:
        return sum(
            (
                value
                for symbol, value in position_values.items()
                if self.resolve(symbol) == group
            ),
            ZERO,
        )

    def max_increment(
        self,
        *,
        group: str,
        position_values: dict[str, Decimal],
        equity: Decimal,
        limit: Decimal,
        expected_price: Decimal,
    ) -> Decimal:
        available = max(
            ZERO,
            equity * limit - self.exposure_value(group, position_values),
        )
        return available / expected_price
