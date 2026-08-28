"""Point-in-time currency conversion contracts; no network/provider implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from trading_ai.portfolio.exceptions import CurrencyConversionError


class CurrencyConverter(ABC):
    @abstractmethod
    def convert(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        timestamp: datetime,
    ) -> Decimal:
        """Convert with a rate known at timestamp or fail explicitly."""

    @abstractmethod
    def has_rate(
        self, from_currency: str, to_currency: str, timestamp: datetime
    ) -> bool:
        """Return whether a valid point-in-time conversion is available."""


class SameCurrencyConverter(CurrencyConverter):
    """Production-safe default: identity only, never implicit 1:1 FX."""

    def has_rate(
        self, from_currency: str, to_currency: str, timestamp: datetime
    ) -> bool:
        del timestamp
        return from_currency.upper() == to_currency.upper()

    def convert(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        timestamp: datetime,
    ) -> Decimal:
        if not self.has_rate(from_currency, to_currency, timestamp):
            raise CurrencyConversionError(
                f"no point-in-time FX rate for {from_currency}/{to_currency}"
            )
        return amount
