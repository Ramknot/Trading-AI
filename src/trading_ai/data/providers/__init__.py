"""DataProvider adapters shipped with Lot 1."""

from trading_ai.data.providers.fake import FakeDataProvider
from trading_ai.data.providers.yahoo import YahooFinanceProvider

__all__ = ["FakeDataProvider", "YahooFinanceProvider"]
