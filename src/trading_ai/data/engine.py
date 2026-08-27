"""Provider-neutral orchestration, cache policy, lineage, and 4h derivation."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Sequence, TypeVar

from trading_ai.core.config import load_profile
from trading_ai.core.models import MarketBar, TradingProfileName
from trading_ai.data.base import DataEngine as DataEngineContract
from trading_ai.data.base import DataProvider
from trading_ai.data.calendar import MarketCalendarService
from trading_ai.data.exceptions import (
    CacheMissError,
    DataProviderTemporaryError,
    DataUnavailableError,
    DataValidationError,
)
from trading_ai.data.models import (
    CacheMode,
    CorporateAction,
    DataFetchResult,
    DataKind,
    DatasetInspection,
    DatasetManifest,
    InstrumentMetadata,
    MarketDataRequest,
)
from trading_ai.data.quality import assess_normalized_bars, normalize_provider_bars
from trading_ai.data.resampling import resample_1h_to_4h
from trading_ai.data.storage import ParquetDataStore


T = TypeVar("T")


class DataEngine(DataEngineContract):
    """Historical Data Engine with explicit provider and local-store boundaries."""

    SUPPORTED_TIMEFRAMES = frozenset({"1h", "4h", "1d"})

    def __init__(
        self,
        provider: DataProvider,
        store: ParquetDataStore | None = None,
        *,
        calendar_service: MarketCalendarService | None = None,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.1,
    ) -> None:
        if not isinstance(provider, DataProvider):
            raise TypeError("provider must implement DataProvider")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        self.provider = provider
        self.store = store or ParquetDataStore()
        self.calendar = calendar_service or MarketCalendarService()
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def _retry(self, operation: Callable[[], T]) -> T:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except DataProviderTemporaryError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay_seconds * (2**attempt))
        raise AssertionError("bounded retry loop exhausted unexpectedly")

    @staticmethod
    def _metadata_from_manifest(manifest: DatasetManifest) -> InstrumentMetadata:
        return InstrumentMetadata(
            symbol=manifest.symbol,
            exchange=manifest.exchange,
            exchange_timezone=manifest.source_timezone,
            calendar=manifest.calendar,
            source=manifest.provider,
        )

    @staticmethod
    def _validate_profile_request(
        profile_name: str | TradingProfileName,
        symbol: str,
        timeframe: str,
    ) -> None:
        profile = load_profile(profile_name)
        if symbol not in profile.asset_universe:
            raise DataValidationError(
                f"symbol {symbol!r} is not in profile {profile.name.value!r}"
            )
        if timeframe not in profile.timeframes:
            raise DataValidationError(
                f"timeframe {timeframe!r} is not in profile {profile.name.value!r}"
            )

    def _cached_result(
        self,
        manifest: DatasetManifest,
    ) -> DataFetchResult:
        bars = self.store.read_bars(manifest)
        metadata = self._metadata_from_manifest(manifest)
        quality = assess_normalized_bars(
            bars,
            metadata,
            manifest.requested_start,
            manifest.requested_end,
            calendar_service=self.calendar,
            warnings=manifest.warnings,
        )
        actions_manifest = self.store.find_exact(
            provider=(
                self.provider.name
                if manifest.data_kind is DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE
                else manifest.provider
            ),
            symbol=manifest.symbol,
            timeframe="actions",
            start=manifest.requested_start,
            end=manifest.requested_end,
            data_kind=DataKind.CORPORATE_ACTIONS,
        )
        actions: tuple[CorporateAction, ...] = ()
        if actions_manifest is not None:
            actions = self.store.read_corporate_actions(actions_manifest)
        return DataFetchResult(
            bars=bars,
            corporate_actions=actions,
            manifest=manifest,
            corporate_actions_manifest=actions_manifest,
            quality_report=quality,
            cache_hit=True,
        )

    def fetch(
        self,
        *,
        profile_name: str | TradingProfileName = TradingProfileName.BALANCED,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        cache_mode: CacheMode = CacheMode.CACHE_FIRST,
    ) -> DataFetchResult:
        """Fetch, validate, store, and reuse exactly one configured series."""

        try:
            cache_mode = CacheMode(cache_mode)
        except ValueError as exc:
            raise DataValidationError(f"unknown cache mode {cache_mode!r}") from exc
        self._validate_profile_request(profile_name, symbol, timeframe)
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise DataValidationError(f"unsupported timeframe {timeframe!r}")
        request = MarketDataRequest(symbol, timeframe, start, end)
        data_kind = (
            DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE
            if timeframe == "4h"
            else DataKind.RAW_WITH_ADJUSTED_CLOSE
        )
        storage_provider = "derived" if timeframe == "4h" else self.provider.name
        if cache_mode is not CacheMode.REFRESH:
            cached = self.store.find_exact(
                provider=storage_provider,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                data_kind=data_kind,
            )
            if cached is not None:
                return self._cached_result(cached)
            if cache_mode is CacheMode.CACHE_ONLY and timeframe != "4h":
                raise CacheMissError(
                    f"no exact cached dataset for {symbol} {timeframe}"
                )
        if timeframe == "4h":
            return self._derive_4h(
                profile_name=profile_name,
                symbol=symbol,
                start=start,
                end=end,
                cache_mode=cache_mode,
            )
        return self._fetch_native(request)

    def _fetch_native(self, request: MarketDataRequest) -> DataFetchResult:
        provider_result = self._retry(lambda: self.provider.fetch_bars(request))
        if provider_result.metadata.symbol != request.symbol:
            raise DataValidationError("provider metadata symbol does not match request")
        bars, quality = normalize_provider_bars(
            provider_result.bars,
            provider_result.metadata,
            request.start,
            request.end,
            calendar_service=self.calendar,
            provider_warnings=provider_result.warnings,
            expected_timeframe=request.timeframe,
        )
        if not bars:
            raise DataUnavailableError(
                f"no normalized data for {request.symbol} {request.timeframe}"
            )
        actions = self._retry(
            lambda: self.provider.fetch_corporate_actions(
                request.symbol, request.start, request.end
            )
        )
        manifest = self.store.save_bars(
            bars=bars,
            request=request,
            metadata=provider_result.metadata,
            provider=self.provider.name,
            provider_version=self.provider.version,
            data_kind=DataKind.RAW_WITH_ADJUSTED_CLOSE,
            warnings=quality.warnings,
        )
        actions_manifest = self.store.save_corporate_actions(
            actions=actions,
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            metadata=provider_result.metadata,
            provider=self.provider.name,
            provider_version=self.provider.version,
        )
        return DataFetchResult(
            bars=bars,
            corporate_actions=actions,
            manifest=manifest,
            corporate_actions_manifest=actions_manifest,
            quality_report=quality,
            cache_hit=False,
        )

    def _derive_4h(
        self,
        *,
        profile_name: str | TradingProfileName,
        symbol: str,
        start: datetime,
        end: datetime,
        cache_mode: CacheMode,
    ) -> DataFetchResult:
        source_result = self.fetch(
            profile_name=profile_name,
            symbol=symbol,
            timeframe="1h",
            start=start,
            end=end,
            cache_mode=cache_mode,
        )
        metadata = self._metadata_from_manifest(source_result.manifest)
        derived = resample_1h_to_4h(
            source_result.bars,
            metadata,
            calendar_service=self.calendar,
        )
        source_warnings = tuple(
            f"source 1h: {warning}"
            for warning in source_result.quality_report.warnings
        )
        quality = assess_normalized_bars(
            derived,
            metadata,
            start,
            end,
            calendar_service=self.calendar,
            warnings=source_warnings,
        )
        request = MarketDataRequest(symbol, "4h", start, end)
        manifest = self.store.save_bars(
            bars=derived,
            request=request,
            metadata=metadata,
            provider="derived",
            provider_version=None,
            data_kind=DataKind.DERIVED_RAW_WITH_ADJUSTED_CLOSE,
            derived_from=(source_result.manifest.dataset_id,),
            warnings=quality.warnings,
        )
        return DataFetchResult(
            bars=derived,
            corporate_actions=source_result.corporate_actions,
            manifest=manifest,
            corporate_actions_manifest=source_result.corporate_actions_manifest,
            quality_report=quality,
            cache_hit=False,
        )

    def load_bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        profile_name: str | TradingProfileName = TradingProfileName.BALANCED,
        cache_mode: CacheMode = CacheMode.CACHE_FIRST,
    ) -> tuple[MarketBar, ...]:
        """Load multiple configured assets and return deterministic global order."""

        bars: list[MarketBar] = []
        for symbol in symbols:
            result = self.fetch(
                profile_name=profile_name,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                cache_mode=cache_mode,
            )
            bars.extend(result.bars)
        return tuple(sorted(bars, key=lambda bar: (bar.symbol, bar.timeframe, bar.timestamp)))

    def fetch_profile_universe(
        self,
        *,
        profile_name: str | TradingProfileName,
        timeframe: str,
        start: datetime,
        end: datetime,
        cache_mode: CacheMode = CacheMode.CACHE_FIRST,
    ) -> tuple[DataFetchResult, ...]:
        profile = load_profile(profile_name)
        return tuple(
            self.fetch(
                profile_name=profile.name,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                cache_mode=cache_mode,
            )
            for symbol in profile.asset_universe
        )

    def inspect_cached(self, symbol: str, timeframe: str) -> DatasetInspection:
        manifest = self.store.find_latest(symbol, timeframe)
        if manifest is None:
            raise CacheMissError(f"no cached dataset for {symbol} {timeframe}")
        integrity = self.store.verify_integrity(manifest)
        bars = self.store.read_bars(manifest)
        quality = assess_normalized_bars(
            bars,
            self._metadata_from_manifest(manifest),
            manifest.requested_start,
            manifest.requested_end,
            calendar_service=self.calendar,
            warnings=manifest.warnings,
        )
        return DatasetInspection(manifest, quality, integrity)

    def validate_cached(self, symbol: str, timeframe: str) -> DatasetInspection:
        return self.inspect_cached(symbol, timeframe)
