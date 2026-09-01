"""Configuration-driven, exact-match IBKR contract resolution and cache."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_ai.brokers.exceptions import BrokerIntegrityError, ContractResolutionError
from trading_ai.core.hashing import stable_hash, to_primitive


@dataclass(frozen=True, slots=True)
class IBKRContractSpec:
    symbol: str
    broker_symbol: str
    sec_type: str
    exchange: str
    primary_exchange: str
    currency: str
    con_id: int | None = None
    local_symbol: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "symbol", "broker_symbol", "sec_type", "exchange", "primary_exchange", "currency"
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} must not be empty")
        if self.con_id is not None and self.con_id <= 0:
            raise ValueError("con_id must be positive when configured")

    @property
    def contract_hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True, slots=True)
class IBKRContractCandidate:
    con_id: int
    symbol: str
    sec_type: str
    exchange: str
    primary_exchange: str
    currency: str
    local_symbol: str


def load_contract_specs(path: Path | str) -> tuple[IBKRContractSpec, ...]:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
        specs = []
        for symbol, item in raw["contracts"].items():
            specs.append(
                IBKRContractSpec(
                    symbol=str(symbol),
                    broker_symbol=str(item.get("broker_symbol", symbol)),
                    sec_type=str(item["sec_type"]),
                    exchange=str(item["exchange"]),
                    primary_exchange=str(item["primary_exchange"]),
                    currency=str(item["currency"]),
                    con_id=(int(item["con_id"]) if "con_id" in item else None),
                    local_symbol=(str(item["local_symbol"]) if "local_symbol" in item else None),
                )
            )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ContractResolutionError(f"invalid IBKR contract config {source}: {exc}") from exc
    symbols = [item.symbol for item in specs]
    if len(symbols) != len(set(symbols)):
        raise ContractResolutionError("IBKR contract config contains duplicate symbols")
    return tuple(sorted(specs, key=lambda item: item.symbol))


class IBKRContractResolver:
    def __init__(
        self,
        specs: tuple[IBKRContractSpec, ...],
        *,
        cache_path: Path | str | None = None,
    ) -> None:
        self._specs = {item.symbol: item for item in specs}
        self._resolved: dict[str, IBKRContractCandidate] = {}
        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None and self.cache_path.is_file():
            self._load_cache()

    def configured(self, symbol: str) -> IBKRContractSpec:
        try:
            return self._specs[symbol]
        except KeyError as exc:
            raise ContractResolutionError(f"symbol {symbol!r} is absent from IBKR contract config") from exc

    def configured_symbol_for(self, broker_symbol: str, currency: str) -> str:
        matches = [
            spec.symbol
            for spec in self._specs.values()
            if spec.broker_symbol == broker_symbol and spec.currency == currency
        ]
        if len(matches) != 1:
            raise ContractResolutionError(
                "broker instrument cannot be mapped to exactly one configured symbol"
            )
        return matches[0]

    def resolve(
        self, symbol: str, candidates: tuple[IBKRContractCandidate, ...] = ()
    ) -> IBKRContractCandidate:
        spec = self.configured(symbol)
        cached = self._resolved.get(symbol)
        if cached is not None:
            self._assert_match(spec, cached)
            return cached
        exact = tuple(item for item in candidates if self._matches(spec, item))
        if spec.con_id is not None:
            exact = tuple(item for item in exact if item.con_id == spec.con_id)
        if len(exact) != 1:
            raise ContractResolutionError(
                f"IBKR contract {symbol!r} requires exactly one verified match; found {len(exact)}"
            )
        self._resolved[symbol] = exact[0]
        self._save_cache()
        return exact[0]

    @staticmethod
    def _matches(spec: IBKRContractSpec, item: IBKRContractCandidate) -> bool:
        return (
            item.symbol == spec.broker_symbol
            and item.sec_type == spec.sec_type
            and item.exchange == spec.exchange
            and item.primary_exchange == spec.primary_exchange
            and item.currency == spec.currency
            and (spec.local_symbol is None or item.local_symbol == spec.local_symbol)
        )

    def _assert_match(self, spec: IBKRContractSpec, item: IBKRContractCandidate) -> None:
        if not self._matches(spec, item) or (
            spec.con_id is not None and item.con_id != spec.con_id
        ):
            raise ContractResolutionError("cached IBKR contract no longer matches configuration")

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        payload = {
            "version": "1.0",
            "config_hash": stable_hash(tuple(self._specs.values())),
            "contracts": [to_primitive(item) for item in sorted(self._resolved.values(), key=lambda x: x.symbol)],
        }
        payload["content_hash"] = stable_hash(payload)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.cache_path)

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        try:
            payload: dict[str, Any] = json.loads(self.cache_path.read_text(encoding="utf-8"))
            expected = payload.pop("content_hash")
            if expected != stable_hash(payload):
                raise BrokerIntegrityError("IBKR contract cache checksum mismatch")
            if payload["config_hash"] != stable_hash(tuple(self._specs.values())):
                raise BrokerIntegrityError("IBKR contract cache belongs to another config")
            for raw in payload["contracts"]:
                item = IBKRContractCandidate(**raw)
                spec_symbol = next(
                    (symbol for symbol, spec in self._specs.items() if spec.broker_symbol == item.symbol),
                    None,
                )
                if spec_symbol is None:
                    raise BrokerIntegrityError("IBKR contract cache contains an unknown symbol")
                self._assert_match(self._specs[spec_symbol], item)
                self._resolved[spec_symbol] = item
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, BrokerIntegrityError):
                raise
            raise BrokerIntegrityError(f"invalid IBKR contract cache: {exc}") from exc
