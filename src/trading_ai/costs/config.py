"""Configuration-driven, dated, source-provenanced transaction economics."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trading_ai.core.hashing import stable_hash
from trading_ai.core.config import PROJECT_ROOT
from trading_ai.core.models import OrderSide, TradingProfile, TradingProfileName
from trading_ai.costs.exceptions import CostConfigurationError
from trading_ai.costs.models import (
    CommissionTier,
    CostStatus,
    InstrumentCostMetadata,
    TariffProfile,
    TariffStatus,
    TransactionTaxRule,
)


DEFAULT_COST_DIRECTORY = PROJECT_ROOT / "config" / "costs"
ZERO = Decimal("0")


def _decimal(raw: object, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise CostConfigurationError(f"{name} must be numeric") from exc
    if not value.is_finite():
        raise CostConfigurationError(f"{name} must be finite")
    return value


def _bool(raw: object, name: str) -> bool:
    if type(raw) is not bool:
        raise CostConfigurationError(f"{name} must be a TOML boolean")
    return raw


def _datetime(raw: object, name: str) -> datetime:
    if not isinstance(raw, datetime) or raw.tzinfo is None or raw.utcoffset() is None:
        raise CostConfigurationError(f"{name} must be a timezone-aware TOML datetime")
    return raw


@dataclass(frozen=True, slots=True)
class OperatingComponentConfig:
    status: CostStatus
    amount: Decimal | None
    source_reference: str

    def __post_init__(self) -> None:
        if not self.source_reference.strip():
            raise CostConfigurationError("operating cost source must not be empty")
        if self.status is CostStatus.UNAVAILABLE:
            if self.amount is not None:
                raise CostConfigurationError("UNAVAILABLE operating cost cannot have amount")
        elif self.status is CostStatus.NOT_APPLICABLE:
            if self.amount != ZERO:
                raise CostConfigurationError("NOT_APPLICABLE operating cost must be zero")
        elif self.amount is None or self.amount < ZERO:
            raise CostConfigurationError("known/estimated operating cost requires amount")


@dataclass(frozen=True, slots=True)
class OperatingCostConfig:
    currency: str
    market_data_subscription: OperatingComponentConfig
    server_vps: OperatingComponentConfig
    software_subscription: OperatingComponentConfig
    other_fixed_cost: OperatingComponentConfig


@dataclass(frozen=True, slots=True)
class BalancedCostConfig:
    name: TradingProfileName
    enabled: bool
    engine_name: str
    engine_version: str
    base_currency: str
    tariff_profile: str
    tax_profile: str
    instrument_profile: str
    operating_profile: str
    cash_buffer_bps: Decimal
    cash_buffer_absolute: Decimal
    fx_cost_bps: Decimal
    minimum_net_edge_bps: Decimal
    minimum_edge_to_cost_ratio: Decimal
    missing_edge_policy: str
    allow_retrospective_tariff: bool
    require_verified_tariff_for_validation: bool
    critical_variable_components: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "engine_name", "engine_version", "base_currency", "tariff_profile",
            "tax_profile", "instrument_profile", "operating_profile", "missing_edge_policy",
        ):
            if not getattr(self, field_name).strip():
                raise CostConfigurationError(f"{field_name} must not be empty")
        for field_name in (
            "cash_buffer_bps", "cash_buffer_absolute", "fx_cost_bps",
            "minimum_net_edge_bps", "minimum_edge_to_cost_ratio",
        ):
            value = getattr(self, field_name)
            if value < ZERO or not value.is_finite():
                raise CostConfigurationError(f"{field_name} must be finite and non-negative")
        if self.missing_edge_policy not in {"INCOMPLETE_ALLOW_RESEARCH", "BLOCK"}:
            raise CostConfigurationError("unsupported missing_edge_policy")
        if tuple(sorted(set(self.critical_variable_components))) != self.critical_variable_components:
            raise CostConfigurationError("critical components must be sorted and unique")

    def to_parameters(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({
            "name": self.name.value,
            "enabled": str(self.enabled).lower(),
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "base_currency": self.base_currency,
            "tariff_profile": self.tariff_profile,
            "tax_profile": self.tax_profile,
            "instrument_profile": self.instrument_profile,
            "operating_profile": self.operating_profile,
            "cash_buffer_bps": str(self.cash_buffer_bps),
            "cash_buffer_absolute": str(self.cash_buffer_absolute),
            "fx_cost_bps": str(self.fx_cost_bps),
            "minimum_net_edge_bps": str(self.minimum_net_edge_bps),
            "minimum_edge_to_cost_ratio": str(self.minimum_edge_to_cost_ratio),
            "missing_edge_policy": self.missing_edge_policy,
            "allow_retrospective_tariff": str(self.allow_retrospective_tariff).lower(),
            "require_verified_tariff_for_validation": str(self.require_verified_tariff_for_validation).lower(),
            "critical_variable_components": ",".join(self.critical_variable_components),
        }.items()))


@dataclass(frozen=True, slots=True)
class CostConfigurationBundle:
    config: BalancedCostConfig
    tariff: TariffProfile
    taxes: tuple[TransactionTaxRule, ...]
    instruments: tuple[InstrumentCostMetadata, ...]
    operating: OperatingCostConfig
    config_hash: str

    def instrument_for(self, symbol: str) -> InstrumentCostMetadata | None:
        return next((item for item in self.instruments if item.symbol == symbol), None)

    def tax_rule(self, rule_id: str) -> TransactionTaxRule | None:
        return next((item for item in self.taxes if item.rule_id == rule_id), None)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CostConfigurationError(f"cost configuration not found: {path}")
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except tomllib.TOMLDecodeError as exc:
        raise CostConfigurationError(f"invalid TOML {path}: {exc}") from exc


def inspect_cost_config(
    profile_name: str | TradingProfileName,
    *,
    cost_directory: Path = DEFAULT_COST_DIRECTORY,
    tariff_profile: str | None = None,
) -> BalancedCostConfig:
    try:
        name = profile_name if isinstance(profile_name, TradingProfileName) else TradingProfileName(str(profile_name).lower())
        raw = _load(cost_directory / f"{name.value}.toml")
        config = BalancedCostConfig(
            name=TradingProfileName(str(raw["name"])),
            enabled=_bool(raw["enabled"], "enabled"),
            engine_name=str(raw["engine_name"]),
            engine_version=str(raw["engine_version"]),
            base_currency=str(raw["base_currency"]).upper(),
            tariff_profile=tariff_profile or str(raw["tariff_profile"]),
            tax_profile=str(raw["tax_profile"]),
            instrument_profile=str(raw["instrument_profile"]),
            operating_profile=str(raw["operating_profile"]),
            cash_buffer_bps=_decimal(raw["cash_buffer_bps"], "cash_buffer_bps"),
            cash_buffer_absolute=_decimal(raw["cash_buffer_absolute"], "cash_buffer_absolute"),
            fx_cost_bps=_decimal(raw["fx_cost_bps"], "fx_cost_bps"),
            minimum_net_edge_bps=_decimal(raw["minimum_net_edge_bps"], "minimum_net_edge_bps"),
            minimum_edge_to_cost_ratio=_decimal(raw["minimum_edge_to_cost_ratio"], "minimum_edge_to_cost_ratio"),
            missing_edge_policy=str(raw["missing_edge_policy"]),
            allow_retrospective_tariff=_bool(raw["allow_retrospective_tariff"], "allow_retrospective_tariff"),
            require_verified_tariff_for_validation=_bool(raw["require_verified_tariff_for_validation"], "require_verified_tariff_for_validation"),
            critical_variable_components=tuple(sorted(str(item) for item in raw["critical_variable_components"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CostConfigurationError):
            raise
        raise CostConfigurationError(f"invalid cost profile: {exc}") from exc
    if config.name is not name:
        raise CostConfigurationError("cost profile identity mismatch")
    return config


def load_tariff_profile(profile_id: str, directory: Path) -> TariffProfile:
    raw = _load(directory / "brokers" / f"{profile_id}.toml")
    commission = raw.get("commission", {})
    exchange = raw.get("exchange_fees", {})
    normalized_hash = stable_hash(raw)
    try:
        tiers = tuple(
            CommissionTier(
                _decimal(item["up_to_monthly_quantity"], "tier boundary")
                if "up_to_monthly_quantity" in item else None,
                _decimal(item["per_unit"], "tier per_unit"),
            )
            for item in commission.get("tiers", [])
        )
        tariff = TariffProfile(
            profile_id=str(raw["profile_id"]),
            provider=str(raw["provider"]),
            plan=str(raw["plan"]),
            currency=str(raw["currency"]).upper(),
            markets=tuple(sorted(str(item) for item in raw["markets"])),
            effective_from=_datetime(raw["effective_from"], "effective_from"),
            effective_to=_datetime(raw["effective_to"], "effective_to") if "effective_to" in raw else None,
            source_name=str(raw["source_name"]),
            source_reference=str(raw["source_reference"]),
            verified_at=_datetime(raw["verified_at"], "verified_at"),
            status=TariffStatus(str(raw["status"])),
            version=str(raw["version"]),
            # Every potentially chargeable field is explicit.  Missing tariff
            # data is a configuration error, never an implicit zero-cost rule.
            fixed_per_order=_decimal(commission["fixed_per_order"], "fixed_per_order"),
            per_unit=_decimal(commission["per_unit"], "per_unit"),
            proportional_bps=_decimal(commission["proportional_bps"], "proportional_bps"),
            minimum_per_order=_decimal(commission["minimum_per_order"], "minimum_per_order"),
            maximum_per_order=_decimal(commission["maximum_per_order"], "maximum_per_order") if "maximum_per_order" in commission else None,
            maximum_notional_fraction=_decimal(commission["maximum_notional_fraction"], "maximum_notional_fraction") if "maximum_notional_fraction" in commission else None,
            tiers=tiers,
            exchange_fee_status=CostStatus(str(exchange["status"])),
            exchange_fee_per_unit=_decimal(exchange["per_unit"], "exchange fee per_unit"),
            exchange_fee_bps=_decimal(exchange["proportional_bps"], "exchange fee bps"),
            exchange_fees_included_in_commission=_bool(exchange["included_in_commission"], "included_in_commission"),
            config_hash=normalized_hash,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CostConfigurationError(f"invalid tariff profile {profile_id}: {exc}") from exc
    if tariff.profile_id != profile_id:
        raise CostConfigurationError("tariff profile ID mismatch")
    return tariff


def load_tax_rules(profile: str, directory: Path) -> tuple[TransactionTaxRule, ...]:
    raw = _load(directory / "taxes" / f"{profile}.toml")
    rules: list[TransactionTaxRule] = []
    for rule_id, values in sorted(raw.get("rules", {}).items()):
        rules.append(TransactionTaxRule(
            rule_id=str(rule_id),
            name=str(values["name"]),
            currency=str(values["currency"]).upper(),
            rate_bps=_decimal(values["rate_bps"], "tax rate_bps"),
            applicable_side=OrderSide(str(values["applicable_side"])),
            effective_from=_datetime(values["effective_from"], "tax effective_from"),
            effective_to=_datetime(values["effective_to"], "tax effective_to") if "effective_to" in values else None,
            source_name=str(values["source_name"]),
            source_reference=str(values["source_reference"]),
            verified_at=_datetime(values["verified_at"], "tax verified_at"),
            status=TariffStatus(str(values["status"])),
            config_hash=stable_hash(values),
        ))
    return tuple(rules)


def load_instruments(profile: str, directory: Path) -> tuple[InstrumentCostMetadata, ...]:
    raw = _load(directory / "instruments" / f"{profile}.toml")
    result = []
    for symbol, values in sorted(raw.get("instruments", {}).items()):
        result.append(InstrumentCostMetadata(
            symbol=str(symbol),
            market=str(values["market"]),
            venue=str(values["venue"]),
            currency=str(values["currency"]).upper(),
            transaction_tax_applicable=values.get("transaction_tax_applicable"),
            transaction_tax_rule_id=str(values["transaction_tax_rule_id"]) if "transaction_tax_rule_id" in values else None,
            metadata_status=TariffStatus(str(values["metadata_status"])),
            source_reference=str(values["source_reference"]),
        ))
    return tuple(result)


def load_operating_config(profile: str, directory: Path) -> OperatingCostConfig:
    raw = _load(directory / "operating" / f"{profile}.toml")
    def component(name: str) -> OperatingComponentConfig:
        values = raw[name]
        return OperatingComponentConfig(
            status=CostStatus(str(values["status"])),
            amount=_decimal(values["amount"], f"{name} amount") if "amount" in values else None,
            source_reference=str(values["source_reference"]),
        )
    return OperatingCostConfig(
        currency=str(raw["currency"]).upper(),
        market_data_subscription=component("market_data_subscription"),
        server_vps=component("server_vps"),
        software_subscription=component("software_subscription"),
        other_fixed_cost=component("other_fixed_cost"),
    )


def load_balanced_cost_config(
    profile: TradingProfile,
    *,
    cost_directory: Path = DEFAULT_COST_DIRECTORY,
    tariff_profile: str | None = None,
) -> CostConfigurationBundle:
    config = inspect_cost_config(profile.name, cost_directory=cost_directory, tariff_profile=tariff_profile)
    if profile.name is not TradingProfileName.BALANCED:
        raise CostConfigurationError("aggressive transaction economics remains locked")
    if not profile.enabled or not config.enabled:
        raise CostConfigurationError("Balanced profile and cost config must be enabled")
    tariff = load_tariff_profile(config.tariff_profile, cost_directory)
    taxes = load_tax_rules(config.tax_profile, cost_directory)
    instruments = load_instruments(config.instrument_profile, cost_directory)
    operating = load_operating_config(config.operating_profile, cost_directory)
    configured = {item.symbol for item in instruments}
    missing = sorted(set(profile.asset_universe) - configured)
    if missing:
        raise CostConfigurationError("profile instruments missing cost metadata: " + ", ".join(missing))
    payload = {
        "config": config.to_parameters(),
        "tariff": tariff,
        "taxes": taxes,
        "instruments": instruments,
        "operating": operating,
    }
    return CostConfigurationBundle(
        config=config,
        tariff=tariff,
        taxes=taxes,
        instruments=instruments,
        operating=operating,
        config_hash=stable_hash(payload),
    )
