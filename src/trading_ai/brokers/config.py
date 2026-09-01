"""Local-only, secret-free configuration for IBKR Paper infrastructure."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from trading_ai.brokers.exceptions import BrokerConfigurationError
from trading_ai.brokers.models import BrokerEnvironment, PaperMode
from trading_ai.core.hashing import stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IBKR_EXAMPLE = PROJECT_ROOT / "config" / "brokers" / "ibkr_paper.example.toml"
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_PLACEHOLDER_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class IBKRPaperConfig:
    host: str
    port: int
    client_id: int
    mode: PaperMode
    expected_environment: BrokerEnvironment
    allowed_account_hashes: tuple[str, ...]
    account_hash_salt_env: str
    request_timeout_seconds: float
    heartbeat_timeout_seconds: float
    max_clock_drift_seconds: float
    max_messages_per_second: int
    tif: str
    official_sdk_version: str
    contract_config: str
    paper_execution_armed: bool = False

    def __post_init__(self) -> None:
        if self.host not in _LOOPBACK:
            raise BrokerConfigurationError("IBKR Paper socket host must be loopback-only")
        if not 1 <= self.port <= 65535:
            raise BrokerConfigurationError("IBKR socket port must be in [1, 65535]")
        if not 0 <= self.client_id <= 31:
            raise BrokerConfigurationError("IBKR client_id must be in [0, 31]")
        if self.mode is PaperMode.PAPER_EXECUTION_ARMED or self.paper_execution_armed:
            raise BrokerConfigurationError("Lot 9 does not permit Paper execution arming")
        if self.expected_environment is not BrokerEnvironment.PAPER:
            raise BrokerConfigurationError("Lot 9 IBKR configuration must explicitly target PAPER")
        if self.allowed_account_hashes != tuple(sorted(set(self.allowed_account_hashes))):
            raise BrokerConfigurationError("allowed account hashes must be sorted and unique")
        for digest in self.allowed_account_hashes:
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                raise BrokerConfigurationError("account allowlist contains a non-SHA-256 value")
        if not self.account_hash_salt_env.strip():
            raise BrokerConfigurationError("account_hash_salt_env must be named explicitly")
        if (
            self.request_timeout_seconds <= 0
            or self.heartbeat_timeout_seconds <= 0
            or self.max_clock_drift_seconds <= 0
        ):
            raise BrokerConfigurationError("broker timeouts must be positive")
        if not 1 <= self.max_messages_per_second <= 50:
            raise BrokerConfigurationError("IBKR pacing must remain in [1, 50] messages/second")
        if self.tif != "DAY":
            raise BrokerConfigurationError("Lot 9 permits conservative DAY time-in-force only")
        if not self.official_sdk_version.strip() or not self.contract_config.strip():
            raise BrokerConfigurationError("SDK version and contract config are required")

    @property
    def config_hash(self) -> str:
        return stable_hash(self)

    @property
    def connectable(self) -> bool:
        return bool(self.allowed_account_hashes) and _PLACEHOLDER_HASH not in self.allowed_account_hashes


def load_ibkr_paper_config(
    path: Path | str = DEFAULT_IBKR_EXAMPLE,
    *,
    allow_example: bool = False,
) -> IBKRPaperConfig:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
        config = IBKRPaperConfig(
            host=str(raw["connection"]["host"]),
            port=int(raw["connection"]["port"]),
            client_id=int(raw["connection"]["client_id"]),
            mode=PaperMode(str(raw["session"]["mode"])),
            expected_environment=BrokerEnvironment(str(raw["account"]["environment"])),
            allowed_account_hashes=tuple(sorted(str(v) for v in raw["account"]["allowed_hashes"])),
            account_hash_salt_env=str(raw["account"]["hash_salt_env"]),
            request_timeout_seconds=float(raw["timeouts"]["request_seconds"]),
            heartbeat_timeout_seconds=float(raw["timeouts"]["heartbeat_seconds"]),
            max_clock_drift_seconds=float(raw["timeouts"]["max_clock_drift_seconds"]),
            max_messages_per_second=int(raw["pacing"]["max_messages_per_second"]),
            tif=str(raw["orders"]["time_in_force"]),
            official_sdk_version=str(raw["sdk"]["expected_version"]),
            contract_config=str(raw["contracts"]["path"]),
            paper_execution_armed=bool(raw["session"].get("paper_execution_armed", False)),
        )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise BrokerConfigurationError(f"invalid IBKR Paper config {source}: {exc}") from exc
    if not allow_example and not config.connectable:
        raise BrokerConfigurationError(
            "IBKR Paper config is an example; create an ignored local config with a salted account hash"
        )
    return config
