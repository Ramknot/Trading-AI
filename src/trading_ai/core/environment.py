"""Explicit environment policies for Lot 0."""

from dataclasses import dataclass

from trading_ai.core.models import ExecutionEnvironment


@dataclass(frozen=True, slots=True)
class EnvironmentPolicy:
    """Capabilities allowed in one isolated runtime environment."""

    startup_allowed: bool
    external_order_transmission_allowed: bool
    description: str


ENVIRONMENT_POLICIES: dict[ExecutionEnvironment, EnvironmentPolicy] = {
    ExecutionEnvironment.DEV: EnvironmentPolicy(
        startup_allowed=True,
        external_order_transmission_allowed=False,
        description="Local development with no external order transmission.",
    ),
    ExecutionEnvironment.TEST: EnvironmentPolicy(
        startup_allowed=True,
        external_order_transmission_allowed=False,
        description="Automated tests with deterministic local doubles.",
    ),
    ExecutionEnvironment.PAPER: EnvironmentPolicy(
        startup_allowed=True,
        external_order_transmission_allowed=False,
        description="Research/paper configuration; orders remain denied in Lot 0.",
    ),
    ExecutionEnvironment.LIVE: EnvironmentPolicy(
        startup_allowed=False,
        external_order_transmission_allowed=False,
        description="Locked until a dedicated future lot is reviewed and approved.",
    ),
}


def policy_for(environment: ExecutionEnvironment) -> EnvironmentPolicy:
    """Return the immutable policy for an environment."""

    return ENVIRONMENT_POLICIES[environment]
