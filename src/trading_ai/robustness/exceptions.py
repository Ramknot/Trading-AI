"""Domain failures for frozen research and robustness diagnostics."""


class RobustnessError(Exception):
    """Base error surfaced by the offline robustness layer."""


class BaselineMismatchError(RobustnessError):
    """The checksum-verified V1 result no longer matches its frozen manifest."""


class HoldoutGovernanceError(RobustnessError):
    """A holdout access or lifecycle transition violates the frozen plan."""


class RobustnessStorageError(RobustnessError):
    """A local robustness artifact is missing, corrupt, or unsafe."""
