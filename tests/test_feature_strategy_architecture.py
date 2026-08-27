import ast
from pathlib import Path

from trading_ai.core.config import PROJECT_ROOT, load_runtime_settings
from trading_ai.features import FeatureEngine
from trading_ai.strategies import (
    BreakoutStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def test_feature_package_is_decision_provider_broker_ml_and_regime_independent() -> None:
    package = PROJECT_ROOT / "src" / "trading_ai" / "features"
    imports = set().union(*(_imports(path) for path in package.glob("*.py")))
    forbidden = (
        "trading_ai.strategies",
        "trading_ai.backtesting",
        "trading_ai.data.providers",
        "trading_ai.brokers",
        "trading_ai.execution",
        "trading_ai.ml",
        "trading_ai.regimes",
        "yfinance",
        "requests",
    )

    assert not any(
        name == blocked or name.startswith(f"{blocked}.")
        for name in imports
        for blocked in forbidden
    )


def test_baselines_use_shared_feature_engine_and_have_no_external_execution_path() -> None:
    package = PROJECT_ROOT / "src" / "trading_ai" / "strategies"
    imports = set().union(*(_imports(path) for path in package.glob("*.py")))

    assert "trading_ai.features" in imports
    assert not any(name.startswith("trading_ai.brokers") for name in imports)
    assert not any(name.startswith("trading_ai.execution") for name in imports)
    assert not any(name.startswith("trading_ai.ml") for name in imports)
    regime_imports = {
        name for name in imports if name.startswith("trading_ai.regimes")
    }
    assert regime_imports <= {"trading_ai.regimes.models"}
    assert not any(name in {"yfinance", "pandas", "sklearn"} for name in imports)
    assert all(
        strategy_class.__module__ == "trading_ai.strategies.baselines"
        for strategy_class in (
            TrendFollowingStrategy,
            MomentumStrategy,
            BreakoutStrategy,
        )
    )


def test_baseline_source_does_not_hardcode_profile_market_symbols() -> None:
    profile = load_runtime_settings("PAPER", "balanced").profile
    strategy_root = PROJECT_ROOT / "src" / "trading_ai" / "strategies"
    string_constants: set[str] = set()
    for path in strategy_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        string_constants.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )

    assert not set(profile.asset_universe).intersection(string_constants)


def test_feature_engine_public_contract_returns_snapshots_not_dataframes() -> None:
    annotation = FeatureEngine.compute.__annotations__["return"]

    assert annotation == "FeatureSnapshot"
    assert "DataFrame" not in str(FeatureEngine.compute.__annotations__)
