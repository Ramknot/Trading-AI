from __future__ import annotations

import ast
from pathlib import Path

from trading_ai.core.config import PROJECT_ROOT


REGIME_ROOT = PROJECT_ROOT / "src" / "trading_ai" / "regimes"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_regime_modules_have_no_provider_broker_network_or_ml_dependencies() -> None:
    forbidden = (
        "yfinance",
        "requests",
        "sklearn",
        "scikit_learn",
        "xgboost",
        "lightgbm",
        "tensorflow",
        "torch",
        "hmmlearn",
        "trading_ai.data.providers",
        "trading_ai.brokers",
    )
    for path in REGIME_ROOT.glob("*.py"):
        imports = _imports(path)
        assert not any(
            name == item or name.startswith(f"{item}.")
            for name in imports
            for item in forbidden
        ), path


def test_regime_production_code_has_no_market_symbol_constants() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in REGIME_ROOT.glob("*.py")
    )
    for symbol in ("AAPL", "SPY", "MC.PA", "BTC-USD"):
        assert symbol not in source


def test_regime_and_feature_production_code_has_no_future_shift_or_centered_window() -> None:
    paths = (
        *REGIME_ROOT.glob("*.py"),
        *(PROJECT_ROOT / "src" / "trading_ai" / "features").glob("*.py"),
    )
    forbidden = ("shift(-1", "center=True", "next_close", "next_return")
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token} in {path}"


def test_mean_reversion_contains_no_short_martingale_or_averaging_down_path() -> None:
    source = (
        PROJECT_ROOT / "src" / "trading_ai" / "strategies" / "baselines.py"
    ).read_text(encoding="utf-8").lower()
    for token in ("martingale", "average_down", "double_after_loss", "orderSide.SELL_SHORT"):
        assert token.lower() not in source


def test_cli_exposes_no_production_regime_bypass_flags() -> None:
    source = (PROJECT_ROOT / "src" / "trading_ai" / "cli.py").read_text(
        encoding="utf-8"
    )
    for flag in ("--disable-regime", "--ignore-regime", "--force-strategy"):
        assert flag not in source
