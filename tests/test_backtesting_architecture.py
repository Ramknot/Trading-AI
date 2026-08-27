import ast
from pathlib import Path

from trading_ai.backtesting.reproducibility import detect_git_commit
from trading_ai.core.config import PROJECT_ROOT


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


def test_backtest_engine_has_no_provider_broker_or_network_dependency() -> None:
    engine_path = PROJECT_ROOT / "src" / "trading_ai" / "backtesting" / "engine.py"
    imports = _imports(engine_path)

    assert "yfinance" not in imports
    assert not any(name.startswith("trading_ai.data.providers") for name in imports)
    assert not any(name.startswith("trading_ai.data.base") for name in imports)
    assert not any(name.startswith("trading_ai.brokers") for name in imports)
    assert not any(name in {"requests", "httpx", "urllib.request"} for name in imports)


def test_backtesting_package_never_imports_yahoo_or_broker_modules() -> None:
    package = PROJECT_ROOT / "src" / "trading_ai" / "backtesting"
    imports = set().union(*(_imports(path) for path in package.glob("*.py")))

    assert "yfinance" not in imports
    assert not any(name.startswith("trading_ai.data.providers") for name in imports)
    assert not any(name.startswith("trading_ai.brokers") for name in imports)


def test_git_commit_provenance_is_detected_when_repository_has_head() -> None:
    commit = detect_git_commit(PROJECT_ROOT)

    assert commit is not None
    assert len(commit) == 40
