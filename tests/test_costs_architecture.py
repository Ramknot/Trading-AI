from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return tuple(names)


def test_cost_and_validation_modules_have_no_broker_network_or_risk_bypass() -> None:
    forbidden = (
        "trading_ai.brokers",
        "trading_ai.data.providers",
        "trading_ai.execution",
        "trading_ai.risk",
        "yfinance",
        "requests",
        "httpx",
        "urllib.request",
        "ibapi",
    )
    roots = (Path("src/trading_ai/costs"), Path("src/trading_ai/validation"))
    for root in roots:
        for path in root.glob("*.py"):
            imported = _imports(path)
            assert not any(
                name.startswith(prefix)
                for name in imported
                for prefix in forbidden
            ), (path, imported)


def test_cost_and_validation_modules_do_not_expose_execution_mutations() -> None:
    content = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for root in (Path("src/trading_ai/costs"), Path("src/trading_ai/validation"))
        for path in root.glob("*.py")
    )
    for forbidden in (
        "place_order",
        "submit_order",
        "enable_live",
        "force_live",
        "verify=false",
        "verify = false",
    ):
        assert forbidden not in content
