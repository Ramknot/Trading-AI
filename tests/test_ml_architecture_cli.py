from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ml_support import ConstantAdapter, model_artifact
from trading_ai.backtesting.models import StrategySignal, StrategySignalAction
from trading_ai.cli import build_parser, main
from trading_ai.features import FeatureSnapshot
from trading_ai.ml.base import ModelRegistry
from trading_ai.ml.inference import InferenceEngine, SignalMLScorer
from trading_ai.ml.inputs import ModelInput
from trading_ai.ml.models import InputKind, MLFilterStatus, MLMode
from trading_ai.regimes.models import RegimeSnapshot, StructureRegime, VolatilityRegime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_sklearn_estimators_are_confined_to_adapter_and_not_business_modules() -> None:
    source_root = PROJECT_ROOT / "src" / "trading_ai"
    forbidden_names = (
        "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"
    )
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.as_posix().endswith("ml/adapters/sklearn.py"):
            assert all(name in text for name in forbidden_names)
            continue
        assert not any(name in text for name in forbidden_names), path
    for package in ("risk", "strategies", "regimes", "backtesting"):
        imports = set().union(
            *(_imports(path) for path in (source_root / package).glob("*.py"))
        )
        assert not any(name.startswith("sklearn") for name in imports)


def test_ml_modules_have_no_broker_network_or_neural_framework_imports() -> None:
    imports = set().union(
        *(_imports(path) for path in (PROJECT_ROOT / "src" / "trading_ai" / "ml").rglob("*.py"))
    )
    forbidden_prefixes = (
        "trading_ai.brokers", "requests", "yfinance", "tensorflow", "torch",
        "jax", "xgboost", "lightgbm", "kafka", "redis",
    )
    assert not any(
        name.startswith(prefix) for name in imports for prefix in forbidden_prefixes
    )


def test_inference_engine_exposes_no_training_or_online_update_operation() -> None:
    public_names = {
        name for name, _ in inspect.getmembers(InferenceEngine) if not name.startswith("_")
    }
    assert public_names.isdisjoint({"fit", "train", "partial_fit", "online_update"})
    source = inspect.getsource(InferenceEngine)
    assert ".fit(" not in source
    assert "partial_fit" not in source


@dataclass(frozen=True)
class SequenceTestInput:
    symbol: str = "AAPL"
    timestamp: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timeframe: str = "1d"
    strategy_name: str = "trend"
    strategy_version: str = "1.0"
    feature_schema_version: str = "1.1"
    ml_feature_schema_version: str = "1.0"
    input_kind: InputKind = InputKind.SEQUENCE
    feature_names: tuple[str, ...] = ("sequence-window",)
    input_hash: str = "f" * 64


class SequenceTestRegistry(ModelRegistry):
    def __init__(self, artifact, adapter):
        self.artifact = artifact
        self.adapter = adapter

    def save(self, outcome):
        return outcome

    def load(self, model_id, **compatibility):
        del compatibility
        if model_id != self.artifact.model_id:
            raise KeyError(model_id)
        return self.artifact, self.adapter

    def list(self):
        return (self.artifact,)

    def inspect(self, model_id):
        return {"model_id": model_id, "input_kind": "SEQUENCE"}


def test_future_sequence_adapter_works_through_generic_registry_and_inference_contract() -> None:
    adapter = ConstantAdapter(
        0.7, ("sequence-window",), input_kind=InputKind.SEQUENCE
    )
    artifact = model_artifact(
        adapter,
        strategy_name="trend",
        model_id="future-sequence-test",
    )
    registry = SequenceTestRegistry(artifact, adapter)
    loaded_artifact, loaded_adapter = registry.load("future-sequence-test")
    prediction = InferenceEngine(loaded_artifact, loaded_adapter).score_one(
        SequenceTestInput()
    )
    assert prediction.probability_positive == 0.7
    assert loaded_artifact.input_kind is InputKind.SEQUENCE


def test_exit_signal_is_not_filtered_even_with_zero_probability_model() -> None:
    adapter = ConstantAdapter(0.0, ("unused",))
    artifact = model_artifact(adapter)
    scorer = SignalMLScorer(
        mode=MLMode.FILTER,
        inference_engine=InferenceEngine(artifact, adapter),
    )
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signal = StrategySignal(
        signal_id="exit-1",
        strategy_name="breakout",
        strategy_version="1.0",
        symbol="AAPL",
        timeframe="1d",
        timestamp=timestamp,
        action=StrategySignalAction.EXIT_LONG,
        strength=1.0,
        reason="reduce risk",
        features_used=(),
    )
    features = FeatureSnapshot(
        symbol="AAPL", timestamp=timestamp, timeframe="1d", values=()
    )
    regime = RegimeSnapshot(
        snapshot_id="regime-exit",
        symbol="AAPL",
        timestamp=timestamp,
        timeframe="1d",
        structure_regime=StructureRegime.UNKNOWN,
        volatility_regime=VolatilityRegime.HIGH,
        detector_name="test",
        detector_version="1",
        config_hash="a" * 64,
        bars_in_current_structure_regime=1,
        evidence=(),
        reason_codes=("TEST",),
    )
    prediction, decision = scorer.evaluate(
        signal=signal, features=features, regime=regime
    )
    assert prediction is None
    assert decision.status is MLFilterStatus.NOT_APPLICABLE


def test_cli_exposes_offline_ml_lifecycle_and_backtest_modes(tmp_path, capsys) -> None:
    parser = build_parser()
    train = parser.parse_args(
        [
            "ml", "train", "--strategy", "trend", "--timeframe", "1d",
            "--model", "logistic", "--train-start", "2020-01-01",
            "--train-end", "2021-01-01", "--validation-start", "2021-01-01",
            "--validation-end", "2022-01-01", "--test-start", "2022-01-01",
            "--test-end", "2023-01-01",
        ]
    )
    assert train.ml_command == "train"
    assert train.model == "logistic"
    backtest = parser.parse_args(
        [
            "backtest", "run", "--strategy", "trend", "--symbol", "AAPL",
            "--timeframe", "1d", "--start", "2024-01-01", "--end", "2025-01-01",
            "--ml-mode", "filter", "--ml-model-id", "ml-explicit",
        ]
    )
    assert backtest.ml_mode == "filter"
    assert backtest.ml_model_id == "ml-explicit"

    assert main(
        ["ml", "model", "list", "--data-root", str(tmp_path), "--json"]
    ) == 0
    assert capsys.readouterr().out.strip() == "[]"


def test_cli_never_selects_a_latest_model_silently() -> None:
    source = (PROJECT_ROOT / "src" / "trading_ai" / "cli.py").read_text(
        encoding="utf-8"
    )
    assert "latest.pkl" not in source
    assert "active ML mode requires explicit --ml-model-id" in source
