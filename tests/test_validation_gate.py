from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from trading_ai.costs.models import (
    CostComponent,
    CostCoverage,
    CostStatus,
    TariffStatus,
)
from trading_ai.validation.config import load_validation_config
from trading_ai.validation.exceptions import ValidationError
from trading_ai.validation.gate import ResearchValidationGate
from trading_ai.validation.models import CriterionStatus, ValidationStatus
from trading_ai.validation.storage import LocalValidationStore


START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _result(*, operating_complete: bool = True, trade_count: int = 30):
    trades = tuple(
        SimpleNamespace(
            symbol="AAPL" if index % 2 == 0 else "MSFT",
            exit_time=START + timedelta(days=index + 2),
            net_pnl=Decimal("2"),
        )
        for index in range(trade_count)
    )
    operating = CostComponent.known(
        "total_operating_cost", Decimal("5"), "USD", "fixture"
    ) if operating_complete else CostComponent.unavailable(
        "total_operating_cost", "USD", "fixture", "not supplied"
    )
    return SimpleNamespace(
        run_id="bt-validation-fixture",
        cost_summary=SimpleNamespace(
            cost_coverage=CostCoverage.COMPLETE,
            net_return_before_operating=0.10,
            total_variable_cost=Decimal("10"),
            tariff_profile_id="fixture-tariff",
            tariff_status=TariffStatus.VERIFIED,
        ),
        operating_costs=SimpleNamespace(total_operating_cost=operating),
        metrics=SimpleNamespace(
            number_of_trades=trade_count,
            expectancy=Decimal("2"),
            profit_factor=2.0,
            max_drawdown_pct=-0.05,
        ),
        initial_cash=Decimal("1000"),
        final_equity=Decimal("1100"),
        equity_curve=tuple(
            SimpleNamespace(
                timestamp=START + timedelta(days=index * 15),
                cash=Decimal("500"),
                equity=Decimal("1000") + Decimal(index * 25),
            )
            for index in range(5)
        ),
        trades=trades,
        risk_state_transitions=(),
        dataset_references=(
            SimpleNamespace(dataset_id="dataset-1", checksum_sha256="a" * 64),
        ),
    )


def _evaluate(gate, result, **overrides):
    values = {
        "datasets_integrity": True,
        "data_quality_acceptable": True,
        "final_oos": True,
        "training_or_edge_overlap": False,
        "tariff_period_verified": True,
        "real_data_available": True,
        "synthetic_mechanics_only": False,
    }
    values.update(overrides)
    return gate.evaluate(result, **values)


def test_validation_config_hash_is_deterministic_and_defaults_are_predeclared() -> None:
    first, first_hash = load_validation_config()
    second, second_hash = load_validation_config()
    assert first == second
    assert first_hash == second_hash and len(first_hash) == 64
    assert first.minimum_closed_trades == 30
    assert first.cost_stress_multipliers == (
        Decimal("1.00"), Decimal("1.25"), Decimal("1.50"), Decimal("2.00")
    )


def test_validation_config_rejects_non_boolean_safety_flags(tmp_path) -> None:
    source = """
name = "balanced-research-validation"
version = "1.0"
enabled = "true"
minimum_closed_trades = 30
minimum_net_return = 0.0
minimum_net_expectancy = 0.0
minimum_profit_factor = 1.0
maximum_drawdown = 0.10
minimum_subperiods = 3
symbol_concentration_warning_fraction = 0.60
cost_stress_multipliers = [1.0, 1.25]
require_final_oos = true
require_complete_variable_costs = true
require_verified_tariff_for_period = true
require_operating_costs_for_pass = true
"""
    path = tmp_path / "invalid-validation.toml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValidationError, match="enabled must be a TOML boolean"):
        load_validation_config(path)


def test_validation_gate_can_pass_research_without_unlocking_execution() -> None:
    report = _evaluate(ResearchValidationGate(), _result())
    assert report.status is ValidationStatus.PASS
    assert report.real_data_campaign_status is ValidationStatus.PASS
    assert report.unlocks_paper_or_live is False
    assert report.survivorship_bias_warning == "SURVIVORSHIP_BIAS_NOT_RESOLVED"
    assert len(report.stress_results) == 4
    assert len(report.subperiods) == 3
    assert report.symbols


def test_operating_costs_missing_produce_warning_not_complete_deployment_net() -> None:
    report = _evaluate(ResearchValidationGate(), _result(operating_complete=False))
    assert report.status is ValidationStatus.WARNING
    assert "OPERATING_COSTS_INCOMPLETE" in report.warnings
    operating = next(item for item in report.criteria if item.name == "operating_costs")
    assert operating.status is CriterionStatus.WARNING


@pytest.mark.parametrize(
    "overrides,criterion",
    (
        ({"data_quality_acceptable": False}, "data_quality"),
        ({"datasets_integrity": False}, "dataset_integrity"),
        ({"final_oos": False}, "final_oos"),
        ({"training_or_edge_overlap": True}, "final_oos"),
        ({"tariff_period_verified": False}, "tariff_period"),
    ),
)
def test_validation_gate_fails_closed_on_critical_research_inputs(overrides, criterion) -> None:
    report = _evaluate(ResearchValidationGate(), _result(), **overrides)
    assert report.status is ValidationStatus.FAIL
    assert next(item for item in report.criteria if item.name == criterion).status is CriterionStatus.FAIL


def test_insufficient_trades_and_negative_net_metrics_fail() -> None:
    result = _result(trade_count=5)
    result.cost_summary.net_return_before_operating = -0.01
    result.metrics.expectancy = Decimal("-1")
    result.metrics.profit_factor = 0.8
    report = _evaluate(ResearchValidationGate(), result)
    assert report.status is ValidationStatus.FAIL
    failures = {item.name for item in report.criteria if item.status is CriterionStatus.FAIL}
    assert {"closed_trades", "net_return", "net_expectancy", "profit_factor"} <= failures


def test_missing_drawdown_or_equity_never_masquerades_as_safe_zero() -> None:
    result = _result()
    result.metrics.max_drawdown_pct = None
    result.equity_curve = ()

    report = _evaluate(ResearchValidationGate(), result)

    failures = {item.name for item in report.criteria if item.status is CriterionStatus.FAIL}
    assert {"max_drawdown", "cash_non_negative"} <= failures


def test_missing_real_data_is_blocked_external_data_never_a_fabricated_pass() -> None:
    report = _evaluate(
        ResearchValidationGate(),
        _result(),
        real_data_available=False,
        synthetic_mechanics_only=True,
    )
    assert report.status is ValidationStatus.BLOCKED_EXTERNAL_DATA
    assert report.real_data_campaign_status is ValidationStatus.BLOCKED_EXTERNAL_DATA
    assert report.synthetic_mechanics_only is True


def test_validation_report_store_is_tamper_evident_and_rejects_path_traversal(tmp_path) -> None:
    report = _evaluate(ResearchValidationGate(), _result())
    store = LocalValidationStore(tmp_path / "validation")
    directory = store.save(report)
    assert store.inspect(report.validation_id)["status"] == "PASS"
    assert store.latest_for_run(report.run_id)["validation_id"] == report.validation_id
    (directory / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="checksum"):
        store.inspect(report.validation_id)
    with pytest.raises(ValidationError, match="invalid validation_id"):
        store.inspect("../escape")


def test_validation_report_store_is_idempotent_and_keeps_first_artifact_immutable(tmp_path) -> None:
    gate = ResearchValidationGate()
    store = LocalValidationStore(tmp_path / "validation")
    first = _evaluate(gate, _result())
    directory = store.save(first)
    original_report = (directory / "report.json").read_bytes()
    original_checksums = (directory / "checksums.json").read_bytes()

    # The deterministic validation ID excludes the technical creation time.
    # Repeating the same evaluation must reuse, not rewrite, the first artifact.
    second = _evaluate(gate, _result())
    assert second.validation_id == first.validation_id
    assert store.save(second) == directory
    assert (directory / "report.json").read_bytes() == original_report
    assert (directory / "checksums.json").read_bytes() == original_checksums
