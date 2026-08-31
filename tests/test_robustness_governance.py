from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_ai.core.hashing import stable_hash
from trading_ai.robustness.config import (
    load_research_baseline_manifest,
    load_research_plan,
    load_robustness_config,
)
from trading_ai.robustness.governance import (
    HoldoutAccessPolicy,
    HoldoutConsumer,
    consume_holdout,
    decision_core_hash,
    make_untouched_holdout,
)
from trading_ai.robustness.exceptions import (
    HoldoutGovernanceError,
    RobustnessStorageError,
)
from trading_ai.robustness.models import (
    HoldoutStatus,
    PeriodClassification,
    PointInTimeUniverse,
    SurvivorshipStatus,
    UniverseMembership,
)
from trading_ai.robustness.service import RobustnessService
from trading_ai.cli import main


def test_frozen_baseline_preserves_consumed_v1_fail_without_relaxed_thresholds() -> None:
    baseline = load_research_baseline_manifest()
    plan = load_research_plan(baseline=baseline)

    assert baseline.period.label == "CONSUMED_DIAGNOSTIC_OOS_V1"
    assert baseline.validation_status == "FAIL"
    assert baseline.closed_trades == 16
    assert baseline.max_drawdown > 0.10
    assert baseline.top_contributor == "AAPL"
    assert baseline.top_contributor_share > 0.79
    criteria = dict(plan.frozen_validation_criteria)
    assert criteria["minimum_closed_trades"] == "30"
    assert criteria["maximum_drawdown"] == "0.10"
    assert plan.plan_hash == "8d8e7c2940762a8ea6cadae94a2f615750d5c25e31652d5c8d1d1501bfe4ca74"
    assert plan.plan_hash == load_research_plan(baseline=baseline).plan_hash
    assert plan.frozen_at == load_research_plan(baseline=baseline).frozen_at
    assert plan.frozen_at < datetime(2026, 8, 30, 11, 27, 26, tzinfo=timezone.utc)


def test_plan_hash_changes_when_a_failed_threshold_is_relaxed(tmp_path: Path) -> None:
    config = load_robustness_config()
    original = config.research_plan_path.read_text(encoding="utf-8")
    changed = tmp_path / "retuned.toml"
    changed.write_text(
        original.replace('minimum_closed_trades = "30"', 'minimum_closed_trades = "16"'),
        encoding="utf-8",
    )
    baseline = load_research_baseline_manifest()

    assert load_research_plan(changed, baseline=baseline).plan_hash != load_research_plan(
        baseline=baseline
    ).plan_hash


def test_holdout_identity_is_stable_but_frozen_core_is_tamper_evident() -> None:
    plan = load_research_plan()
    first = make_untouched_holdout(plan, core_hash="a" * 64)
    changed = make_untouched_holdout(plan, core_hash="b" * 64)

    assert first.holdout_id == changed.holdout_id
    assert first.record_hash != changed.record_hash
    assert "datasets" not in dict(first.expected_config_hashes)


def test_holdout_consumes_once_exact_rerun_reproduces_and_changed_core_invalidates() -> None:
    plan = load_research_plan()
    record = make_untouched_holdout(plan, core_hash="a" * 64)
    consumed, reproduced = consume_holdout(
        record,
        result_hash="c" * 64,
        core_hash="a" * 64,
        config_hashes=record.expected_config_hashes,
        consumed_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    assert consumed.status is HoldoutStatus.CONSUMED
    assert reproduced is False

    rerun, reproduced = consume_holdout(
        consumed,
        result_hash="c" * 64,
        core_hash="a" * 64,
        config_hashes=record.expected_config_hashes,
    )
    assert rerun == consumed
    assert reproduced is True

    invalidated, reproduced = consume_holdout(
        consumed,
        result_hash="d" * 64,
        core_hash="b" * 64,
        config_hashes=record.expected_config_hashes,
        consumed_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    assert invalidated.status is HoldoutStatus.INVALIDATED
    assert reproduced is False
    assert invalidated.invalidation_reason == "CORE_OR_CONFIG_CHANGED_AFTER_HOLDOUT_CONSUMPTION"


def test_plan_command_cannot_regress_or_recreate_a_consumed_period(
    tmp_path: Path,
) -> None:
    service = RobustnessService(tmp_path / "data_local")
    plan = load_research_plan()
    untouched = make_untouched_holdout(plan, core_hash=decision_core_hash())
    service.store.save_holdout(untouched)
    consumed, _ = consume_holdout(
        untouched,
        result_hash="c" * 64,
        core_hash=untouched.expected_core_hash,
        config_hashes=untouched.expected_config_hashes,
        consumed_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    service.store.save_holdout(consumed)

    assert service.freeze_plan()["holdout"]["status"] == "CONSUMED"
    with pytest.raises(RobustnessStorageError, match="cannot regress"):
        service.store.save_holdout(untouched)

    changed_plan = replace(plan, plan_hash="f" * 64)
    same_period = service._current_holdout(changed_plan)
    assert same_period.holdout_id == consumed.holdout_id
    assert same_period.status is HoldoutStatus.CONSUMED
    with pytest.raises(HoldoutGovernanceError, match="different frozen plan"):
        service._require_frozen_plan_match(same_period, changed_plan)

    renamed_periods = tuple(
        replace(item, label="RENAMED_DOES_NOT_CREATE_NEW_EVIDENCE")
        if item.classification is PeriodClassification.FINAL_HOLDOUT
        else item
        for item in plan.periods
    )
    renamed_plan = replace(plan, plan_hash="e" * 64, periods=renamed_periods)
    assert service._current_holdout(renamed_plan).holdout_id == consumed.holdout_id
    with pytest.raises(HoldoutGovernanceError, match="different frozen plan"):
        service._require_frozen_plan_match(consumed, renamed_plan)


def test_consumed_holdout_status_immediately_invalidates_after_core_change(
    tmp_path: Path,
) -> None:
    service = RobustnessService(tmp_path / "data_local")
    plan = load_research_plan()
    stale_core = "a" * 64
    untouched = make_untouched_holdout(plan, core_hash=stale_core)
    service.store.save_holdout(untouched)
    consumed, _ = consume_holdout(
        untouched,
        result_hash="c" * 64,
        core_hash=stale_core,
        config_hashes=untouched.expected_config_hashes,
        consumed_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    service.store.save_holdout(consumed)

    current = service._current_holdout(plan)
    assert current.status is HoldoutStatus.INVALIDATED
    assert current.invalidation_reason == "CORE_OR_CONFIG_CHANGED_AFTER_HOLDOUT_CONSUMPTION"
    assert service.holdout_status()["status"] == "INVALIDATED"


@pytest.mark.parametrize(
    "consumer",
    (
        HoldoutConsumer.TRAINING_PIPELINE,
        HoldoutConsumer.EDGE_ESTIMATOR,
        HoldoutConsumer.CONFIG_SELECTION,
    ),
)
def test_untouched_holdout_is_inaccessible_to_training_edge_and_selection(consumer) -> None:
    record = make_untouched_holdout(load_research_plan(), core_hash="a" * 64)
    with pytest.raises(Exception, match="forbidden"):
        HoldoutAccessPolicy.authorize(record, consumer)
    HoldoutAccessPolicy.authorize(record, HoldoutConsumer.FINAL_EVALUATION)


def test_point_in_time_membership_has_explicit_intervals_and_no_implicit_resolution() -> None:
    universe = PointInTimeUniverse(
        universe_id="official-example",
        source="official membership archive",
        status=SurvivorshipStatus.POINT_IN_TIME,
        memberships=(
            UniverseMembership(
                symbol="AAA",
                valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
                valid_to=datetime(2022, 1, 1, tzinfo=timezone.utc),
                source="official archive",
                status="VERIFIED",
            ),
            UniverseMembership(
                symbol="BBB",
                valid_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
                valid_to=None,
                source="official archive",
                status="VERIFIED",
            ),
        ),
    )
    assert universe.members_at(datetime(2020, 6, 1, tzinfo=timezone.utc)) == ("AAA",)
    assert universe.members_at(datetime(2022, 6, 1, tzinfo=timezone.utc)) == ("BBB",)


def test_robustness_source_never_mutates_decision_configuration() -> None:
    root = Path(__file__).parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "src" / "trading_ai" / "robustness").glob("*.py"))
    )
    forbidden = ("config.write_", "write_text(config", "GridSearch", "Optuna")
    assert not any(item in source for item in forbidden)
    assert stable_hash(source) == stable_hash(source)


def test_robustness_layer_has_no_trading_broker_or_network_dependency() -> None:
    root = Path(__file__).parents[1] / "src" / "trading_ai" / "robustness"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py"))
    )
    forbidden_imports = (
        "from trading_ai.backtesting",
        "from trading_ai.strategies",
        "from trading_ai.risk",
        "from trading_ai.portfolio",
        "from trading_ai.execution",
        "BrokerAdapter",
        "yfinance",
        "requests",
        "urllib.request",
    )
    assert not any(item in source for item in forbidden_imports)


def _leave_one_summary(
    symbols: tuple[str, ...], strategies: tuple[str, ...]
) -> dict[str, object]:
    period = load_research_plan().period(PeriodClassification.CONSUMED_DIAGNOSTIC)
    return {
        "dataset_references": [
            {
                "symbol": symbol,
                "requested_start": period.start.isoformat(),
                "requested_end": period.end.isoformat(),
            }
            for symbol in symbols
        ],
        "strategy_parameters": [
            [f"strategy.{strategy}.timeframe", "1d"] for strategy in strategies
        ],
    }


def test_leave_one_diagnostics_require_exactly_one_declared_exclusion() -> None:
    plan = load_research_plan()
    period = plan.period(PeriodClassification.CONSUMED_DIAGNOSTIC)
    without_aapl = tuple(item for item in plan.symbols if item != "AAPL")
    RobustnessService._validate_leave_one_runs(
        {"AAPL": _leave_one_summary(without_aapl, plan.strategies)},
        expected_items=plan.symbols,
        expected_symbols=plan.symbols,
        expected_strategies=plan.strategies,
        period=period,
        kind="symbol",
    )
    with pytest.raises(ValueError, match="exactly"):
        RobustnessService._validate_leave_one_runs(
            {"AAPL": _leave_one_summary(plan.symbols, plan.strategies)},
            expected_items=plan.symbols,
            expected_symbols=plan.symbols,
            expected_strategies=plan.strategies,
            period=period,
            kind="symbol",
        )

    reference = _leave_one_summary(plan.symbols, plan.strategies)
    retuned = _leave_one_summary(without_aapl, plan.strategies)
    retuned["strategy_parameters"][0][1] = "4h"  # type: ignore[index]
    with pytest.raises(ValueError, match="strategy parameter"):
        RobustnessService._validate_leave_one_runs(
            {"AAPL": retuned},
            expected_items=plan.symbols,
            expected_symbols=plan.symbols,
            expected_strategies=plan.strategies,
            reference_strategy_parameters=reference["strategy_parameters"],
            period=period,
            kind="symbol",
        )

    without_trend = tuple(item for item in plan.strategies if item != "trend")
    RobustnessService._validate_leave_one_runs(
        {"trend": _leave_one_summary(plan.symbols, without_trend)},
        expected_items=plan.strategies,
        expected_symbols=plan.symbols,
        expected_strategies=plan.strategies,
        period=period,
        kind="strategy",
    )
    with pytest.raises(ValueError, match="exactly"):
        RobustnessService._validate_leave_one_runs(
            {"trend": _leave_one_summary(plan.symbols, plan.strategies)},
            expected_items=plan.strategies,
            expected_symbols=plan.symbols,
            expected_strategies=plan.strategies,
            period=period,
            kind="strategy",
        )


def test_single_strategy_diagnostics_preserve_universe_period_and_parameters() -> None:
    plan = load_research_plan()
    period = plan.period(PeriodClassification.CONSUMED_DIAGNOSTIC)
    reference = _leave_one_summary(plan.symbols, plan.strategies)
    trend_only = _leave_one_summary(plan.symbols, ("trend",))
    RobustnessService._validate_single_strategy_runs(
        {"trend": trend_only},
        expected_strategies=plan.strategies,
        expected_symbols=plan.symbols,
        expected_config_hashes=(),
        reference_strategy_parameters=reference["strategy_parameters"],
        period=period,
    )

    legacy_single = dict(trend_only)
    legacy_single["strategy_name"] = "trend"
    legacy_single["strategy_parameters"] = [
        [str(key).removeprefix("strategy.trend."), value]
        for key, value in trend_only["strategy_parameters"]
    ]
    RobustnessService._validate_single_strategy_runs(
        {"trend": legacy_single},
        expected_strategies=plan.strategies,
        expected_symbols=plan.symbols,
        expected_config_hashes=(),
        reference_strategy_parameters=reference["strategy_parameters"],
        period=period,
    )

    wrong_strategy = _leave_one_summary(plan.symbols, ("trend", "momentum"))
    with pytest.raises(ValueError, match="exactly"):
        RobustnessService._validate_single_strategy_runs(
            {"trend": wrong_strategy},
            expected_strategies=plan.strategies,
            expected_symbols=plan.symbols,
            expected_config_hashes=(),
            reference_strategy_parameters=reference["strategy_parameters"],
            period=period,
        )

    retuned = _leave_one_summary(plan.symbols, ("trend",))
    retuned["strategy_parameters"][0][1] = "4h"  # type: ignore[index]
    with pytest.raises(ValueError, match="strategy parameters"):
        RobustnessService._validate_single_strategy_runs(
            {"trend": retuned},
            expected_strategies=plan.strategies,
            expected_symbols=plan.symbols,
            expected_config_hashes=(),
            reference_strategy_parameters=reference["strategy_parameters"],
            period=period,
        )


def test_robustness_and_holdout_cli_are_explicit_offline_and_read_only(
    tmp_path: Path, capsys
) -> None:
    data_root = tmp_path / "data_local"
    assert main(("robustness", "plan", "--data-root", str(data_root), "--json")) == 0
    plan = __import__("json").loads(capsys.readouterr().out)
    assert plan["network_access"] is False
    assert plan["thresholds_relaxed"] is False
    assert plan["holdout"]["status"] == "UNTOUCHED"

    assert main(("validation", "holdout-status", "--data-root", str(data_root), "--json")) == 0
    holdout = __import__("json").loads(capsys.readouterr().out)
    assert holdout["status"] == "UNTOUCHED"
    assert holdout["period"]["label"].startswith("FINAL_HOLDOUT_V2")
