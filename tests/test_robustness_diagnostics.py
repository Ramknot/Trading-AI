from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from trading_ai.robustness.config import load_research_plan
from trading_ai.robustness.diagnostics import (
    ConcentrationAnalyzer,
    DecisionFunnelAnalyzer,
    DrawdownAnalyzer,
    HistoricalCoverageAnalyzer,
    StatisticalUncertaintyAnalyzer,
    TemporalAnalyzer,
    comparison_result,
)
from trading_ai.robustness.models import DiagnosticAvailability, UncertaintyStatus


def _at(day: int) -> str:
    return datetime(2020, 1, day, tzinfo=timezone.utc).isoformat()


def test_decision_funnel_is_monotone_and_keeps_closed_trades_symbol_level() -> None:
    tables = {
        "signals": (
            {"signal_id": "s1", "strategy_name": "trend", "symbol": "AAA", "action": "ENTER_LONG"},
            {"signal_id": "s2", "strategy_name": "trend", "symbol": "AAA", "action": "ENTER_LONG"},
            {"signal_id": "sx", "strategy_name": "trend", "symbol": "AAA", "action": "EXIT_LONG"},
        ),
        "activation_decisions": (
            {"signal_id": "s1", "status": "ALLOW", "reason_codes": []},
            {"signal_id": "s2", "status": "BLOCK", "reason_codes": ["RANGE"]},
        ),
        "portfolio_opportunities": (
            {"opportunity_id": "opp1", "signal_id": "s1"},
        ),
        "portfolio_decisions": (
            {"opportunity_id": "opp1", "signal_id": "s1", "status": "SELECT", "reason_codes": []},
        ),
        "economic_decisions": (
            {"signal_id": "s1", "status": "PASS", "allows_new_risk": True},
        ),
        "orders": ({"order_id": "o1", "signal_id": "s1", "symbol": "AAA"},),
        "risk_decisions": ({"order_id": "o1", "status": "APPROVE", "reason_codes": []},),
        "fills": ({"order_id": "o1", "side": "BUY"},),
        "trades": ({"symbol": "AAA"},),
    }
    report = DecisionFunnelAnalyzer.build({"ml": {"mode": "DISABLED"}}, tables)
    trend = next(row for row in report.rows if row.strategy_name == "trend")
    aggregate = next(row for row in report.rows if row.strategy_name == "ALL_STRATEGIES")
    assert (trend.candidate_entries, trend.activation_eligible, trend.filled_entries) == (2, 1, 1)
    assert trend.closed_trades == 0
    assert aggregate.closed_trades == 1
    assert ("RANGE", 1) in report.drop_reasons


def test_historical_coverage_uses_the_true_common_interval_and_keeps_warnings() -> None:
    summary = {
        "dataset_references": [
            {
                "symbol": "AAA", "timeframe": "1d", "dataset_id": "dataset-a",
                "checksum_sha256": "a" * 64, "provider": "fake",
            },
            {
                "symbol": "BBB", "timeframe": "1d", "dataset_id": "dataset-b",
                "checksum_sha256": "b" * 64, "provider": "fake",
            },
        ],
        "data_quality_reports": [
            {
                "symbol": "AAA", "timeframe": "1d", "row_count": 10,
                "first_timestamp": "2020-01-01T00:00:00+00:00",
                "last_timestamp": "2020-01-10T00:00:00+00:00",
                "missing_expected_bar_count": 0, "duplicate_count": 0,
                "invalid_bar_count": 0, "quality_status": "PASS", "warnings": [],
            },
            {
                "symbol": "BBB", "timeframe": "1d", "row_count": 6,
                "first_timestamp": "2020-01-03T00:00:00+00:00",
                "last_timestamp": "2020-01-08T00:00:00+00:00",
                "missing_expected_bar_count": 1, "duplicate_count": 0,
                "invalid_bar_count": 0, "quality_status": "WARNING",
                "warnings": ["EXPECTED_SESSION_GAP"],
            },
        ],
    }
    report = HistoricalCoverageAnalyzer.build(summary)
    assert report.common_start == datetime(2020, 1, 3, tzinfo=timezone.utc)
    assert report.common_end == datetime(2020, 1, 8, tzinfo=timezone.utc)
    assert report.common_history_available is True
    assert "BBB:EXPECTED_SESSION_GAP" in report.provider_limitations
    assert report.rows[1].missing_expected_bars == 1


def test_drawdown_episode_finds_peak_trough_recovery_and_observed_contributor() -> None:
    tables = {
        "equity": tuple(
            {"timestamp": _at(day), "equity": equity, "positions_value": "50"}
            for day, equity in ((1, "100"), (2, "90"), (3, "80"), (4, "101"))
        ),
        "ledger": ({"timestamp": _at(1), "symbol": "AAA", "quantity_change": "1"},),
        "trades": ({"exit_time": _at(3), "symbol": "AAA", "net_pnl": "-20"},),
        "risk_decisions": ({"timestamp": _at(2), "status": "REDUCE"},),
        "risk_states": ({"timestamp": _at(2), "new_state": "REDUCED"},),
        "regime_snapshots": (
            {"timestamp": _at(2), "symbol": "AAA", "structure_regime": "RANGE", "volatility_regime": "HIGH"},
        ),
        "cost_actuals": (),
    }
    episode = DrawdownAnalyzer.build(tables, Decimal("0.05"))[0]
    assert episode.peak_equity == 100
    assert episode.trough_equity == 80
    assert episode.drawdown_fraction == Decimal("0.2")
    assert episode.recovery_timestamp == datetime(2020, 1, 4, tzinfo=timezone.utc)
    assert episode.held_symbols_at_trough == ("AAA",)
    assert episode.realized_closure_pnl_by_symbol == (("AAA", Decimal("-20")),)


def test_concentration_reports_top_shares_hhi_and_keeps_losses_visible() -> None:
    report = ConcentrationAnalyzer.build(
        {
            "trades": (
                {"symbol": "A", "gross_pnl": "80", "net_pnl": "80"},
                {"symbol": "B", "gross_pnl": "20", "net_pnl": "20"},
                {"symbol": "C", "gross_pnl": "-10", "net_pnl": "-10"},
            ),
            "portfolio_targets": (),
        },
        ("A", "B", "C"),
        Decimal("0.75"),
    )
    assert report.top_contributor == "A"
    assert report.top1_positive_pnl_share == Decimal("0.8")
    assert report.top3_positive_pnl_share == Decimal("1")
    assert report.positive_pnl_hhi == Decimal("0.68")
    assert report.dominant_symbol_warning is True
    assert next(item for item in report.symbols if item.symbol == "C").gross_loss == -10


def test_concentration_with_no_positive_pnl_does_not_invent_shares() -> None:
    report = ConcentrationAnalyzer.build(
        {"trades": ({"symbol": "A", "gross_pnl": "-1", "net_pnl": "-1"},), "portfolio_targets": ()},
        ("A",),
        Decimal("0.75"),
    )
    assert report.top1_positive_pnl_share is None
    assert report.positive_pnl_hhi is None


def test_temporal_report_has_calendar_years_and_three_predeclared_nonoverlapping_subperiods() -> None:
    period = load_research_plan().period(
        __import__("trading_ai.robustness.models", fromlist=["PeriodClassification"]).PeriodClassification.CONSUMED_DIAGNOSTIC
    )
    rows = TemporalAnalyzer.build(
        {
            "equity": (
                {"timestamp": "2020-01-02T00:00:00+00:00", "equity": "100", "positions_value": "0"},
                {"timestamp": "2024-12-31T00:00:00+00:00", "equity": "120", "positions_value": "0"},
            ),
            "trades": (), "cost_actuals": (), "fills": (),
        },
        period=period,
    )
    subperiods = [item for item in rows if item.label.startswith("SUBPERIOD_")]
    assert len(subperiods) == 3
    assert subperiods[0].start == period.start
    assert subperiods[-1].end == period.end
    assert all(left.end == right.start for left, right in zip(subperiods, subperiods[1:]))
    assert any(item.availability is DiagnosticAvailability.UNAVAILABLE for item in rows)


def test_small_sample_uncertainty_stays_explicitly_insufficient_and_deterministic() -> None:
    trades = tuple({"net_pnl": value} for value in ("1", "-1", "2"))
    first = StatisticalUncertaintyAnalyzer.build(
        trades, initial_cash=Decimal("100"), minimum_samples=30, seed=8202, resamples=100
    )
    second = StatisticalUncertaintyAnalyzer.build(
        trades, initial_cash=Decimal("100"), minimum_samples=30, seed=8202, resamples=100
    )
    assert first == second
    assert first.status is UncertaintyStatus.INSUFFICIENT_FOR_RELIABLE_CI
    assert first.expectancy_interval_95 is None


def test_single_strategy_comparison_explicitly_ignores_only_legacy_portfolio_hash() -> None:
    expected = tuple(
        sorted(
            {
                "cost": "a" * 64,
                "policy": "b" * 64,
                "portfolio": "c" * 64,
                "regime": "d" * 64,
                "risk": "e" * 64,
            }.items()
        )
    )
    summary = {
        "run_id": "bt-trend-only",
        "metrics": {"number_of_trades": 1},
        "costs": {"config_hash": "a" * 64, "summary": {}},
        "regime": {"config_hash": "d" * 64, "policy_config_hash": "b" * 64},
        "portfolio": {"config_hash": "f" * 64},
        "risk": {"config_hash": "e" * 64},
    }
    result = comparison_result(
        diagnostic_type="SINGLE_STRATEGY_COMPARISON",
        excluded_item="trend",
        summary=summary,
        expected_config_hashes=expected,
        label_prefix="only",
        ignored_config_hash_names=("portfolio",),
    )
    assert result.label == "only-trend"
    assert result.config_hashes_unchanged is True
