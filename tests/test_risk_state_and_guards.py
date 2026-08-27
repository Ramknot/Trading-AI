"""State, volatility, concentration, and exact-alignment correlation tests."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from backtest_support import bar
from test_balanced_risk_engine import (
    BASE_CONFIG,
    NOW,
    _context,
    _engine,
    _feature,
)
from trading_ai.core.models import OrderSide, Position, RiskDecisionStatus
from trading_ai.features import FeatureEngine, FeatureRequest
from trading_ai.features.models import ReturnObservation, ReturnSeries
from trading_ai.risk.correlation import CorrelationGuard, aligned_correlation
from trading_ai.risk.models import (
    CircuitBreakerReason,
    RiskReasonCode,
    RiskState,
    UnknownRiskPolicy,
)
from trading_ai.risk.state import RiskStateTracker


def _series(
    symbol: str,
    values: tuple[float, ...],
    *,
    offset_days: int = 0,
) -> ReturnSeries:
    return ReturnSeries(
        symbol=symbol,
        timeframe="1d",
        observations=tuple(
            ReturnObservation(
                timestamp=NOW + timedelta(days=index + offset_days),
                value=value,
            )
            for index, value in enumerate(values)
        ),
    )


def test_daily_loss_halts_new_risk_at_or_beyond_configured_threshold() -> None:
    tracker = RiskStateTracker(BASE_CONFIG)
    tracker.reset(NOW, Decimal("100000"))

    state = tracker.observe(NOW + timedelta(hours=1), Decimal("97999"))

    assert state.state is RiskState.HALTED
    assert state.daily_loss_pct == pytest.approx(0.02001)
    assert state.halt_reason == CircuitBreakerReason.DAILY_LOSS_LIMIT.value


def test_daily_halt_resets_next_risk_day_but_peak_is_retained() -> None:
    tracker = RiskStateTracker(BASE_CONFIG)
    tracker.reset(NOW, Decimal("100000"))
    tracker.observe(NOW + timedelta(hours=1), Decimal("97999"))

    next_day = tracker.observe(NOW + timedelta(days=1), Decimal("98000"))

    assert next_day.state is RiskState.NORMAL
    assert next_day.daily_loss_pct == 0.0
    assert next_day.peak_equity == Decimal("100000")


def test_soft_drawdown_reduces_new_risk_quantity() -> None:
    engine = _engine()
    engine.reset(NOW, Decimal("100000"))
    later = NOW + timedelta(days=1)
    state = engine.current_state(later, Decimal("95000"))
    assert state.state is RiskState.REDUCED

    decision = engine.evaluate_context(
        _context(
            engine,
            quantity="10",
            cash="95000",
            equity="95000",
            timestamp=later,
            feature_snapshot=_feature(0.01, later),
            reset=False,
        )
    )
    assert decision.status is RiskDecisionStatus.REDUCE
    assert decision.approved_quantity == Decimal("5")
    assert RiskReasonCode.SOFT_DRAWDOWN.value in decision.reason_codes


def test_hard_drawdown_halt_is_persistent_until_explicit_reset() -> None:
    engine = _engine()
    engine.reset(NOW, Decimal("100000"))
    hard_time = NOW + timedelta(days=1)
    assert engine.current_state(hard_time, Decimal("89999")).state is RiskState.HALTED

    recovery_time = NOW + timedelta(days=2)
    recovered = engine.current_state(recovery_time, Decimal("99000"))
    assert recovered.state is RiskState.HALTED
    assert recovered.halt_reason == CircuitBreakerReason.HARD_DRAWDOWN.value

    reset = engine.reset_halt(
        recovery_time,
        Decimal("99000"),
        authorization_reason="explicit test operator reset",
    )
    assert reset.state is RiskState.NORMAL


def test_manual_halt_requires_an_explicit_nonempty_reset_reason() -> None:
    engine = _engine()
    engine.reset(NOW, Decimal("100000"))
    engine.halt(CircuitBreakerReason.MANUAL_HALT, NOW, Decimal("100000"))

    with pytest.raises(ValueError, match="authorization_reason"):
        engine.reset_halt(NOW, Decimal("100000"), authorization_reason="")
    assert engine.state_snapshot.state is RiskState.HALTED


def test_circuit_breaker_records_normal_reduced_halted_sequence() -> None:
    engine = _engine()
    engine.reset(NOW, Decimal("100000"))
    engine.current_state(NOW + timedelta(days=1), Decimal("95000"))
    engine.current_state(NOW + timedelta(days=2), Decimal("89000"))

    assert [item.previous_state for item in engine.state_transitions] == [
        RiskState.NORMAL,
        RiskState.REDUCED,
    ]
    assert [item.new_state for item in engine.state_transitions] == [
        RiskState.REDUCED,
        RiskState.HALTED,
    ]


@pytest.mark.parametrize(
    ("metric", "status", "approved"),
    [
        (0.039, RiskDecisionStatus.APPROVE, Decimal("10")),
        (0.040, RiskDecisionStatus.REDUCE, Decimal("5")),
        (0.080, RiskDecisionStatus.REJECT, Decimal("0")),
    ],
)
def test_volatility_guard_uses_configured_feature_thresholds(
    metric: float,
    status: RiskDecisionStatus,
    approved: Decimal,
) -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(engine, feature_snapshot=_feature(metric))
    )
    assert decision.status is status
    assert decision.approved_quantity == approved


def test_missing_volatility_is_warning_by_default_and_can_fail_closed() -> None:
    warning_engine = _engine()
    warning = warning_engine.evaluate_context(_context(warning_engine))
    assert warning.status is RiskDecisionStatus.APPROVE
    assert RiskReasonCode.VOLATILITY_UNKNOWN.value in warning.reason_codes

    strict_config = replace(
        BASE_CONFIG,
        missing_volatility_policy=UnknownRiskPolicy.REJECT,
    )
    strict_engine = _engine(config=strict_config)
    rejected = strict_engine.evaluate_context(_context(strict_engine))
    assert rejected.status is RiskDecisionStatus.REJECT


def test_appending_future_bars_cannot_change_risk_volatility_at_t() -> None:
    history = tuple(
        bar(
            index,
            opening=str(100 + (index % 2) * 0.1),
            high="101",
            low="99",
            close=str(100 + (index % 2) * 0.1),
            timestamp=NOW + timedelta(days=index),
        )
        for index in range(25)
    )
    future = tuple(
        bar(
            25 + index,
            opening=str(100 + index * 20),
            high=str(121 + index * 20),
            low=str(79 + index * 20),
            close=str(120 + index * 20),
            timestamp=NOW + timedelta(days=25 + index),
        )
        for index in range(3)
    )
    request = FeatureRequest()
    feature_engine = FeatureEngine()
    at_time = history[-1].timestamp
    before = feature_engine.compute(history, request, as_of=at_time)
    after = feature_engine.compute((*history, *future), request, as_of=at_time)

    first_engine = _engine()
    second_engine = _engine()
    first = first_engine.evaluate_context(
        _context(
            first_engine,
            timestamp=at_time,
            feature_snapshot=before,
        )
    )
    second = second_engine.evaluate_context(
        _context(
            second_engine,
            timestamp=at_time,
            feature_snapshot=after,
        )
    )
    assert before == after
    assert first == second


def test_group_concentration_reduces_same_group_exposure() -> None:
    engine = _engine()
    position = Position("MSFT", Decimal("250"), Decimal("100"))
    decision = engine.evaluate_context(
        _context(
            engine,
            quantity="100",
            cash="75000",
            equity="100000",
            positions=(position,),
            prices={"MSFT": Decimal("100")},
            feature_snapshot=_feature(0.01),
        )
    )
    assert decision.status is RiskDecisionStatus.REDUCE
    assert decision.approved_quantity == Decimal("50")
    assert RiskReasonCode.CONCENTRATION_LIMIT.value in decision.reason_codes


def test_correlation_calculates_plus_one_zero_and_minus_one() -> None:
    x = _series("AAPL", (-1, 0, 1, -1, 0, 1))
    positive = _series("MSFT", (-2, 0, 2, -2, 0, 2))
    uncorrelated = _series("META", (1, 1, -2, -1, -1, 2))
    negative = _series("QQQ", (1, 0, -1, 1, 0, -1))

    assert aligned_correlation(x, positive, minimum_observations=6)[0] == pytest.approx(1)
    assert aligned_correlation(x, uncorrelated, minimum_observations=6)[0] == pytest.approx(0)
    assert aligned_correlation(x, negative, minimum_observations=6)[0] == pytest.approx(-1)


def test_correlation_uses_only_exact_common_timestamps_without_fill() -> None:
    left = _series("AAPL", (0.1, 0.2, 0.3))
    disjoint = _series("MSFT", (0.1, 0.2, 0.3), offset_days=10)

    coefficient, observations = aligned_correlation(
        left, disjoint, minimum_observations=2
    )
    assert coefficient is None
    assert observations == 0


def test_correlation_guard_reports_insufficient_observations() -> None:
    guard = CorrelationGuard(threshold=0.85, minimum_observations=20)
    assessments = guard.assess(
        "AAPL",
        ("MSFT",),
        (_series("AAPL", (0.1, 0.2)), _series("MSFT", (0.1, 0.2))),
    )
    assert assessments[0].coefficient is None
    assert assessments[0].observations == 2
    assert assessments[0].highly_correlated is False


def test_high_correlation_exposure_reduces_quantity() -> None:
    config = replace(BASE_CONFIG, correlation_min_observations=3)
    engine = _engine(config=config)
    returns = (
        _series("AAPL", (0.01, 0.02, 0.03)),
        _series("QQQ", (0.02, 0.04, 0.06)),
    )
    decision_time = NOW + timedelta(days=2)
    position = Position("QQQ", Decimal("200"), Decimal("100"))
    decision = engine.evaluate_context(
        _context(
            engine,
            quantity="150",
            cash="80000",
            equity="100000",
            positions=(position,),
            prices={"QQQ": Decimal("100")},
            timestamp=decision_time,
            feature_snapshot=_feature(0.01, decision_time),
            return_series=returns,
        )
    )
    assert decision.status is RiskDecisionStatus.REDUCE
    assert decision.approved_quantity == Decimal("100")
    assert decision.correlation_metric == pytest.approx(1)
    assert RiskReasonCode.CORRELATION_LIMIT.value in decision.reason_codes


def test_unknown_correlation_policy_warns_or_rejects_explicitly() -> None:
    position = Position("QQQ", Decimal("1"), Decimal("100"))
    warning_engine = _engine()
    warning = warning_engine.evaluate_context(
        _context(
            warning_engine,
            positions=(position,),
            prices={"QQQ": Decimal("100")},
            feature_snapshot=_feature(0.01),
        )
    )
    assert warning.status is RiskDecisionStatus.APPROVE
    assert RiskReasonCode.CORRELATION_UNKNOWN.value in warning.reason_codes

    strict_config = replace(
        BASE_CONFIG,
        correlation_unknown_policy=UnknownRiskPolicy.REJECT,
    )
    strict_engine = _engine(config=strict_config)
    rejected = strict_engine.evaluate_context(
        _context(
            strict_engine,
            positions=(position,),
            prices={"QQQ": Decimal("100")},
            feature_snapshot=_feature(0.01),
        )
    )
    assert rejected.status is RiskDecisionStatus.REJECT


def test_halted_exit_is_not_blocked_by_extreme_volatility() -> None:
    engine = _engine()
    position = Position("AAPL", Decimal("10"), Decimal("100"))
    engine.reset(NOW, Decimal("10000"))
    engine.halt(CircuitBreakerReason.INVALID_MARKET_DATA, NOW, Decimal("10000"))
    decision = engine.evaluate_context(
        _context(
            engine,
            quantity="5",
            side=OrderSide.SELL,
            cash="9000",
            equity="10000",
            positions=(position,),
            feature_snapshot=_feature(0.5),
            reset=False,
        )
    )
    assert decision.status is RiskDecisionStatus.APPROVE
    assert decision.approved_quantity == Decimal("5")
