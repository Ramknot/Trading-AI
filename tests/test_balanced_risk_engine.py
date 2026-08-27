"""Decision-level tests for the fail-closed Balanced Risk Engine."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trading_ai.core.config import load_runtime_settings
from trading_ai.core.models import (
    OrderRequest,
    OrderSide,
    PortfolioSnapshot,
    Position,
    RiskDecisionStatus,
    TradingContext,
)
from trading_ai.features import FeatureSnapshot, FeatureValue
from trading_ai.features.models import ReturnSeries
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.config import load_balanced_risk_config
from trading_ai.risk.models import (
    CircuitBreakerReason,
    RiskContext,
    RiskReasonCode,
    RiskState,
    UnknownRiskPolicy,
)


NOW = datetime(2024, 1, 2, 21, tzinfo=timezone.utc)
ZERO = Decimal("0")
PROFILE = load_runtime_settings().profile
BASE_CONFIG, BASE_GROUPS = load_balanced_risk_config(PROFILE)


def _engine(*, config=None, groups=None) -> BalancedRiskEngine:
    return BalancedRiskEngine(
        PROFILE,
        config or BASE_CONFIG,
        groups or BASE_GROUPS,
    )


def _feature(value: float | None, timestamp: datetime = NOW) -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="AAPL",
        timestamp=timestamp,
        timeframe="1d",
        values=(FeatureValue("rolling_vol_20", value),),
    )


def _context(
    engine: BalancedRiskEngine,
    *,
    quantity: str = "10",
    side: OrderSide = OrderSide.BUY,
    price: str = "100",
    cash: str = "100000",
    equity: str = "100000",
    positions: tuple[Position, ...] = (),
    prices: dict[str, Decimal] | None = None,
    timestamp: datetime = NOW,
    feature_snapshot: FeatureSnapshot | None = None,
    invalidation_price: str | None = None,
    risk_distance: str | None = None,
    reset: bool = True,
    symbol: str = "AAPL",
    return_series: tuple[ReturnSeries, ...] = (),
) -> RiskContext:
    equity_value = Decimal(equity)
    if reset:
        engine.reset(timestamp, equity_value)
    market_prices = dict(prices or {})
    market_prices.setdefault(symbol, Decimal(price))
    for position in positions:
        market_prices.setdefault(position.symbol, Decimal("100"))
    portfolio = PortfolioSnapshot(
        as_of=timestamp,
        cash=Decimal(cash),
        total_equity=equity_value,
        positions=positions,
    )
    order = OrderRequest(
        order_id="risk-order",
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        created_at=timestamp,
        expected_entry_price=Decimal(price),
        invalidation_price=(
            Decimal(invalidation_price) if invalidation_price is not None else None
        ),
        risk_distance=(Decimal(risk_distance) if risk_distance is not None else None),
    )
    state = engine.current_state(timestamp, equity_value)
    return RiskContext(
        timestamp=timestamp,
        profile=PROFILE,
        portfolio=portfolio,
        order=order,
        expected_entry_price=Decimal(price),
        market_prices=tuple(sorted(market_prices.items())),
        risk_state=state,
        timeframe="1d",
        feature_snapshot=feature_snapshot,
        return_series=return_series,
    )


def _codes(decision) -> set[str]:
    return set(decision.reason_codes)


def test_small_balanced_entry_is_approved_with_traceable_warning() -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(engine, feature_snapshot=_feature(0.01))
    )

    assert decision.status is RiskDecisionStatus.APPROVE
    assert decision.requested_quantity == decision.approved_quantity == Decimal("10")
    assert decision.engine_name == "balanced-risk"
    assert decision.engine_version == "1.0"
    assert decision.config_hash == engine.config_hash
    assert RiskReasonCode.NO_EXPLICIT_RISK_DISTANCE.value in _codes(decision)
    assert decision.risk_state == RiskState.NORMAL.value
    with pytest.raises(FrozenInstanceError):
        decision.approved_quantity = Decimal("11")  # type: ignore[misc]


def test_single_position_exposure_reduces_a_large_request() -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(engine, quantity="1000", feature_snapshot=_feature(0.01))
    )

    assert decision.status is RiskDecisionStatus.REDUCE
    assert decision.approved_quantity == Decimal("150")
    assert decision.position_exposure_after == pytest.approx(0.15)
    assert RiskReasonCode.POSITION_LIMIT.value in _codes(decision)


def test_single_position_exact_limit_is_approved() -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(engine, quantity="150", feature_snapshot=_feature(0.01))
    )
    assert decision.status is RiskDecisionStatus.APPROVE
    assert decision.approved_quantity == Decimal("150")


def test_portfolio_exposure_reduces_then_rejects_when_no_capacity() -> None:
    positions = (
        Position("QQQ", Decimal("1.5"), Decimal("100")),
        Position("AMZN", Decimal("1.5"), Decimal("100")),
        Position("META", Decimal("1.5"), Decimal("100")),
        Position("AIR.PA", Decimal("1"), Decimal("100")),
    )
    prices = {item.symbol: Decimal("100") for item in positions}
    engine = _engine()
    reduced = engine.evaluate_context(
        _context(
            engine,
            quantity="1",
            cash="450",
            equity="1000",
            positions=positions,
            prices=prices,
            feature_snapshot=_feature(0.01),
        )
    )
    assert reduced.status is RiskDecisionStatus.REDUCE
    assert reduced.approved_quantity == Decimal("0.5")
    assert RiskReasonCode.PORTFOLIO_EXPOSURE_LIMIT.value in _codes(reduced)

    full_positions = (
        *positions[:-1],
        Position("AIR.PA", Decimal("1.5"), Decimal("100")),
    )
    rejected = engine.evaluate_context(
        _context(
            engine,
            quantity="1",
            cash="400",
            equity="1000",
            positions=full_positions,
            prices={item.symbol: Decimal("100") for item in full_positions},
            feature_snapshot=_feature(0.01),
        )
    )
    assert rejected.status is RiskDecisionStatus.REJECT
    assert rejected.approved_quantity == ZERO
    assert RiskReasonCode.PORTFOLIO_EXPOSURE_LIMIT.value in _codes(rejected)


def test_sixth_position_is_rejected_but_existing_position_can_be_increased() -> None:
    positions = tuple(
        Position(symbol, Decimal("1"), Decimal("100"))
        for symbol in ("QQQ", "IWM", "MSFT", "NVDA", "META")
    )
    prices = {item.symbol: Decimal("100") for item in positions}
    engine = _engine()
    sixth = engine.evaluate_context(
        _context(
            engine,
            quantity="1",
            positions=positions,
            prices=prices,
            feature_snapshot=_feature(0.01),
        )
    )
    assert sixth.status is RiskDecisionStatus.REJECT
    assert RiskReasonCode.MAX_POSITIONS.value in _codes(sixth)

    increase = engine.evaluate_context(
        _context(
            engine,
            symbol="MSFT",
            quantity="1",
            positions=positions,
            prices=prices,
            feature_snapshot=None,
        )
    )
    assert increase.status is RiskDecisionStatus.APPROVE
    assert RiskReasonCode.MAX_POSITIONS.value not in _codes(increase)


def test_fifth_position_is_allowed_when_all_other_limits_have_capacity() -> None:
    positions = tuple(
        Position(symbol, Decimal("1"), Decimal("100"))
        for symbol in ("QQQ", "AMZN", "META", "AIR.PA")
    )
    engine = _engine()
    decision = engine.evaluate_context(
        _context(
            engine,
            quantity="1",
            positions=positions,
            prices={item.symbol: Decimal("100") for item in positions},
            feature_snapshot=_feature(0.01),
        )
    )
    assert decision.status is RiskDecisionStatus.APPROVE
    assert RiskReasonCode.MAX_POSITIONS.value not in decision.reason_codes


def test_cash_limit_reduces_and_zero_cash_rejects_without_leverage() -> None:
    engine = _engine()
    reduced = engine.evaluate_context(
        _context(
            engine,
            quantity="10",
            cash="500",
            equity="10000",
            feature_snapshot=_feature(0.01),
        )
    )
    assert reduced.status is RiskDecisionStatus.REDUCE
    assert reduced.approved_quantity == Decimal("5")
    assert RiskReasonCode.INSUFFICIENT_CASH.value in _codes(reduced)

    rejected = engine.evaluate_context(
        _context(
            engine,
            quantity="1",
            cash="0",
            equity="10000",
            feature_snapshot=_feature(0.01),
        )
    )
    assert rejected.status is RiskDecisionStatus.REJECT
    assert rejected.approved_quantity == ZERO


@pytest.mark.parametrize(
    ("requested", "expected_status", "expected_quantity"),
    [
        ("5", RiskDecisionStatus.APPROVE, Decimal("5")),
        ("10", RiskDecisionStatus.APPROVE, Decimal("10")),
        ("15", RiskDecisionStatus.REDUCE, Decimal("10")),
    ],
)
def test_sell_can_only_reduce_or_close_a_long_position(
    requested: str,
    expected_status: RiskDecisionStatus,
    expected_quantity: Decimal,
) -> None:
    engine = _engine()
    position = Position("AAPL", Decimal("10"), Decimal("100"))
    decision = engine.evaluate_context(
        _context(
            engine,
            quantity=requested,
            side=OrderSide.SELL,
            cash="9000",
            equity="10000",
            positions=(position,),
            feature_snapshot=None,
        )
    )
    assert decision.status is expected_status
    assert decision.approved_quantity == expected_quantity
    assert RiskReasonCode.RISK_REDUCING_ORDER.value in _codes(decision)


def test_sell_without_position_is_rejected_as_short() -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(engine, side=OrderSide.SELL, feature_snapshot=None)
    )
    assert decision.status is RiskDecisionStatus.REJECT
    assert RiskReasonCode.SHORT_NOT_ALLOWED.value in _codes(decision)


def test_halted_engine_blocks_new_and_increased_risk_but_allows_exit() -> None:
    engine = _engine()
    position = Position("AAPL", Decimal("10"), Decimal("100"))
    engine.reset(NOW, Decimal("10000"))
    engine.halt(CircuitBreakerReason.MANUAL_HALT, NOW, Decimal("10000"))

    new_buy = engine.evaluate_context(
        _context(
            engine,
            symbol="MSFT",
            quantity="1",
            cash="9000",
            equity="10000",
            positions=(position,),
            prices={"AAPL": Decimal("100")},
            reset=False,
        )
    )
    increase = engine.evaluate_context(
        _context(
            engine,
            quantity="1",
            cash="9000",
            equity="10000",
            positions=(position,),
            reset=False,
        )
    )
    exit_decision = engine.evaluate_context(
        _context(
            engine,
            quantity="10",
            side=OrderSide.SELL,
            cash="9000",
            equity="10000",
            positions=(position,),
            reset=False,
        )
    )

    assert new_buy.status is RiskDecisionStatus.REJECT
    assert increase.status is RiskDecisionStatus.REJECT
    assert exit_decision.status is RiskDecisionStatus.APPROVE


def test_explicit_trade_risk_distance_caps_quantity() -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(
            engine,
            quantity="300",
            price="10",
            risk_distance="5",
            feature_snapshot=_feature(0.01),
        )
    )
    assert decision.status is RiskDecisionStatus.REDUCE
    assert decision.approved_quantity == Decimal("200")
    assert RiskReasonCode.TRADE_RISK_LIMIT.value in _codes(decision)
    assert RiskReasonCode.NO_EXPLICIT_RISK_DISTANCE.value not in _codes(decision)


def test_invalidation_price_uses_same_explicit_trade_risk_budget() -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(
            engine,
            quantity="300",
            price="10",
            invalidation_price="5",
            feature_snapshot=_feature(0.01),
        )
    )
    assert decision.approved_quantity == Decimal("200")


def test_missing_invalidation_never_fabricates_a_stop_or_false_loss_claim() -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(engine, feature_snapshot=_feature(0.01))
    )
    assert decision.status is RiskDecisionStatus.APPROVE
    assert RiskReasonCode.NO_EXPLICIT_RISK_DISTANCE.value in _codes(decision)
    assert RiskReasonCode.TRADE_RISK_LIMIT.value not in _codes(decision)


def test_unknown_concentration_group_rejects_by_default_or_warns_explicitly() -> None:
    groups_without_aapl = replace(
        BASE_GROUPS,
        groups=tuple(
            (
                name,
                tuple(symbol for symbol in symbols if symbol != "AAPL"),
            )
            for name, symbols in BASE_GROUPS.groups
        ),
    )
    rejecting_engine = _engine(groups=groups_without_aapl)
    rejected = rejecting_engine.evaluate_context(
        _context(rejecting_engine, feature_snapshot=_feature(0.01))
    )
    assert rejected.status is RiskDecisionStatus.REJECT
    assert RiskReasonCode.UNKNOWN_GROUP.value in _codes(rejected)

    warning_config = replace(
        BASE_CONFIG,
        unknown_group_policy=UnknownRiskPolicy.ALLOW_WITH_WARNING,
    )
    warning_engine = _engine(config=warning_config, groups=groups_without_aapl)
    allowed = warning_engine.evaluate_context(
        _context(warning_engine, feature_snapshot=_feature(0.01))
    )
    assert allowed.status is RiskDecisionStatus.APPROVE
    assert RiskReasonCode.UNKNOWN_GROUP.value in _codes(allowed)


@pytest.mark.parametrize("requested", ["0.000001", "1", "100", "10000"])
def test_risk_engine_never_increases_quantity(requested: str) -> None:
    engine = _engine()
    decision = engine.evaluate_context(
        _context(engine, quantity=requested, feature_snapshot=_feature(0.01))
    )
    assert decision.approved_quantity <= decision.requested_quantity


def test_generic_execution_evaluation_remains_fail_closed() -> None:
    engine = _engine()
    order = OrderRequest(
        order_id="generic",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=NOW,
    )
    portfolio = PortfolioSnapshot(NOW, Decimal("10000"), Decimal("10000"))
    decision = engine.evaluate(
        order,
        portfolio,
        TradingContext(environment="PAPER", profile="balanced"),  # type: ignore[arg-type]
    )
    assert decision.status is RiskDecisionStatus.REJECT
    assert RiskReasonCode.INVALID_RISK_CONTEXT.value in _codes(decision)


def test_invalid_existing_short_position_fails_closed() -> None:
    engine = _engine()
    short = Position("MSFT", Decimal("-1"), Decimal("100"))
    decision = engine.evaluate_context(
        _context(
            engine,
            positions=(short,),
            prices={"MSFT": Decimal("100")},
            feature_snapshot=_feature(0.01),
        )
    )
    assert decision.status is RiskDecisionStatus.REJECT
    assert RiskReasonCode.INVALID_RISK_CONTEXT.value in decision.reason_codes
