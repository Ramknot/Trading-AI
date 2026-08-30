from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_ai.backtesting.models import Fill
from trading_ai.core.config import load_profile
from trading_ai.core.hashing import stable_hash
from trading_ai.core.models import OrderSide
from trading_ai.costs.commission import commission_amount
from trading_ai.costs.config import (
    DEFAULT_COST_DIRECTORY,
    inspect_cost_config,
    load_balanced_cost_config,
    load_tariff_profile,
)
from trading_ai.costs.exceptions import CostConfigurationError
from trading_ai.costs.economics import (
    EconomicGate,
    HistoricalEdgeEstimator,
    HistoricalEdgeObservation,
)
from trading_ai.costs.engine import BalancedTransactionCostEngine
from trading_ai.costs.exchange_fees import exchange_fee_amount
from trading_ai.costs.models import (
    CostComponent,
    CostCoverage,
    CostStatus,
    EconomicDecisionStatus,
    EdgeStatus,
    ExpectedEdgeEstimate,
    PreTradeCostRequest,
    TariffStatus,
)
from trading_ai.costs.taxes import transaction_tax_component
from trading_ai.portfolio.currency import CurrencyConverter


NOW = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)


class FixedTestFxBook(CurrencyConverter):
    def __init__(self, rate: Decimal = Decimal("1.10")) -> None:
        self.rate = rate

    def has_rate(self, from_currency, to_currency, timestamp) -> bool:
        del timestamp
        return (from_currency.upper(), to_currency.upper()) in {
            ("USD", "USD"), ("EUR", "EUR"), ("EUR", "USD")
        }

    def convert(self, amount, from_currency, to_currency, timestamp):
        if not self.has_rate(from_currency, to_currency, timestamp):
            raise ValueError("missing test FX rate")
        return amount if from_currency.upper() == to_currency.upper() else amount * self.rate


def _request(
    *,
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    quantity: str = "10",
    price: str = "100",
    timestamp: datetime = NOW,
    spread: str = "5",
    slippage: str = "2",
) -> PreTradeCostRequest:
    return PreTradeCostRequest(
        timestamp=timestamp,
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        reference_price=Decimal(price),
        timeframe="1d",
        spread_bps=Decimal(spread),
        slippage_bps=Decimal(slippage),
        order_id="order-1",
        signal_id="signal-1",
    )


def _edge(amount_bps: str, *, timestamp: datetime = NOW) -> ExpectedEdgeEstimate:
    digest = stable_hash((amount_bps, timestamp))
    return ExpectedEdgeEstimate(
        edge_id=f"edge-{digest[:24]}",
        timestamp=timestamp,
        strategy_name="trend",
        timeframe="1d",
        status=EdgeStatus.AVAILABLE,
        expected_gross_edge_bps=Decimal(amount_bps),
        horizon_bars=5,
        sample_count=40,
        validation_start=timestamp - timedelta(days=100),
        validation_end=timestamp - timedelta(days=1),
        source="deterministic test validation sample",
        provenance_hash=digest,
    )


def test_balanced_cost_configuration_is_deterministic_and_aggressive_stays_disabled() -> None:
    profile = load_profile("balanced")
    first = load_balanced_cost_config(profile)
    second = load_balanced_cost_config(profile)
    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64
    assert first.config.enabled is True
    assert first.tariff.status is TariffStatus.VERIFIED
    assert inspect_cost_config("aggressive").enabled is False
    with pytest.raises(Exception, match="aggressive.*locked"):
        load_balanced_cost_config(load_profile("aggressive"))


def test_fixed_and_tiered_commissions_apply_minimum_cap_and_volume_tiers() -> None:
    directory = load_balanced_cost_config(load_profile("balanced")).config
    del directory  # configuration loading above also exercises referenced hashes
    fixed = load_tariff_profile("ibkr_pro_fixed", DEFAULT_COST_DIRECTORY)
    tiered = load_tariff_profile("ibkr_pro_tiered", DEFAULT_COST_DIRECTORY)
    assert commission_amount(fixed, quantity=Decimal("10"), notional=Decimal("1000")) == Decimal("1.00")
    # The 1% notional cap remains authoritative even when the minimum is larger.
    assert commission_amount(fixed, quantity=Decimal("1"), notional=Decimal("10")) == Decimal("0.10")
    assert commission_amount(tiered, quantity=Decimal("1000"), notional=Decimal("100000")) == Decimal("3.5000")
    assert commission_amount(
        tiered,
        quantity=Decimal("1000"),
        notional=Decimal("100000"),
        monthly_volume_before=Decimal("3000000"),
    ) == Decimal("1.5000")
    assert commission_amount(
        tiered,
        quantity=Decimal("200"),
        notional=Decimal("20000"),
        monthly_volume_before=Decimal("299900"),
    ) == Decimal("0.5500")

    proportional = replace(
        fixed,
        fixed_per_order=Decimal("2"),
        per_unit=Decimal("0.10"),
        proportional_bps=Decimal("10"),
        minimum_per_order=Decimal("0.50"),
        maximum_per_order=Decimal("3"),
        maximum_notional_fraction=None,
    )
    assert commission_amount(
        proportional, quantity=Decimal("10"), notional=Decimal("1000")
    ) == Decimal("3")


def test_missing_tariff_fields_never_default_to_zero(tmp_path) -> None:
    broker_directory = tmp_path / "brokers"
    broker_directory.mkdir()
    source = (DEFAULT_COST_DIRECTORY / "brokers" / "ibkr_pro_fixed.toml").read_text(
        encoding="utf-8"
    )
    source = source.replace('per_unit = "0.005"\n', "", 1)
    (broker_directory / "incomplete.toml").write_text(
        source.replace('profile_id = "ibkr_pro_fixed"', 'profile_id = "incomplete"'),
        encoding="utf-8",
    )

    with pytest.raises(CostConfigurationError, match="per_unit"):
        load_tariff_profile("incomplete", tmp_path)


@pytest.mark.parametrize("status", (TariffStatus.UNVERIFIED, TariffStatus.EXPIRED))
def test_unverified_or_expired_tariff_fails_closed_without_retrospective_policy(
    status,
) -> None:
    bundle = load_balanced_cost_config(load_profile("balanced"))
    strict_config = replace(bundle.config, allow_retrospective_tariff=False)
    strict_bundle = replace(
        bundle,
        config=strict_config,
        tariff=replace(bundle.tariff, status=status),
        config_hash=stable_hash((bundle.config_hash, strict_config, status)),
    )
    estimate = BalancedTransactionCostEngine(strict_bundle).estimate(_request())

    assert estimate.entry_costs.commission.status is CostStatus.UNAVAILABLE
    assert estimate.entry_costs.coverage is CostCoverage.INCOMPLETE


def test_retrospective_tariff_is_disclosed_and_never_period_verified() -> None:
    historical = NOW.replace(year=2024)
    estimate = BalancedTransactionCostEngine.from_profile(
        load_profile("balanced")
    ).estimate(_request(timestamp=historical))

    assert estimate.entry_costs.commission.status is CostStatus.ESTIMATED
    assert estimate.tariff_period_covered is False
    assert "CURRENT_TARIFF_APPLIED_RETROSPECTIVELY" in estimate.warnings


def test_unavailable_cost_can_never_masquerade_as_zero() -> None:
    component = CostComponent.unavailable("tax", "USD", "fixture", "unknown")
    assert component.amount is None
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        CostComponent("tax", CostStatus.UNAVAILABLE, Decimal("0"), "USD", "fixture")


def test_pretrade_estimate_breakdown_round_trip_and_cash_buffer() -> None:
    engine = BalancedTransactionCostEngine.from_profile(load_profile("balanced"))
    estimate = engine.estimate(_request())
    assert estimate.entry_costs.coverage is CostCoverage.COMPLETE
    assert estimate.entry_costs.commission.amount == Decimal("1.00000000")
    assert estimate.entry_costs.spread.amount == Decimal("0.50000000")
    assert estimate.entry_costs.slippage.amount == Decimal("0.20000000")
    assert estimate.entry_costs.total_variable_cost.amount == Decimal("1.70000000")
    assert estimate.round_trip_costs.total_variable_cost.amount == Decimal("3.40000000")
    assert estimate.cash_requirement.cost_buffer == Decimal("0.50000000")
    assert estimate.cash_requirement.total_cash_required == Decimal("1002.20000000")
    assert estimate.tariff_period_covered is True


def test_unknown_instrument_and_missing_fx_fail_cost_coverage_closed() -> None:
    engine = BalancedTransactionCostEngine.from_profile(load_profile("balanced"))
    unknown = engine.estimate(_request(symbol="NOT-CONFIGURED"))
    european = engine.estimate(_request(symbol="MC.PA"))
    assert unknown.entry_costs.coverage is CostCoverage.INCOMPLETE
    assert unknown.cash_requirement.total_cash_required is None
    assert european.entry_costs.fx_cost.status is CostStatus.UNAVAILABLE
    assert european.entry_costs.transaction_tax.status is CostStatus.UNAVAILABLE
    assert european.cash_requirement.coverage is CostCoverage.INCOMPLETE


def test_tax_rule_is_explicit_by_side_date_and_verified_instrument_metadata() -> None:
    bundle = load_balanced_cost_config(load_profile("balanced"))
    rule = bundle.tax_rule("france_ftt_2026")
    assert rule is not None
    verified = replace(
        bundle.instrument_for("MC.PA"), metadata_status=TariffStatus.VERIFIED
    )
    buy = transaction_tax_component(
        metadata=verified,
        rule=rule,
        side=OrderSide.BUY,
        notional=Decimal("1000"),
        timestamp=NOW,
        allow_retrospective=False,
    )
    sell = transaction_tax_component(
        metadata=verified,
        rule=rule,
        side=OrderSide.SELL,
        notional=Decimal("1000"),
        timestamp=NOW,
        allow_retrospective=False,
    )
    assert buy.status is CostStatus.KNOWN and buy.amount == Decimal("4")
    assert sell.status is CostStatus.NOT_APPLICABLE and sell.amount == Decimal("0")

    outside = transaction_tax_component(
        metadata=verified,
        rule=rule,
        side=OrderSide.BUY,
        notional=Decimal("1000"),
        timestamp=NOW.replace(year=2025),
        allow_retrospective=False,
    )
    retrospective = transaction_tax_component(
        metadata=verified,
        rule=rule,
        side=OrderSide.BUY,
        notional=Decimal("1000"),
        timestamp=NOW.replace(year=2025),
        allow_retrospective=True,
    )
    assert outside.status is CostStatus.UNAVAILABLE and outside.amount is None
    assert retrospective.status is CostStatus.ESTIMATED


def test_tiered_exchange_fees_and_operating_costs_are_explicit() -> None:
    engine = BalancedTransactionCostEngine.from_profile(
        load_profile("balanced"), tariff_profile="ibkr_pro_tiered"
    )
    estimate = engine.estimate(_request())
    operating = engine.operating_costs(NOW, NOW + timedelta(days=30))

    assert estimate.entry_costs.exchange_fees.status is CostStatus.UNAVAILABLE
    assert estimate.entry_costs.exchange_fees.amount is None
    assert exchange_fee_amount(
        engine.tariff, quantity=Decimal("10"), notional=Decimal("1000")
    ) == Decimal("0.00200")
    assert operating.market_data_subscription.status is CostStatus.UNAVAILABLE
    assert operating.server_vps.status is CostStatus.UNAVAILABLE
    assert operating.total_operating_cost.amount is None


def test_actual_cost_and_reconciliation_preserve_the_immutable_estimate() -> None:
    engine = BalancedTransactionCostEngine.from_profile(load_profile("balanced"))
    estimate = engine.estimate(_request())
    fill = Fill(
        fill_id="fill-1",
        order_id="order-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        reference_price=Decimal("100"),
        price=Decimal("101"),
        timestamp=NOW + timedelta(days=1),
        commission=Decimal("0"),
        slippage_cost=Decimal("0.2"),
        spread_cost=Decimal("0.5"),
    )
    actual = engine.actualize(fill, estimate)
    reconciliation = engine.reconcile(estimate, actual)
    assert estimate.reference_price == Decimal("100")
    assert actual.execution_price == Decimal("101")
    assert actual.breakdown.commission.amount == Decimal("1.00000000")
    assert reconciliation.estimate_id == estimate.estimate_id
    assert reconciliation.coverage is CostCoverage.COMPLETE


def test_economic_gate_pass_block_incomplete_and_exit_passthrough() -> None:
    bundle = load_balanced_cost_config(load_profile("balanced"))
    engine = BalancedTransactionCostEngine(bundle)
    gate = EconomicGate(bundle.config, bundle.config_hash)
    estimate = engine.estimate(_request())
    passed = gate.evaluate(
        estimate=estimate,
        edge=_edge("100"),
        signal_id="signal-1",
        is_risk_reducing_exit=False,
    )
    blocked = gate.evaluate(
        estimate=estimate,
        edge=_edge("20"),
        signal_id="signal-1",
        is_risk_reducing_exit=False,
    )
    unavailable = gate.evaluate(
        estimate=estimate,
        edge=ExpectedEdgeEstimate.unavailable(
            timestamp=NOW, strategy_name="trend", timeframe="1d", reason="fixture"
        ),
        signal_id="signal-1",
        is_risk_reducing_exit=False,
    )
    exit_decision = gate.evaluate(
        estimate=engine.estimate(_request(side=OrderSide.SELL)),
        edge=ExpectedEdgeEstimate.unavailable(
            timestamp=NOW, strategy_name="trend", timeframe="1d", reason="fixture"
        ),
        signal_id="signal-exit",
        is_risk_reducing_exit=True,
    )
    assert passed.status is EconomicDecisionStatus.PASS and passed.allows_new_risk
    assert blocked.status is EconomicDecisionStatus.BLOCK and not blocked.allows_new_risk
    assert unavailable.status is EconomicDecisionStatus.INCOMPLETE
    assert exit_decision.status is EconomicDecisionStatus.NOT_APPLICABLE
    assert exit_decision.allows_new_risk is True


def test_historical_edge_estimator_uses_only_prior_train_or_validation() -> None:
    estimator = HistoricalEdgeEstimator(minimum_samples=2)
    observations = (
        HistoricalEdgeObservation(NOW - timedelta(days=2), "trend", "1d", Decimal("20"), "TRAIN"),
        HistoricalEdgeObservation(NOW - timedelta(days=1), "trend", "1d", Decimal("40"), "VALIDATION"),
        HistoricalEdgeObservation(NOW + timedelta(days=1), "trend", "1d", Decimal("999"), "VALIDATION"),
    )
    estimate = estimator.estimate(
        observations,
        as_of=NOW,
        strategy_name="trend",
        timeframe="1d",
        horizon_bars=5,
    )
    assert estimate.expected_gross_edge_bps == Decimal("30")
    assert estimate.sample_count == 2
    with pytest.raises(ValueError, match="FINAL TEST"):
        HistoricalEdgeObservation(NOW, "trend", "1d", Decimal("1"), "TEST")


def test_fixed_fx_book_does_not_turn_unverified_european_tariff_into_complete_costs() -> None:
    engine = BalancedTransactionCostEngine.from_profile(
        load_profile("balanced"), currency_converter=FixedTestFxBook()
    )
    estimate = engine.estimate(_request(symbol="MC.PA"))
    assert estimate.entry_costs.fx_cost.status is CostStatus.ESTIMATED
    assert estimate.entry_costs.fx_cost.amount == Decimal("0.22000000")
    assert estimate.entry_costs.commission.status is CostStatus.UNAVAILABLE
    assert estimate.entry_costs.coverage is CostCoverage.INCOMPLETE
