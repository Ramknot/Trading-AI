"""Research validation gate with OOS, costs, stress, and robustness checks."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from trading_ai.core.hashing import stable_hash
from trading_ai.core.models import BacktestResult
from trading_ai.costs.models import CostCoverage, CostStatus, TariffStatus
from trading_ai.validation.config import ValidationConfig, load_validation_config
from trading_ai.validation.exceptions import ValidationError
from trading_ai.validation.models import (
    CostStressResult,
    CriterionStatus,
    SubperiodResult,
    SymbolRobustnessResult,
    ValidationCriterion,
    ValidationReport,
    ValidationStatus,
)


ZERO = Decimal("0")


def _finite_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


class ResearchValidationGate:
    gate_name = "balanced-research-validation"
    gate_version = "1.0"

    def __init__(
        self, config: ValidationConfig | None = None, config_hash: str | None = None
    ) -> None:
        if config is None or config_hash is None:
            config, config_hash = load_validation_config()
        self.config = config
        self.config_hash = config_hash

    def evaluate(
        self,
        result: BacktestResult,
        *,
        datasets_integrity: bool,
        data_quality_acceptable: bool,
        final_oos: bool,
        training_or_edge_overlap: bool,
        tariff_period_verified: bool,
        real_data_available: bool,
        synthetic_mechanics_only: bool = False,
    ) -> ValidationReport:
        criteria: list[ValidationCriterion] = []
        def add(name: str, passed: bool, observed, required, reason: str) -> None:
            criteria.append(ValidationCriterion(
                name,
                CriterionStatus.PASS if passed else CriterionStatus.FAIL,
                str(observed), str(required), reason,
            ))
        add("dataset_integrity", datasets_integrity, datasets_integrity, True, "All dataset checksums must verify.")
        add("data_quality", data_quality_acceptable, data_quality_acceptable, True, "DataQuality FAIL is never accepted.")
        oos_valid = final_oos and not training_or_edge_overlap
        add(
            "final_oos",
            oos_valid or not self.config.require_final_oos,
            oos_valid,
            self.config.require_final_oos,
            "Final OOS must not overlap ML or edge calibration.",
        )
        summary = result.cost_summary
        cost_complete = summary is not None and summary.cost_coverage is CostCoverage.COMPLETE
        add(
            "variable_cost_coverage",
            cost_complete or not self.config.require_complete_variable_costs,
            getattr(summary, "cost_coverage", "UNAVAILABLE"),
            "COMPLETE" if self.config.require_complete_variable_costs else "optional",
            "Every critical variable cost must be explicit.",
        )
        add(
            "tariff_period",
            tariff_period_verified
            or not self.config.require_verified_tariff_for_period,
            tariff_period_verified,
            self.config.require_verified_tariff_for_period,
            "Tariff must be VERIFIED for the tested period.",
        )
        trades = result.metrics.number_of_trades
        add("closed_trades", trades >= self.config.minimum_closed_trades, trades, self.config.minimum_closed_trades, "Minimum sample size is fixed before TEST review.")
        net_return = _finite_decimal(
            summary.net_return_before_operating if summary is not None else None
        )
        add("net_return", net_return is not None and net_return > self.config.minimum_net_return, net_return if net_return is not None else "UNAVAILABLE", f"> {self.config.minimum_net_return}", "Cost-aware net return threshold.")
        expectancy = _finite_decimal(result.metrics.expectancy)
        add("net_expectancy", expectancy is not None and expectancy > self.config.minimum_net_expectancy, expectancy if expectancy is not None else "UNAVAILABLE", f"> {self.config.minimum_net_expectancy}", "Average net trade outcome must clear the fixed threshold.")
        profit_factor = _finite_decimal(result.metrics.profit_factor)
        add("profit_factor", profit_factor is not None and profit_factor > self.config.minimum_profit_factor, profit_factor if profit_factor is not None else "UNAVAILABLE", f"> {self.config.minimum_profit_factor}", "Net profit factor threshold.")
        max_drawdown = _finite_decimal(result.metrics.max_drawdown_pct)
        drawdown = abs(max_drawdown) if max_drawdown is not None else None
        add("max_drawdown", drawdown is not None and drawdown <= self.config.maximum_drawdown, drawdown if drawdown is not None else "UNAVAILABLE", f"<= {self.config.maximum_drawdown}", "Drawdown must stay within the predeclared Balanced limit.")
        cash_non_negative = bool(result.equity_curve) and all(
            _finite_decimal(getattr(point, "cash", None)) is not None
            and _finite_decimal(point.cash) >= ZERO
            for point in result.equity_curve
        )
        add("cash_non_negative", cash_non_negative, cash_non_negative, True, "No leverage or negative cash is permitted.")
        breaker_coherent = all(item.new_state.value in {"NORMAL", "REDUCED", "HALTED"} for item in result.risk_state_transitions)
        add("circuit_breaker", breaker_coherent, breaker_coherent, True, "Risk state transitions must remain valid.")

        stresses = self._stress(result)
        subperiods = self._subperiods(result)
        symbols = self._symbols(result)
        warnings = ["SURVIVORSHIP_BIAS_NOT_RESOLVED"]
        operating_complete = (
            result.operating_costs is not None
            and result.operating_costs.total_operating_cost.status is not CostStatus.UNAVAILABLE
        )
        if not operating_complete:
            warnings.append("OPERATING_COSTS_INCOMPLETE")
            criteria.append(ValidationCriterion(
                "operating_costs", CriterionStatus.WARNING, "UNAVAILABLE", "COMPLETE for deployment economics",
                "Missing period-level operating costs prevent complete deployment profitability.",
            ))
        if any(item.share_of_positive_pnl is not None and item.share_of_positive_pnl > float(self.config.symbol_concentration_warning_fraction) for item in symbols):
            warnings.append("SYMBOL_RESULT_CONCENTRATION")
        if subperiods and sum(item.net_return > 0 for item in subperiods) <= 1:
            warnings.append("TEMPORAL_RESULT_CONCENTRATION")
        hard_fail = any(item.status is CriterionStatus.FAIL for item in criteria)
        if not real_data_available:
            status = ValidationStatus.BLOCKED_EXTERNAL_DATA
        elif hard_fail:
            status = ValidationStatus.FAIL
        elif not operating_complete and self.config.require_operating_costs_for_pass:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASS
        campaign = status if real_data_available else ValidationStatus.BLOCKED_EXTERNAL_DATA
        payload = (
            result.run_id, self.config_hash, criteria, stresses, subperiods, symbols,
            real_data_available, synthetic_mechanics_only,
        )
        digest = stable_hash(payload)
        return ValidationReport(
            validation_id=f"validation-{digest[:24]}",
            run_id=result.run_id,
            created_at=datetime.now(timezone.utc),
            gate_name=self.gate_name,
            gate_version=self.gate_version,
            config_hash=self.config_hash,
            status=status,
            implementation_status="DONE",
            real_data_campaign_status=campaign,
            synthetic_mechanics_only=synthetic_mechanics_only,
            final_oos=final_oos,
            dataset_ids=tuple(item.dataset_id for item in result.dataset_references),
            dataset_checksums=tuple(item.checksum_sha256 for item in result.dataset_references),
            tariff_profile_id=summary.tariff_profile_id if summary is not None else None,
            tariff_status=summary.tariff_status.value if summary is not None else None,
            tariff_period_verified=tariff_period_verified,
            cost_coverage=summary.cost_coverage.value if summary is not None else "UNAVAILABLE",
            criteria=tuple(criteria),
            stress_results=stresses,
            subperiods=subperiods,
            symbols=symbols,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def evaluate_export(
        self,
        *,
        summary: dict[str, Any],
        tables: dict[str, tuple[dict[str, Any], ...]],
        integrity_verified: bool,
        final_oos: bool,
        no_training_or_edge_overlap_confirmed: bool,
        real_data_available: bool,
        synthetic_mechanics_only: bool = False,
    ) -> ValidationReport:
        """Validate a checksum-verified schema 1.6 export without re-running engines."""

        metrics_raw = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        costs_root = summary.get("costs") if isinstance(summary.get("costs"), dict) else {}
        costs_raw = costs_root.get("summary") if isinstance(costs_root.get("summary"), dict) else None
        cost_summary = None
        if costs_raw is not None:
            cost_summary = SimpleNamespace(
                cost_coverage=CostCoverage(str(costs_raw.get("cost_coverage"))),
                net_return_before_operating=_finite_decimal(
                    costs_raw.get("net_return_before_operating")
                ),
                total_variable_cost=_finite_decimal(costs_raw.get("total_variable_cost")),
                tariff_profile_id=str(costs_raw.get("tariff_profile_id")),
                tariff_status=TariffStatus(str(costs_raw.get("tariff_status"))),
            )
        operating_raw = costs_root.get("operating") if isinstance(costs_root.get("operating"), dict) else None
        operating_costs = None
        if operating_raw is not None and isinstance(operating_raw.get("total_operating_cost"), dict):
            operating_costs = SimpleNamespace(
                total_operating_cost=SimpleNamespace(
                    status=CostStatus(str(operating_raw["total_operating_cost"].get("status")))
                )
            )
        equity_items = []
        for item in tables.get("equity", ()):
            cash = _finite_decimal(item.get("cash"))
            equity_value = _finite_decimal(item.get("equity"))
            if cash is None or equity_value is None or not item.get("timestamp"):
                raise ValidationError(
                    "validation export contains an incomplete equity observation"
                )
            equity_items.append(SimpleNamespace(
                timestamp=datetime.fromisoformat(str(item["timestamp"]).replace("Z", "+00:00")),
                cash=cash,
                equity=equity_value,
            ))
        equity = tuple(equity_items)
        trade_items = []
        for item in tables.get("trades", ()):
            net_pnl = _finite_decimal(item.get("net_pnl"))
            if net_pnl is None or not item.get("exit_time") or not item.get("symbol"):
                raise ValidationError(
                    "validation export contains an incomplete closed trade"
                )
            trade_items.append(SimpleNamespace(
                symbol=str(item["symbol"]),
                exit_time=datetime.fromisoformat(str(item["exit_time"]).replace("Z", "+00:00")),
                net_pnl=net_pnl,
            ))
        trades = tuple(trade_items)
        initial_cash = _finite_decimal(summary.get("initial_cash"))
        final_equity = _finite_decimal(summary.get("final_equity"))
        if initial_cash is None or initial_cash <= ZERO or final_equity is None:
            raise ValidationError(
                "validation export requires finite initial cash and final equity"
            )
        result = SimpleNamespace(
            run_id=str(summary.get("run_id")),
            cost_summary=cost_summary,
            operating_costs=operating_costs,
            metrics=SimpleNamespace(
                number_of_trades=(
                    int(metrics_raw["number_of_trades"])
                    if metrics_raw.get("number_of_trades") is not None else -1
                ),
                expectancy=_finite_decimal(metrics_raw.get("expectancy")),
                profit_factor=metrics_raw.get("profit_factor"),
                max_drawdown_pct=_finite_decimal(metrics_raw.get("max_drawdown_pct")),
            ),
            initial_cash=initial_cash,
            final_equity=final_equity,
            equity_curve=equity,
            trades=trades,
            risk_state_transitions=tuple(
                SimpleNamespace(new_state=SimpleNamespace(value=str(item.get("new_state"))))
                for item in tables.get("risk_states", ())
            ),
            dataset_references=tuple(
                SimpleNamespace(
                    dataset_id=str(item.get("dataset_id")),
                    checksum_sha256=str(item.get("checksum_sha256")),
                )
                for item in summary.get("dataset_references", ())
            ),
        )
        quality = summary.get("data_quality_reports") or []
        quality_acceptable = bool(quality) and all(
            str(item.get("quality_status")) in {"PASS", "WARNING"}
            for item in quality if isinstance(item, dict)
        )
        estimates = tables.get("cost_estimates", ())
        tariff_period_verified = bool(estimates) and all(
            item.get("tariff_period_covered") is True
            and item.get("tariff_status") == "VERIFIED"
            for item in estimates
        )
        references = tuple(
            item
            for item in summary.get("dataset_references", ())
            if isinstance(item, dict)
        )
        non_real_providers = {"memory", "synthetic", "fake", "fixture"}
        export_is_real = bool(references) and all(
            str(item.get("provider", "")).strip().lower()
            not in non_real_providers
            and bool(str(item.get("provider", "")).strip())
            for item in references
        )
        return self.evaluate(
            result,
            datasets_integrity=integrity_verified,
            data_quality_acceptable=quality_acceptable,
            final_oos=final_oos,
            training_or_edge_overlap=not no_training_or_edge_overlap_confirmed,
            tariff_period_verified=tariff_period_verified,
            real_data_available=real_data_available and export_is_real,
            synthetic_mechanics_only=synthetic_mechanics_only,
        )

    def _stress(self, result: BacktestResult) -> tuple[CostStressResult, ...]:
        summary = result.cost_summary
        if summary is None or summary.total_variable_cost is None:
            return tuple(
                CostStressResult(item, None, None, None, CriterionStatus.UNAVAILABLE)
                for item in self.config.cost_stress_multipliers
            )
        base_net = result.final_equity - result.initial_cash
        values = []
        for multiplier in self.config.cost_stress_multipliers:
            stressed_cost = summary.total_variable_cost * multiplier
            stressed_net = base_net - summary.total_variable_cost * (multiplier - Decimal("1"))
            values.append(CostStressResult(
                multiplier, stressed_cost, stressed_net,
                float(stressed_net / result.initial_cash),
                CriterionStatus.PASS if stressed_net > ZERO else CriterionStatus.WARNING,
            ))
        return tuple(values)

    def _subperiods(self, result: BacktestResult) -> tuple[SubperiodResult, ...]:
        curve = result.equity_curve
        count = min(self.config.minimum_subperiods, max(1, len(curve) - 1))
        if count < 2:
            return ()
        results = []
        for index in range(count):
            start_index = index * (len(curve) - 1) // count
            end_index = (index + 1) * (len(curve) - 1) // count
            start, end = curve[start_index], curve[end_index]
            closed = sum(start.timestamp < trade.exit_time <= end.timestamp for trade in result.trades)
            results.append(SubperiodResult(
                index + 1, start.timestamp, end.timestamp,
                float(end.equity / start.equity - Decimal("1")) if start.equity > ZERO else 0.0,
                closed,
            ))
        return tuple(results)

    @staticmethod
    def _symbols(result: BacktestResult) -> tuple[SymbolRobustnessResult, ...]:
        positive_total = sum((trade.net_pnl for trade in result.trades if trade.net_pnl > ZERO), ZERO)
        values = []
        for symbol in sorted({trade.symbol for trade in result.trades}):
            trades = tuple(item for item in result.trades if item.symbol == symbol)
            pnl = sum((item.net_pnl for item in trades), ZERO)
            share = float(max(pnl, ZERO) / positive_total) if positive_total > ZERO else None
            values.append(SymbolRobustnessResult(symbol, len(trades), pnl, share))
        return tuple(values)
