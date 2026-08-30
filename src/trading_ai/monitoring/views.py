"""Deterministic read models derived only from verified engine outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from trading_ai.backtesting.reproducibility import stable_hash, to_primitive
from trading_ai.monitoring.costs import build_cost_snapshot
from trading_ai.monitoring.models import (
    DecisionTrace,
    DecisionTraceStep,
    HealthComponent,
    HealthSnapshot,
    MonitoringSnapshot,
    SystemStatus,
)
from trading_ai.monitoring.source import BacktestMonitoringData


_TRACE_STAGES = (
    "Dataset", "Feature", "Regime", "Strategy", "Signal", "ML",
    "Activation", "Portfolio", "Cost Estimate", "Economic Gate", "Risk",
    "Order", "Fill", "Actual Cost", "Cost Reconciliation",
)


def _unavailable(reason: str = "not present in this export schema") -> dict[str, Any]:
    return {"status": "UNAVAILABLE", "value": None, "reason": reason}


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("monitoring source timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _iso_key(row: dict[str, Any], field: str = "timestamp") -> tuple[str, str]:
    return (str(row.get(field) or ""), str(row.get("symbol") or row.get("order_id") or ""))


def _latest_by(
    rows: Iterable[dict[str, Any]], key: str, timestamp: str = "timestamp"
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: _iso_key(item, timestamp)):
        value = row.get(key)
        if value:
            latest[str(value)] = row
    return latest


def _drawdown_curve(equity: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    peak: Decimal | None = None
    points: list[dict[str, Any]] = []
    for row in equity:
        value = _decimal(row.get("equity"))
        if value is None:
            continue
        peak = value if peak is None or value > peak else peak
        drawdown = Decimal("0") if peak == 0 else (value / peak) - Decimal("1")
        points.append({"timestamp": row.get("timestamp"), "drawdown": str(drawdown)})
    return points


class BacktestViewBuilder:
    """Project verified export facts into dashboard views without engine recomputation."""

    def build(
        self, data: BacktestMonitoringData, *, monitoring_store_healthy: bool
    ) -> MonitoringSnapshot:
        summary = data.summary
        timestamp = _utc(summary.get("completed_at") or summary.get("started_at"))
        tables = data.tables
        costs = build_cost_snapshot(summary, timestamp)
        portfolio = self._portfolio(summary, tables)
        strategies = self._strategies(summary, tables)
        regimes = self._regimes(summary, tables)
        ml = self._ml(summary, tables)
        risk = self._risk(summary, tables)
        validation = (
            summary.get("validation")
            if isinstance(summary.get("validation"), dict)
            else {"status": "UNAVAILABLE"}
        )
        data_quality = self._data_quality(summary)
        decisions = self._decisions(tables)
        traces = [to_primitive(item) for item in self.build_traces(data)]
        health = self._health(
            summary,
            timestamp,
            data_quality,
            strategies,
            regimes,
            ml,
            portfolio,
            risk,
            monitoring_store_healthy,
            costs.coverage_status.value,
        )
        equity = list(tables.get("equity", ()))
        last_equity = equity[-1] if equity else {}
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        current_equity = last_equity.get("equity", summary.get("final_equity"))
        initial_equity = summary.get("initial_cash")
        cash = last_equity.get("cash")
        positions_value = _decimal(last_equity.get("positions_value"))
        equity_value = _decimal(current_equity)
        gross_exposure = (
            float(abs(positions_value) / equity_value)
            if positions_value is not None and equity_value not in (None, Decimal("0"))
            else None
        )
        sections = {
            "overview": {
                "mode": "BACKTEST",
                "run_id": data.run_id,
                "schema_version": data.schema_version,
                "result_status": summary.get("status", "UNAVAILABLE"),
                "system_status": health.status.value,
                "integrity": "VERIFIED",
                "initial_equity": initial_equity,
                "current_equity": current_equity,
                "final_equity": summary.get("final_equity"),
                "cash": cash,
                "cash_reserved": _unavailable("pending cash reservation is not exported"),
                "cash_free": _unavailable("cannot be trusted without exported reservations"),
                "gross_exposure": gross_exposure,
                "gross_pnl": str(costs.gross_pnl) if costs.gross_pnl is not None else None,
                "net_pnl_known": str(costs.net_pnl_known) if costs.net_pnl_known is not None else None,
                "max_drawdown": metrics.get("max_drawdown_pct", _unavailable()),
                "risk_state": risk.get("current_state", "UNAVAILABLE"),
                "position_count": len(portfolio["positions"]),
                "active_strategy_count": len(strategies["strategies"]),
                "ml_mode": ml.get("mode", "UNAVAILABLE"),
                "ml_model": ml.get("model_id"),
                "data_quality_status": data_quality["overall_status"],
                "known_trading_costs": str(costs.known_trading_costs),
                "cost_coverage_status": costs.coverage_status.value,
                "net_completeness": (
                    "NET COMPLETE"
                    if costs.coverage_status.value == "COMPLETE"
                    else "NET INCOMPLETE"
                ),
                "cost_warning": costs.warnings[0] if costs.warnings else None,
            },
            "equity": {
                "curve": equity,
                "drawdown_curve": _drawdown_curve(tuple(equity)),
                "metrics": metrics,
            },
            "portfolio": portfolio,
            "strategies": strategies,
            "regimes": regimes,
            "ml": ml,
            "risk": risk,
            "data_quality": data_quality,
            "costs": to_primitive(costs),
            "validation": validation,
            "decisions": decisions,
            "decision_traces": traces,
            "health": to_primitive(health),
        }
        payload = json.dumps(to_primitive(sections), sort_keys=True, separators=(",", ":"), allow_nan=False)
        snapshot_id = "mon-" + stable_hash(
            {"run_id": data.run_id, "source_fingerprint": data.source_fingerprint, "sections": sections}
        )[:24]
        return MonitoringSnapshot(
            snapshot_id=snapshot_id,
            run_id=data.run_id,
            timestamp=timestamp,
            mode="BACKTEST",
            status=health.status,
            source_schema_version=data.schema_version,
            source_fingerprint=data.source_fingerprint,
            sections_json=payload,
        )

    @staticmethod
    def _portfolio(summary: dict[str, Any], tables: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, Any]:
        quantities: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        for row in tables.get("ledger", ()):
            symbol = row.get("symbol")
            change = _decimal(row.get("quantity_change"))
            if symbol and change is not None:
                quantities[str(symbol)] += change
        positions = [
            {"symbol": symbol, "quantity": str(quantity)}
            for symbol, quantity in sorted(quantities.items())
            if quantity != 0
        ]
        latest_targets = _latest_by(tables.get("portfolio_targets", ()), "symbol")
        latest_sleeves = _latest_by(
            tables.get("portfolio_sleeves", ()), "signal_id", "last_updated_at"
        )
        portfolio = summary.get("portfolio")
        portfolio = portfolio if isinstance(portfolio, dict) else {}
        metrics = portfolio.get("metrics") if isinstance(portfolio.get("metrics"), dict) else {}
        decisions = tables.get("portfolio_decisions", ())
        counts = Counter(str(item.get("status", "UNAVAILABLE")) for item in decisions)
        return {
            "status": "AVAILABLE" if portfolio.get("engine_name") not in (None, "unavailable") else "UNAVAILABLE",
            "engine_name": portfolio.get("engine_name", "UNAVAILABLE"),
            "engine_version": portfolio.get("engine_version", "UNAVAILABLE"),
            "config_hash": portfolio.get("config_hash", "UNAVAILABLE"),
            "config": portfolio.get("config", []),
            "metrics": metrics,
            "positions": positions,
            "targets": [latest_targets[key] for key in sorted(latest_targets)],
            "sleeves": sorted(latest_sleeves.values(), key=lambda row: (str(row.get("strategy_name")), str(row.get("symbol")))),
            "decision_counts": dict(sorted(counts.items())),
            "opportunities": len(tables.get("portfolio_opportunities", ())),
            "plans": portfolio.get("plans", []),
        }

    @staticmethod
    def _strategies(summary: dict[str, Any], tables: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, Any]:
        signals = tables.get("signals", ())
        by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in signals:
            by_strategy[str(row.get("strategy_name") or "UNAVAILABLE")].append(row)
        parameters = summary.get("strategy_parameters", [])
        parameter_map = dict(parameters) if isinstance(parameters, list) else {}
        names = set(by_strategy)
        for key in parameter_map:
            if key.startswith("strategy.") and key.count(".") >= 2:
                names.add(key.split(".", 2)[1])
        if not names and summary.get("strategy_name"):
            names.add(str(summary["strategy_name"]))
        records = []
        for name in sorted(names):
            items = by_strategy.get(name, [])
            actions = Counter(str(item.get("action", "UNAVAILABLE")) for item in items)
            prefix = (
                f"strategy.{name}."
                if summary.get("strategy_name") == "multi-strategy-portfolio"
                else ""
            )
            config = {
                key[len(prefix):]: value
                for key, value in parameter_map.items()
                if prefix and key.startswith(prefix)
            }
            if not prefix:
                config = parameter_map
            records.append(
                {
                    "name": name,
                    "version": next((item.get("strategy_version") for item in items), summary.get("strategy_version", "UNAVAILABLE")),
                    "config": config,
                    "signal_count": len(items),
                    "enter_signals": actions.get("ENTER_LONG", 0),
                    "exit_signals": actions.get("EXIT_LONG", 0),
                    "recent_signals": sorted(items, key=_iso_key)[-20:],
                }
            )
        return {"strategies": records, "total_signals": len(signals)}

    @staticmethod
    def _regimes(summary: dict[str, Any], tables: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, Any]:
        regime = summary.get("regime")
        regime = regime if isinstance(regime, dict) else {}
        latest = _latest_by(tables.get("regime_snapshots", ()), "symbol")
        transitions = sorted(tables.get("regime_transitions", ()), key=_iso_key)
        last_transition = _latest_by(transitions, "symbol")
        records = []
        for symbol in sorted(latest):
            item = dict(latest[symbol])
            item["last_transition"] = last_transition.get(symbol)
            records.append(item)
        return {
            "status": "AVAILABLE" if records or regime.get("detector_name") not in (None, "unavailable") else "UNAVAILABLE",
            "detector_name": regime.get("detector_name", "UNAVAILABLE"),
            "detector_version": regime.get("detector_version", "UNAVAILABLE"),
            "config_hash": regime.get("config_hash", "UNAVAILABLE"),
            "policy_name": regime.get("policy_name", "UNAVAILABLE"),
            "policy_version": regime.get("policy_version", "UNAVAILABLE"),
            "latest_by_symbol": records,
            "transitions": transitions[-100:],
            "report": regime.get("report", _unavailable()),
        }

    @staticmethod
    def _ml(summary: dict[str, Any], tables: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, Any]:
        ml = summary.get("ml")
        ml = ml if isinstance(ml, dict) else {}
        decisions = sorted(tables.get("ml_decisions", ()), key=_iso_key)
        counts = Counter(str(item.get("status", "UNAVAILABLE")) for item in decisions)
        return {
            "status": ml.get("status", "AVAILABLE" if ml else "UNAVAILABLE"),
            "mode": ml.get("mode", "UNAVAILABLE"),
            "model_id": ml.get("model_id"),
            "model_family": ml.get("model_family"),
            "model_version": ml.get("model_version"),
            "model_status": ml.get("model_status"),
            "base_feature_schema_version": ml.get("base_feature_schema_version"),
            "ml_feature_schema_version": ml.get("ml_feature_schema_version"),
            "threshold": ml.get("threshold"),
            "decision_counts": dict(sorted(counts.items())),
            "recent_predictions": sorted(tables.get("ml_predictions", ()), key=_iso_key)[-50:],
            "recent_decisions": decisions[-50:],
        }

    @staticmethod
    def _risk(summary: dict[str, Any], tables: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, Any]:
        risk = summary.get("risk")
        risk = risk if isinstance(risk, dict) else {}
        decisions = sorted(tables.get("risk_decisions", ()), key=_iso_key)
        transitions = sorted(tables.get("risk_states", ()), key=_iso_key)
        counts = Counter(str(item.get("status", "UNAVAILABLE")) for item in decisions)
        reasons = Counter(
            str(code) for item in decisions for code in (item.get("reason_codes") or [])
        )
        current_state = (
            transitions[-1].get("new_state")
            if transitions
            else decisions[-1].get("risk_state") if decisions else "UNAVAILABLE"
        )
        return {
            "status": "AVAILABLE" if risk.get("engine_name") else "UNAVAILABLE",
            "engine_name": risk.get("engine_name", "UNAVAILABLE"),
            "engine_version": risk.get("engine_version", "UNAVAILABLE"),
            "config_hash": risk.get("config_hash", "UNAVAILABLE"),
            "config": risk.get("config", []),
            "current_state": current_state or "UNAVAILABLE",
            "summary": risk.get("summary", _unavailable()),
            "decision_counts": dict(sorted(counts.items())),
            "top_reasons": reasons.most_common(10),
            "state_transitions": transitions,
            "latest_decisions": decisions[-100:],
            "deny_all_fail_safe": risk.get("engine_name") == "DenyAllRiskEngine",
        }

    @staticmethod
    def _data_quality(summary: dict[str, Any]) -> dict[str, Any]:
        reports = {
            (str(item.get("symbol")), str(item.get("timeframe"))): item
            for item in (summary.get("data_quality_reports", []) or [])
            if isinstance(item, dict)
        }
        datasets = []
        for item in summary.get("dataset_references", []) or []:
            report = reports.get((str(item.get("symbol")), str(item.get("timeframe"))))
            datasets.append(
                {
                    **item,
                    "row_count": report.get("row_count") if report else _unavailable(),
                    "gaps": report.get("missing_expected_bar_count") if report else _unavailable(),
                    "duplicates": report.get("duplicate_count") if report else _unavailable(),
                    "invalid_bars": report.get("invalid_bar_count") if report else _unavailable(),
                    "quality_status": report.get("quality_status", "UNAVAILABLE") if report else "UNAVAILABLE",
                }
            )
        statuses = {str(item.get("quality_status")) for item in datasets}
        overall = (
            "FAIL" if "FAIL" in statuses
            else "WARNING" if "WARNING" in statuses
            else "PASS" if statuses == {"PASS"}
            else "UNAVAILABLE"
        )
        return {
            "overall_status": overall,
            "datasets": datasets,
            "warnings": list(summary.get("warnings", []) or []),
            "integrity": "VERIFIED",
        }

    @staticmethod
    def _decisions(tables: dict[str, tuple[dict[str, Any], ...]]) -> list[dict[str, Any]]:
        specs = (
            ("Strategy", "signals", "signal_id", "action"),
            ("ML", "ml_decisions", "decision_id", "status"),
            ("Activation", "activation_decisions", "decision_id", "status"),
            ("Portfolio", "portfolio_decisions", "decision_id", "status"),
            ("Costs", "cost_estimates", "estimate_id", None),
            ("Economics", "economic_decisions", "decision_id", "status"),
            ("Risk", "risk_decisions", "decision_id", "status"),
            ("Execution", "orders", "order_id", "status"),
            ("Execution", "fills", "fill_id", None),
            ("Costs", "cost_actuals", "actual_cost_id", None),
            ("Costs", "cost_reconciliation", "reconciliation_id", "coverage"),
        )
        records: list[dict[str, Any]] = []
        for component, table_name, id_name, status_name in specs:
            for row in tables.get(table_name, ()):
                reasons = row.get("reason_codes") or row.get("reason_code") or row.get("reason") or []
                if isinstance(reasons, str):
                    reasons = [reasons]
                records.append(
                    {
                        "component": component,
                        "timestamp": row.get("timestamp") or row.get("created_at"),
                        "entity_id": row.get(id_name),
                        "symbol": row.get("symbol"),
                        "strategy": row.get("strategy_name"),
                        "status": row.get(status_name) if status_name else "FILLED",
                        "reasons": reasons,
                        "payload": row,
                    }
                )
        return sorted(records, key=lambda row: (str(row.get("timestamp") or ""), str(row.get("entity_id") or "")))

    @staticmethod
    def _health(
        summary: dict[str, Any], timestamp: datetime, data_quality: dict[str, Any],
        strategies: dict[str, Any], regimes: dict[str, Any], ml: dict[str, Any],
        portfolio: dict[str, Any], risk: dict[str, Any], store_healthy: bool,
        cost_coverage: str,
    ) -> HealthSnapshot:
        components = [
            HealthComponent("Data", SystemStatus.WARNING if data_quality["overall_status"] == "UNAVAILABLE" else SystemStatus.HEALTHY, "source integrity verified; detailed DataQualityReport unavailable" if data_quality["overall_status"] == "UNAVAILABLE" else "data quality available", timestamp),
            HealthComponent("Execution simulator", SystemStatus.HEALTHY if summary.get("status") == "COMPLETED" else SystemStatus.WARNING, f"backtest status {summary.get('status', 'UNAVAILABLE')}", timestamp),
            HealthComponent("Feature", SystemStatus.HEALTHY if any("feature_schema_version" in str(item) for item in summary.get("strategy_parameters", [])) else SystemStatus.UNAVAILABLE, "feature lineage present" if any("feature_schema_version" in str(item) for item in summary.get("strategy_parameters", [])) else "feature lineage unavailable", timestamp),
            HealthComponent("ML", SystemStatus.UNAVAILABLE if ml.get("mode") in (None, "UNAVAILABLE", "DISABLED") else SystemStatus.HEALTHY, f"mode {ml.get('mode', 'UNAVAILABLE')}", timestamp),
            HealthComponent("Monitoring store", SystemStatus.HEALTHY if store_healthy else SystemStatus.ERROR, "local SQLite store available" if store_healthy else "local SQLite store error", timestamp),
            HealthComponent("Portfolio", SystemStatus.HEALTHY if portfolio.get("status") == "AVAILABLE" else SystemStatus.UNAVAILABLE, str(portfolio.get("status")), timestamp),
            HealthComponent("Regime", SystemStatus.HEALTHY if regimes.get("status") == "AVAILABLE" else SystemStatus.UNAVAILABLE, str(regimes.get("status")), timestamp),
            HealthComponent("Risk", SystemStatus.HEALTHY if risk.get("status") == "AVAILABLE" else SystemStatus.UNAVAILABLE, f"{risk.get('engine_name', 'UNAVAILABLE')} / {risk.get('current_state', 'UNAVAILABLE')}", timestamp),
            HealthComponent("Strategies", SystemStatus.HEALTHY if strategies.get("strategies") else SystemStatus.UNAVAILABLE, f"{len(strategies.get('strategies', []))} strategy record(s)", timestamp),
            HealthComponent("Trading costs", SystemStatus.WARNING if cost_coverage == "INCOMPLETE" else SystemStatus.UNAVAILABLE if cost_coverage == "UNAVAILABLE" else SystemStatus.HEALTHY, f"coverage {cost_coverage}", timestamp),
        ]
        components = sorted(components, key=lambda item: item.name)
        status = (
            SystemStatus.ERROR if any(item.status is SystemStatus.ERROR for item in components)
            else SystemStatus.WARNING if any(item.status in {SystemStatus.WARNING, SystemStatus.UNAVAILABLE} for item in components)
            else SystemStatus.HEALTHY
        )
        return HealthSnapshot(str(summary.get("run_id", "unknown")), timestamp, status, tuple(components))

    def build_traces(self, data: BacktestMonitoringData) -> tuple[DecisionTrace, ...]:
        tables = data.tables
        signals = {str(item.get("signal_id")): item for item in tables.get("signals", ())}
        regimes = {str(item.get("snapshot_id")): item for item in tables.get("regime_snapshots", ())}
        activations = {str(item.get("decision_id")): item for item in tables.get("activation_decisions", ())}
        ml_decisions = {str(item.get("decision_id")): item for item in tables.get("ml_decisions", ())}
        predictions = {str(item.get("prediction_id")): item for item in tables.get("ml_predictions", ())}
        portfolio_decisions = {str(item.get("decision_id")): item for item in tables.get("portfolio_decisions", ())}
        risk_decisions = {str(item.get("decision_id")): item for item in tables.get("risk_decisions", ())}
        cost_estimates = {str(item.get("estimate_id")): item for item in tables.get("cost_estimates", ())}
        economic_decisions = {str(item.get("decision_id")): item for item in tables.get("economic_decisions", ())}
        actual_costs = {str(item.get("fill_id")): item for item in tables.get("cost_actuals", ())}
        reconciliations = {str(item.get("fill_id")): item for item in tables.get("cost_reconciliation", ())}
        fills: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in tables.get("fills", ()):
            fills[str(item.get("order_id"))].append(item)
        dataset_refs = data.summary.get("dataset_references", []) or []
        traces: list[DecisionTrace] = []
        for order in sorted(tables.get("orders", ()), key=lambda row: _iso_key(row, "created_at")):
            order_id = str(order.get("order_id"))
            signal = signals.get(str(order.get("signal_id")))
            activation = activations.get(str(order.get("activation_decision_id")))
            regime = regimes.get(str(activation.get("regime_snapshot_id"))) if activation else None
            ml_decision = ml_decisions.get(str(order.get("ml_decision_id")))
            prediction = predictions.get(str(ml_decision.get("prediction_id"))) if ml_decision else None
            portfolio = portfolio_decisions.get(str(order.get("portfolio_decision_id")))
            cost_estimate = cost_estimates.get(str(order.get("cost_estimate_id")))
            economic = economic_decisions.get(str(order.get("economic_decision_id")))
            risk = risk_decisions.get(str(order.get("risk_decision_id")))
            order_fills = fills.get(order_id, [])
            actual = (
                actual_costs.get(str(order_fills[0].get("fill_id")))
                if order_fills else None
            )
            reconciliation = (
                reconciliations.get(str(order_fills[0].get("fill_id")))
                if order_fills else None
            )
            timestamp = _utc(order.get("created_at"))
            symbol = str(order.get("symbol") or "UNAVAILABLE")
            strategy_name = signal.get("strategy_name") if signal else None
            dataset_for_symbol = [item for item in dataset_refs if item.get("symbol") == symbol]
            raw_steps: dict[str, tuple[dict[str, Any] | list[Any] | None, str | None, Iterable[str], Iterable[str]]] = {
                "Dataset": (dataset_for_symbol, dataset_for_symbol[0].get("dataset_id") if dataset_for_symbol else None, (), ()),
                "Feature": (signal.get("features_used") if signal else None, "feature-" + stable_hash(signal.get("features_used"))[:16] if signal and signal.get("features_used") else None, (), ()),
                "Regime": (regime, regime.get("snapshot_id") if regime else None, regime.get("reason_codes", ()) if regime else (), ()),
                "Strategy": (signal, signal.get("strategy_name") if signal else None, (), (signal.get("reason"),) if signal and signal.get("reason") else ()),
                "Signal": (signal, signal.get("signal_id") if signal else None, (), (signal.get("reason"),) if signal and signal.get("reason") else ()),
                "ML": ({"decision": ml_decision, "prediction": prediction} if ml_decision or prediction else None, ml_decision.get("decision_id") if ml_decision else None, (ml_decision.get("reason_code"),) if ml_decision and ml_decision.get("reason_code") else (), (ml_decision.get("human_reason"),) if ml_decision and ml_decision.get("human_reason") else ()),
                "Activation": (activation, activation.get("decision_id") if activation else None, activation.get("reason_codes", ()) if activation else (), activation.get("human_readable_reasons", ()) if activation else ()),
                "Portfolio": (portfolio, portfolio.get("decision_id") if portfolio else None, portfolio.get("reason_codes", ()) if portfolio else (), portfolio.get("human_reasons", ()) if portfolio else ()),
                "Cost Estimate": (cost_estimate, cost_estimate.get("estimate_id") if cost_estimate else None, cost_estimate.get("warnings", ()) if cost_estimate else (), ()),
                "Economic Gate": (economic, economic.get("decision_id") if economic else None, economic.get("reason_codes", ()) if economic else (), economic.get("human_reasons", ()) if economic else ()),
                "Risk": (risk, risk.get("decision_id") if risk else None, risk.get("reason_codes", ()) if risk else (), risk.get("human_readable_reasons", ()) if risk else ()),
                "Order": (order, order_id, (str(order.get("status_reason")),) if order.get("status_reason") else (), ()),
                "Fill": (order_fills, order_fills[0].get("fill_id") if order_fills else None, (), ()),
                "Actual Cost": (actual, actual.get("actual_cost_id") if actual else None, (), ()),
                "Cost Reconciliation": (reconciliation, reconciliation.get("reconciliation_id") if reconciliation else None, (), ()),
            }
            steps = []
            for stage in _TRACE_STAGES:
                details, entity_id, codes, reasons = raw_steps[stage]
                status = SystemStatus.HEALTHY if details else SystemStatus.UNAVAILABLE
                details_object = details if isinstance(details, dict) else {"items": details or []}
                steps.append(
                    DecisionTraceStep(
                        stage=stage,
                        status=status,
                        entity_id=str(entity_id) if entity_id else None,
                        reason_codes=tuple(str(item) for item in codes if item),
                        human_reasons=tuple(str(item) for item in reasons if item),
                        details_json=json.dumps(to_primitive(details_object), sort_keys=True, separators=(",", ":"), allow_nan=False),
                    )
                )
            traces.append(
                DecisionTrace(
                    trace_id="trace-" + stable_hash({"run_id": data.run_id, "order_id": order_id})[:24],
                    run_id=data.run_id,
                    timestamp=timestamp,
                    symbol=symbol,
                    strategy_name=str(strategy_name) if strategy_name else None,
                    steps=tuple(steps),
                )
            )
        return tuple(traces)
