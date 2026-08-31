"""Point-in-time, post-hoc diagnostics over checksum-verified engine exports."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from trading_ai.core.hashing import stable_hash
from trading_ai.robustness.config import RobustnessConfig
from trading_ai.robustness.cost_evidence import load_historical_cost_evidence
from trading_ai.robustness.governance import BaselineReproducer
from trading_ai.robustness.models import (
    CampaignStatus,
    CostCompleteness,
    CostRobustnessReport,
    DecisionFunnelReport,
    DecisionFunnelRow,
    DiagnosticAvailability,
    DiagnosticComparisonResult,
    DrawdownEpisode,
    HistoricalCoverageMatrix,
    HistoricalCoverageRow,
    HoldoutStatus,
    PeriodClassification,
    PnlConcentrationReport,
    RegimeRobustnessRow,
    ResearchBaselineManifest,
    ResearchPeriod,
    RobustnessReport,
    RobustnessResearchPlan,
    StatisticalUncertaintyReport,
    SurvivorshipStatus,
    SymbolPnlContribution,
    TemporalRobustnessRow,
    UncertaintyStatus,
)


ZERO = Decimal("0")
ONE = Decimal("1")


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _utc(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("robustness diagnostics require timezone-aware timestamps")
    return parsed.astimezone(timezone.utc)


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _status(summary: Mapping[str, Any], *path: str) -> Any:
    current: Any = summary
    for name in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(name)
    return current


def _rows_between(
    rows: Iterable[dict[str, Any]],
    start: datetime,
    end: datetime,
    field: str = "timestamp",
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        raw = row.get(field)
        if raw is None:
            continue
        timestamp = _utc(raw)
        if start <= timestamp <= end:
            result.append(row)
    return result


class HistoricalCoverageAnalyzer:
    @staticmethod
    def build(summary: dict[str, Any]) -> HistoricalCoverageMatrix:
        references = {
            str(item.get("symbol")): item
            for item in summary.get("dataset_references", ())
            if isinstance(item, dict) and item.get("symbol")
        }
        quality = {
            str(item.get("symbol")): item
            for item in summary.get("data_quality_reports", ())
            if isinstance(item, dict) and item.get("symbol")
        }
        rows: list[HistoricalCoverageRow] = []
        limitations: set[str] = set()
        for symbol in sorted(set(references) | set(quality)):
            reference = references.get(symbol, {})
            report = quality.get(symbol, {})
            first = report.get("first_timestamp") or reference.get("actual_start")
            last = report.get("last_timestamp") or reference.get("actual_end")
            if first is None or last is None:
                limitations.add(f"{symbol}:COVERAGE_RANGE_UNAVAILABLE")
                continue
            warnings = tuple(str(item) for item in report.get("warnings", ()) or ())
            if warnings:
                limitations.update(f"{symbol}:{item}" for item in warnings)
            rows.append(
                HistoricalCoverageRow(
                    symbol=symbol,
                    timeframe=str(reference.get("timeframe") or report.get("timeframe")),
                    dataset_id=str(reference.get("dataset_id")),
                    checksum=str(reference.get("checksum_sha256")),
                    first_timestamp=_utc(first),
                    last_timestamp=_utc(last),
                    row_count=int(report.get("row_count", 0)),
                    missing_expected_bars=int(
                        report.get("missing_expected_bar_count", 0)
                    ),
                    duplicate_count=int(report.get("duplicate_count", 0)),
                    invalid_bar_count=int(report.get("invalid_bar_count", 0)),
                    corporate_actions_available=bool(
                        reference.get("corporate_actions_dataset_id")
                        and reference.get("corporate_actions_checksum_sha256")
                    ),
                    quality_status=str(report.get("quality_status", "UNAVAILABLE")),
                    warnings=warnings,
                )
            )
        common_start = max((item.first_timestamp for item in rows), default=None)
        common_end = min((item.last_timestamp for item in rows), default=None)
        available = bool(rows) and common_start is not None and common_end is not None and common_start <= common_end
        providers = sorted(
            {str(item.get("provider")) for item in references.values() if item.get("provider")}
        )
        limitations.add(
            "PROVIDER_AVAILABILITY_IS_NOT_POINT_IN_TIME_UNIVERSE_MEMBERSHIP"
        )
        if providers:
            limitations.add("PROVIDERS:" + ",".join(providers))
        return HistoricalCoverageMatrix(
            rows=tuple(rows),
            common_start=common_start,
            common_end=common_end,
            common_history_available=available,
            provider_limitations=tuple(sorted(limitations)),
        )


class DecisionFunnelAnalyzer:
    @staticmethod
    def build(
        summary: dict[str, Any], tables: dict[str, tuple[dict[str, Any], ...]]
    ) -> DecisionFunnelReport:
        entry_signals = {
            str(item.get("signal_id")): item
            for item in tables.get("signals", ())
            if item.get("signal_id") and item.get("action") == "ENTER_LONG"
        }
        exit_signals = [
            item for item in tables.get("signals", ()) if item.get("action") == "EXIT_LONG"
        ]
        ml_mode = str(_status(summary, "ml", "mode") or "DISABLED")
        ml_by_signal = {
            str(item.get("signal_id")): item
            for item in tables.get("ml_decisions", ())
            if item.get("signal_id")
        }
        activation_by_signal = {
            str(item.get("signal_id")): item
            for item in tables.get("activation_decisions", ())
            if item.get("signal_id")
        }
        opportunity_by_id = {
            str(item.get("opportunity_id")): item
            for item in tables.get("portfolio_opportunities", ())
            if item.get("opportunity_id")
        }
        portfolio_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in tables.get("portfolio_decisions", ()):
            signal_id = item.get("signal_id")
            if not signal_id and item.get("opportunity_id"):
                signal_id = opportunity_by_id.get(str(item["opportunity_id"]), {}).get(
                    "signal_id"
                )
            if signal_id:
                portfolio_by_signal[str(signal_id)].append(item)
        economic_by_signal = {
            str(item.get("signal_id")): item
            for item in tables.get("economic_decisions", ())
            if item.get("signal_id")
        }
        orders_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
        orders_by_id: dict[str, dict[str, Any]] = {}
        for item in tables.get("orders", ()):
            order_id = str(item.get("order_id"))
            orders_by_id[order_id] = item
            if item.get("signal_id"):
                orders_by_signal[str(item["signal_id"])].append(item)
        risk_by_order = {
            str(item.get("order_id")): item
            for item in tables.get("risk_decisions", ())
            if item.get("order_id")
        }
        filled_order_ids = {
            str(item.get("order_id"))
            for item in tables.get("fills", ())
            if item.get("order_id") and item.get("side") == "BUY"
        }
        closed_by_symbol = Counter(
            str(item.get("symbol")) for item in tables.get("trades", ()) if item.get("symbol")
        )
        exits_by_key = Counter(
            (str(item.get("strategy_name")), str(item.get("symbol")))
            for item in exit_signals
        )
        keys = sorted(
            {
                (str(item.get("strategy_name")), str(item.get("symbol")))
                for item in entry_signals.values()
            }
        )
        rows: list[DecisionFunnelRow] = []
        reasons: Counter[str] = Counter()
        for item in activation_by_signal.values():
            if str(item.get("status")) == "BLOCK":
                reasons.update(str(reason) for reason in item.get("reason_codes", ()) or ())
        for items in portfolio_by_signal.values():
            for item in items:
                if str(item.get("status")) in {"DEFER", "REJECT"}:
                    reasons.update(str(reason) for reason in item.get("reason_codes", ()) or ())
        for item in economic_by_signal.values():
            if not bool(item.get("allows_new_risk")):
                reasons.update(str(reason) for reason in item.get("reason_codes", ()) or ())
        for item in risk_by_order.values():
            if str(item.get("status")) == "REJECT":
                reasons.update(str(reason) for reason in item.get("reason_codes", ()) or ())

        for strategy_name, symbol in keys:
            candidates = {
                signal_id
                for signal_id, item in entry_signals.items()
                if str(item.get("strategy_name")) == strategy_name
                and str(item.get("symbol")) == symbol
            }
            if ml_mode == "DISABLED":
                ml_eligible = set(candidates)
                ml_blocked: set[str] = set()
            else:
                ml_eligible = {
                    signal_id
                    for signal_id in candidates
                    if str(ml_by_signal.get(signal_id, {}).get("status"))
                    in {"PASS", "NOT_APPLICABLE"}
                }
                ml_blocked = {
                    signal_id
                    for signal_id in candidates
                    if str(ml_by_signal.get(signal_id, {}).get("status"))
                    in {"BLOCK", "UNAVAILABLE"}
                }
            activation_eligible = {
                signal_id
                for signal_id in ml_eligible
                if str(activation_by_signal.get(signal_id, {}).get("status"))
                in {"ALLOW", "REDUCE"}
            }
            activation_blocked = {
                signal_id
                for signal_id in ml_eligible
                if str(activation_by_signal.get(signal_id, {}).get("status")) == "BLOCK"
            }
            selected = {
                signal_id
                for signal_id in activation_eligible
                if any(
                    str(item.get("status")) in {"SELECT", "NO_CHANGE"}
                    for item in portfolio_by_signal.get(signal_id, ())
                )
            }
            deferred = {
                signal_id
                for signal_id in activation_eligible
                if any(
                    str(item.get("status")) == "DEFER"
                    for item in portfolio_by_signal.get(signal_id, ())
                )
            }
            rejected = {
                signal_id
                for signal_id in activation_eligible
                if any(
                    str(item.get("status")) == "REJECT"
                    for item in portfolio_by_signal.get(signal_id, ())
                )
            }
            economic_eligible = {
                signal_id
                for signal_id in selected
                if bool(economic_by_signal.get(signal_id, {}).get("allows_new_risk"))
            }
            economic_blocked = {
                signal_id
                for signal_id in selected
                if economic_by_signal.get(signal_id)
                and not bool(economic_by_signal[signal_id].get("allows_new_risk"))
            }
            economic_incomplete = {
                signal_id
                for signal_id in selected
                if str(economic_by_signal.get(signal_id, {}).get("status")) == "INCOMPLETE"
            }
            risk_approved: set[str] = set()
            risk_reduced: set[str] = set()
            risk_rejected: set[str] = set()
            filled: set[str] = set()
            for signal_id in economic_eligible:
                for order in orders_by_signal.get(signal_id, ()):
                    order_id = str(order.get("order_id"))
                    status = str(risk_by_order.get(order_id, {}).get("status"))
                    if status == "APPROVE":
                        risk_approved.add(signal_id)
                    elif status == "REDUCE":
                        risk_reduced.add(signal_id)
                    elif status == "REJECT":
                        risk_rejected.add(signal_id)
                    if order_id in filled_order_ids:
                        filled.add(signal_id)
            rows.append(
                DecisionFunnelRow(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    candidate_entries=len(candidates),
                    ml_eligible=len(ml_eligible),
                    ml_blocked=len(ml_blocked),
                    activation_eligible=len(activation_eligible),
                    activation_blocked=len(activation_blocked),
                    portfolio_selected=len(selected),
                    portfolio_deferred=len(deferred),
                    portfolio_rejected=len(rejected),
                    economic_eligible=len(economic_eligible),
                    economic_blocked=len(economic_blocked),
                    economic_incomplete=len(economic_incomplete),
                    risk_approved=len(risk_approved),
                    risk_reduced=len(risk_reduced),
                    risk_rejected=len(risk_rejected),
                    filled_entries=len(filled),
                    exit_signals=exits_by_key[(strategy_name, symbol)],
                    closed_trades=0,
                )
            )
        # Closed trades cannot be rigorously assigned to overlapping sleeves.
        # Keep exact symbol totals in explicit aggregate rows instead of inventing
        # strategy PnL attribution.
        for symbol, count in sorted(closed_by_symbol.items()):
            candidates = {
                signal_id
                for signal_id, item in entry_signals.items()
                if str(item.get("symbol")) == symbol
            }
            filled = {
                str(order.get("signal_id"))
                for order_id, order in orders_by_id.items()
                if order_id in filled_order_ids
                and str(order.get("symbol")) == symbol
                and order.get("signal_id")
            }
            # Aggregate rows use the exact unique sets at each stage.
            selected = {
                signal_id
                for signal_id in candidates
                if any(
                    str(item.get("status")) in {"SELECT", "NO_CHANGE"}
                    for item in portfolio_by_signal.get(signal_id, ())
                )
            }
            econ = {
                signal_id
                for signal_id in selected
                if bool(economic_by_signal.get(signal_id, {}).get("allows_new_risk"))
            }
            risk_ok = {
                str(order.get("signal_id"))
                for signal_id in econ
                for order in orders_by_signal.get(signal_id, ())
                if str(risk_by_order.get(str(order.get("order_id")), {}).get("status"))
                in {"APPROVE", "REDUCE"}
            }
            rows.append(
                DecisionFunnelRow(
                    strategy_name="ALL_STRATEGIES",
                    symbol=symbol,
                    candidate_entries=len(candidates),
                    ml_eligible=len(candidates) if ml_mode == "DISABLED" else len(
                        {item for item in candidates if item in ml_by_signal and str(ml_by_signal[item].get("status")) == "PASS"}
                    ),
                    ml_blocked=0 if ml_mode == "DISABLED" else len(
                        {item for item in candidates if str(ml_by_signal.get(item, {}).get("status")) in {"BLOCK", "UNAVAILABLE"}}
                    ),
                    activation_eligible=len(
                        {item for item in candidates if str(activation_by_signal.get(item, {}).get("status")) in {"ALLOW", "REDUCE"}}
                    ),
                    activation_blocked=len(
                        {item for item in candidates if str(activation_by_signal.get(item, {}).get("status")) == "BLOCK"}
                    ),
                    portfolio_selected=len(selected),
                    portfolio_deferred=len(
                        {item for item in candidates if any(str(row.get("status")) == "DEFER" for row in portfolio_by_signal.get(item, ()))}
                    ),
                    portfolio_rejected=len(
                        {item for item in candidates if any(str(row.get("status")) == "REJECT" for row in portfolio_by_signal.get(item, ()))}
                    ),
                    economic_eligible=len(econ),
                    economic_blocked=len(
                        {item for item in selected if economic_by_signal.get(item) and not bool(economic_by_signal[item].get("allows_new_risk"))}
                    ),
                    economic_incomplete=len(
                        {item for item in selected if str(economic_by_signal.get(item, {}).get("status")) == "INCOMPLETE"}
                    ),
                    risk_approved=len(
                        {str(order.get("signal_id")) for item in econ for order in orders_by_signal.get(item, ()) if str(risk_by_order.get(str(order.get("order_id")), {}).get("status")) == "APPROVE"}
                    ),
                    risk_reduced=len(
                        {str(order.get("signal_id")) for item in econ for order in orders_by_signal.get(item, ()) if str(risk_by_order.get(str(order.get("order_id")), {}).get("status")) == "REDUCE"}
                    ),
                    risk_rejected=len(
                        {str(order.get("signal_id")) for item in econ for order in orders_by_signal.get(item, ()) if str(risk_by_order.get(str(order.get("order_id")), {}).get("status")) == "REJECT"}
                    ),
                    filled_entries=len(filled & risk_ok),
                    exit_signals=sum(1 for item in exit_signals if str(item.get("symbol")) == symbol),
                    closed_trades=count,
                )
            )
        totals = [item for item in rows if item.strategy_name == "ALL_STRATEGIES"]
        interpretation = (
            f"candidate ENTER_LONG signals: {sum(item.candidate_entries for item in totals)}",
            f"activation eligible: {sum(item.activation_eligible for item in totals)}",
            f"portfolio selected: {sum(item.portfolio_selected for item in totals)}",
            f"risk approved/reduced: {sum(item.risk_approved + item.risk_reduced for item in totals)}",
            f"filled entries: {sum(item.filled_entries for item in totals)}",
            f"closed trades: {sum(item.closed_trades for item in totals)}",
            "Closed trades are symbol-level only because overlapping sleeves do not provide rigorous strategy PnL attribution.",
        )
        return DecisionFunnelReport(
            rows=tuple(sorted(rows, key=lambda item: (item.strategy_name, item.symbol))),
            drop_reasons=tuple(sorted(reasons.items())),
            interpretation=interpretation,
        )


class DrawdownAnalyzer:
    @staticmethod
    def build(
        tables: dict[str, tuple[dict[str, Any], ...]], minimum_fraction: Decimal
    ) -> tuple[DrawdownEpisode, ...]:
        equity = sorted(
            (
                (_utc(item["timestamp"]), _decimal(item.get("equity")), item)
                for item in tables.get("equity", ())
                if item.get("timestamp") is not None and _decimal(item.get("equity")) is not None
            ),
            key=lambda item: item[0],
        )
        if not equity:
            return ()
        episodes_raw: list[tuple[datetime, Decimal, datetime, Decimal, datetime | None]] = []
        peak_time, peak, _ = equity[0]
        assert peak is not None
        active_peak_time: datetime | None = None
        active_peak: Decimal | None = None
        trough_time: datetime | None = None
        trough: Decimal | None = None
        for timestamp, value, _ in equity[1:]:
            assert value is not None
            if active_peak is None:
                if value >= peak:
                    peak, peak_time = value, timestamp
                    continue
                active_peak_time, active_peak = peak_time, peak
                trough_time, trough = timestamp, value
                continue
            if value < (trough if trough is not None else value + ONE):
                trough_time, trough = timestamp, value
            if value >= active_peak:
                assert active_peak_time is not None and trough_time is not None and trough is not None
                episodes_raw.append(
                    (active_peak_time, active_peak, trough_time, trough, timestamp)
                )
                peak_time, peak = timestamp, value
                active_peak_time = active_peak = trough_time = trough = None
        if active_peak is not None:
            assert active_peak_time is not None and trough_time is not None and trough is not None
            episodes_raw.append((active_peak_time, active_peak, trough_time, trough, None))

        result: list[DrawdownEpisode] = []
        final_time = equity[-1][0]
        ledger = sorted(tables.get("ledger", ()), key=lambda item: str(item.get("timestamp")))
        regimes = sorted(tables.get("regime_snapshots", ()), key=lambda item: str(item.get("timestamp")))
        for peak_at, peak_value, trough_at, trough_value, recovery_at in episodes_raw:
            fraction = ZERO if peak_value == ZERO else (peak_value - trough_value) / peak_value
            if fraction < minimum_fraction:
                continue
            episode_end = recovery_at or final_time
            quantities: dict[str, Decimal] = defaultdict(lambda: ZERO)
            for item in ledger:
                if item.get("timestamp") and _utc(item["timestamp"]) <= trough_at:
                    change = _decimal(item.get("quantity_change"))
                    if item.get("symbol") and change is not None:
                        quantities[str(item["symbol"])] += change
            held = tuple(sorted(symbol for symbol, quantity in quantities.items() if quantity != ZERO))
            realized: dict[str, Decimal] = defaultdict(lambda: ZERO)
            for trade in _rows_between(tables.get("trades", ()), peak_at, episode_end, "exit_time"):
                pnl = _decimal(trade.get("net_pnl"))
                if trade.get("symbol") and pnl is not None:
                    realized[str(trade["symbol"])] += pnl
            risk_decisions = _rows_between(tables.get("risk_decisions", ()), peak_at, episode_end)
            risk_counts = Counter(str(item.get("status")) for item in risk_decisions)
            state_rows = _rows_between(tables.get("risk_states", ()), peak_at, episode_end)
            risk_states = tuple(
                dict.fromkeys(
                    str(item.get("new_state"))
                    for item in state_rows
                    if item.get("new_state")
                )
            )
            latest_regime: dict[str, dict[str, Any]] = {}
            for item in regimes:
                if item.get("timestamp") and _utc(item["timestamp"]) <= trough_at and item.get("symbol"):
                    latest_regime[str(item["symbol"])] = item
            structure = tuple(
                sorted(
                    (symbol, str(latest_regime[symbol].get("structure_regime", "UNKNOWN")))
                    for symbol in held
                    if symbol in latest_regime
                )
            )
            volatility = tuple(
                sorted(
                    (symbol, str(latest_regime[symbol].get("volatility_regime", "UNKNOWN")))
                    for symbol in held
                    if symbol in latest_regime
                )
            )
            costs = ZERO
            for item in _rows_between(tables.get("cost_actuals", ()), peak_at, episode_end):
                amount = _decimal(
                    _json(item.get("breakdown")).get("total_variable_cost", {}).get("amount")
                )
                if amount is not None:
                    costs += amount
            max_exposure: Decimal | None = None
            for _, _, row in equity:
                timestamp = _utc(row["timestamp"])
                if not peak_at <= timestamp <= episode_end:
                    continue
                positions = _decimal(row.get("positions_value"))
                value = _decimal(row.get("equity"))
                if positions is None or value in (None, ZERO):
                    continue
                exposure = abs(positions) / value
                max_exposure = exposure if max_exposure is None else max(max_exposure, exposure)
            payload = {
                "peak": peak_at,
                "trough": trough_at,
                "recovery": recovery_at,
                "drawdown": fraction,
            }
            result.append(
                DrawdownEpisode(
                    episode_id="drawdown-" + stable_hash(payload)[:24],
                    peak_timestamp=peak_at,
                    peak_equity=peak_value,
                    trough_timestamp=trough_at,
                    trough_equity=trough_value,
                    recovery_timestamp=recovery_at,
                    drawdown_fraction=fraction,
                    duration_seconds=(episode_end - peak_at).total_seconds(),
                    held_symbols_at_trough=held,
                    realized_closure_pnl_by_symbol=tuple(sorted(realized.items())),
                    max_gross_exposure=max_exposure,
                    risk_states=risk_states,
                    risk_decision_counts=tuple(sorted(risk_counts.items())),
                    structure_regimes_at_trough=structure,
                    volatility_regimes_at_trough=volatility,
                    variable_costs_during_episode=costs,
                )
            )
        return tuple(sorted(result, key=lambda item: item.peak_timestamp))


class ConcentrationAnalyzer:
    @staticmethod
    def build(
        tables: dict[str, tuple[dict[str, Any], ...]],
        symbols: tuple[str, ...],
        warning_fraction: Decimal,
    ) -> PnlConcentrationReport:
        stats: dict[str, dict[str, Any]] = {
            symbol: {"count": 0, "gross": ZERO, "net": ZERO, "profit": ZERO, "loss": ZERO}
            for symbol in symbols
        }
        for trade in tables.get("trades", ()):
            symbol = str(trade.get("symbol"))
            stats.setdefault(symbol, {"count": 0, "gross": ZERO, "net": ZERO, "profit": ZERO, "loss": ZERO})
            gross = _decimal(trade.get("gross_pnl")) or ZERO
            net = _decimal(trade.get("net_pnl")) or ZERO
            stats[symbol]["count"] += 1
            stats[symbol]["gross"] += gross
            stats[symbol]["net"] += net
            if net > ZERO:
                stats[symbol]["profit"] += net
            elif net < ZERO:
                stats[symbol]["loss"] += net
        target_weights: dict[str, list[Decimal]] = defaultdict(list)
        for target in tables.get("portfolio_targets", ()):
            weight = _decimal(target.get("current_weight"))
            if target.get("symbol") and weight is not None:
                target_weights[str(target["symbol"])].append(abs(weight))
        positive_total = sum((value["profit"] for value in stats.values()), ZERO)
        rows: list[SymbolPnlContribution] = []
        for symbol in sorted(stats):
            value = stats[symbol]
            weights = target_weights.get(symbol, ())
            share = value["profit"] / positive_total if positive_total > ZERO else None
            rows.append(
                SymbolPnlContribution(
                    symbol=symbol,
                    closed_trades=int(value["count"]),
                    gross_pnl=value["gross"],
                    net_pnl=value["net"],
                    gross_profit=value["profit"],
                    gross_loss=value["loss"],
                    average_target_weight=(sum(weights, ZERO) / len(weights)) if weights else None,
                    share_of_positive_pnl=share,
                )
            )
        positive_rows = sorted(
            (item for item in rows if item.share_of_positive_pnl is not None),
            key=lambda item: (item.share_of_positive_pnl or ZERO, item.symbol),
            reverse=True,
        )
        top1 = positive_rows[0].share_of_positive_pnl if positive_rows else None
        top3 = (
            sum((item.share_of_positive_pnl or ZERO for item in positive_rows[:3]), ZERO)
            if positive_rows else None
        )
        hhi = (
            sum(((item.share_of_positive_pnl or ZERO) ** 2 for item in positive_rows), ZERO)
            if positive_rows else None
        )
        return PnlConcentrationReport(
            symbols=tuple(rows),
            top_contributor=positive_rows[0].symbol if positive_rows else None,
            top1_positive_pnl_share=top1,
            top3_positive_pnl_share=top3,
            positive_pnl_hhi=hhi,
            dominant_symbol_warning=top1 is not None and top1 > warning_fraction,
            definition=(
                "Shares and HHI use positive closed-trade net PnL; losses remain "
                "reported separately and no asset is removed automatically."
            ),
        )


class TemporalAnalyzer:
    @staticmethod
    def build(
        tables: dict[str, tuple[dict[str, Any], ...]],
        *,
        period: ResearchPeriod | None = None,
    ) -> tuple[TemporalRobustnessRow, ...]:
        equity = sorted(tables.get("equity", ()), key=lambda item: str(item.get("timestamp")))
        if not equity:
            return ()
        years = range(_utc(equity[0]["timestamp"]).year, _utc(equity[-1]["timestamp"]).year + 1)
        windows: list[tuple[str, datetime, datetime]] = [
            (
                str(year),
                datetime(year, 1, 1, tzinfo=timezone.utc),
                datetime(year + 1, 1, 1, tzinfo=timezone.utc),
            )
            for year in years
        ]
        if period is not None:
            width = (period.end - period.start) / 3
            boundaries = (
                period.start,
                period.start + width,
                period.start + width * 2,
                period.end,
            )
            windows.extend(
                (
                    f"SUBPERIOD_{index}",
                    boundaries[index - 1],
                    boundaries[index],
                )
                for index in range(1, 4)
            )
        rows: list[TemporalRobustnessRow] = []
        for label, start, end in windows:
            points = [item for item in equity if start <= _utc(item["timestamp"]) < end]
            trades = [item for item in tables.get("trades", ()) if start <= _utc(item["exit_time"]) < end]
            costs = [item for item in tables.get("cost_actuals", ()) if start <= _utc(item["timestamp"]) < end]
            fills = [item for item in tables.get("fills", ()) if start <= _utc(item["timestamp"]) < end]
            if not points:
                rows.append(
                    TemporalRobustnessRow(
                        label=label, start=start, end=end,
                        availability=DiagnosticAvailability.UNAVAILABLE,
                        gross_return=None, net_return=None, closed_trades=0,
                        expectancy=None, profit_factor=None, max_drawdown=None,
                        average_exposure=None, turnover=None, variable_costs=None,
                    )
                )
                continue
            first = _decimal(points[0].get("equity"))
            last = _decimal(points[-1].get("equity"))
            net_return = (last / first - ONE) if first not in (None, ZERO) and last is not None else None
            variable_cost = ZERO
            for item in costs:
                amount = _decimal(_json(item.get("breakdown")).get("total_variable_cost", {}).get("amount"))
                if amount is not None:
                    variable_cost += amount
            gross_return = (
                ((last + variable_cost) / first - ONE)
                if first not in (None, ZERO) and last is not None else None
            )
            pnl = [_decimal(item.get("net_pnl")) or ZERO for item in trades]
            wins = sum((item for item in pnl if item > ZERO), ZERO)
            losses = sum((item for item in pnl if item < ZERO), ZERO)
            expectancy = sum(pnl, ZERO) / len(pnl) if pnl else None
            profit_factor = wins / abs(losses) if losses < ZERO else None
            peak: Decimal | None = None
            max_dd = ZERO
            exposures: list[Decimal] = []
            for item in points:
                value = _decimal(item.get("equity"))
                positions = _decimal(item.get("positions_value"))
                if value is None:
                    continue
                peak = value if peak is None or value > peak else peak
                if peak != ZERO:
                    max_dd = max(max_dd, (peak - value) / peak)
                if value != ZERO and positions is not None:
                    exposures.append(abs(positions) / value)
            notionals = sum(
                (
                    abs((_decimal(item.get("quantity")) or ZERO) * (_decimal(item.get("price")) or ZERO))
                    for item in fills
                ),
                ZERO,
            )
            turnover = notionals / first if first not in (None, ZERO) else None
            rows.append(
                TemporalRobustnessRow(
                    label=label, start=start, end=end,
                    availability=DiagnosticAvailability.AVAILABLE,
                    gross_return=gross_return, net_return=net_return,
                    closed_trades=len(trades), expectancy=expectancy,
                    profit_factor=profit_factor, max_drawdown=max_dd,
                    average_exposure=(sum(exposures, ZERO) / len(exposures)) if exposures else None,
                    turnover=turnover, variable_costs=variable_cost,
                )
            )
        return tuple(rows)


class RegimeAnalyzer:
    @staticmethod
    def build(
        tables: dict[str, tuple[dict[str, Any], ...]]
    ) -> tuple[RegimeRobustnessRow, ...]:
        by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in tables.get("regime_snapshots", ()):
            if item.get("symbol") and item.get("timestamp"):
                by_symbol[str(item["symbol"])].append(item)
        for values in by_symbol.values():
            values.sort(key=lambda item: _utc(item["timestamp"]))
        stats: dict[tuple[str, str], list[tuple[Decimal, Decimal | None]]] = defaultdict(list)
        for trade in tables.get("trades", ()):
            symbol = str(trade.get("symbol"))
            entry = _utc(trade["entry_time"])
            available = [
                item for item in by_symbol.get(symbol, ()) if _utc(item["timestamp"]) <= entry
            ]
            if not available:
                key = ("UNKNOWN", "UNKNOWN")
            else:
                latest = available[-1]
                key = (
                    str(latest.get("structure_regime", "UNKNOWN")),
                    str(latest.get("volatility_regime", "UNKNOWN")),
                )
            stats[key].append(
                (
                    _decimal(trade.get("net_pnl")) or ZERO,
                    _decimal(trade.get("return_pct")),
                )
            )
        return tuple(
            RegimeRobustnessRow(
                structure_regime=key[0],
                volatility_regime=key[1],
                closed_trades=len(values),
                net_pnl=sum((item[0] for item in values), ZERO),
                average_return=(
                    sum((item[1] for item in values if item[1] is not None), ZERO)
                    / sum(1 for item in values if item[1] is not None)
                    if any(item[1] is not None for item in values)
                    else None
                ),
            )
            for key, values in sorted(stats.items())
        )


class CostRobustnessAnalyzer:
    @staticmethod
    def build(
        summary: dict[str, Any], tables: dict[str, tuple[dict[str, Any], ...]]
    ) -> CostRobustnessReport:
        evidence = load_historical_cost_evidence()
        cost_root = summary.get("costs") if isinstance(summary.get("costs"), dict) else {}
        cost_summary = cost_root.get("summary") if isinstance(cost_root.get("summary"), dict) else {}
        operating = cost_root.get("operating") if isinstance(cost_root.get("operating"), dict) else {}
        variable_complete = str(cost_summary.get("cost_coverage")) == "COMPLETE"
        operating_total = operating.get("total_operating_cost") if isinstance(operating.get("total_operating_cost"), dict) else {}
        operating_status = str(operating_total.get("status", "UNAVAILABLE"))
        tariff_status = str(cost_summary.get("tariff_status") or cost_root.get("tariff_status") or "UNAVAILABLE")
        tariff_period_verified = all(
            bool(item.get("tariff_period_covered"))
            for item in tables.get("cost_estimates", ())
        ) if tables.get("cost_estimates", ()) else False
        if variable_complete and operating_status == "KNOWN" and tariff_period_verified:
            completeness = CostCompleteness.COMPLETE_VERIFIED
        elif variable_complete and operating_status in {"KNOWN", "ESTIMATED"}:
            completeness = CostCompleteness.COMPLETE_ESTIMATED
        elif variable_complete:
            completeness = CostCompleteness.VARIABLE_COST_COMPLETE
        elif operating_status in {"KNOWN", "ESTIMATED"}:
            completeness = CostCompleteness.OPERATING_COST_COMPLETE
        else:
            completeness = CostCompleteness.INCOMPLETE
        total = _decimal(cost_summary.get("total_variable_cost"))
        trades = len(tables.get("trades", ()))
        notional = sum(
            (
                abs((_decimal(item.get("quantity")) or ZERO) * (_decimal(item.get("reference_price")) or ZERO))
                for item in tables.get("fills", ())
            ),
            ZERO,
        )
        gross_profit = sum(
            (
                max(_decimal(item.get("gross_pnl")) or ZERO, ZERO)
                for item in tables.get("trades", ())
            ),
            ZERO,
        )
        estimate_error = sum(
            (_decimal(item.get("estimate_error")) or ZERO for item in tables.get("cost_reconciliation", ())),
            ZERO,
        ) if tables.get("cost_reconciliation", ()) else None
        gross_pnl = _decimal(cost_summary.get("gross_trading_pnl"))
        initial = _decimal(summary.get("initial_cash"))
        stresses = []
        for multiplier in (Decimal("1"), Decimal("1.25"), Decimal("1.5"), Decimal("2")):
            stressed_cost = total * multiplier if total is not None else None
            stressed_return = (
                (gross_pnl - stressed_cost) / initial
                if gross_pnl is not None and stressed_cost is not None and initial not in (None, ZERO)
                else None
            )
            stresses.append((multiplier, stressed_cost, stressed_return))
        warnings = []
        if not tariff_period_verified:
            warnings.extend(
                ("HISTORICAL_TARIFF_UNVERIFIED", "CURRENT_TARIFF_APPLIED_RETROSPECTIVELY")
            )
        if operating_status == "UNAVAILABLE":
            warnings.append("OPERATING_COSTS_INCOMPLETE")
        return CostRobustnessReport(
            completeness=completeness,
            evidence_registry_hash=evidence.registry_hash,
            evidence_verified_at=evidence.verified_at,
            broker_tariff_evidence_kind=(
                evidence.broker_tariffs[0].evidence_kind.value
                if evidence.broker_tariffs else "UNAVAILABLE"
            ),
            exchange_fee_evidence_status=evidence.exchange_fee_status.value,
            fx_cost_evidence_status=evidence.fx_cost_status.value,
            annual_tax_eligibility_records=len(evidence.tax_eligibility),
            tariff_profile_id=str(cost_summary.get("tariff_profile_id") or cost_root.get("tariff_profile_id")) if cost_summary or cost_root else None,
            tariff_status=tariff_status,
            historical_tariff_status=(
                "HISTORICAL_TARIFF_VERIFIED" if tariff_period_verified else "HISTORICAL_TARIFF_UNVERIFIED"
            ),
            total_variable_cost=total,
            cost_per_closed_trade=(total / trades) if total is not None and trades else None,
            cost_to_notional=(total / notional) if total is not None and notional > ZERO else None,
            cost_to_gross_profit=(total / gross_profit) if total is not None and gross_profit > ZERO else None,
            aggregate_estimate_error=estimate_error,
            stress_results=tuple(stresses),
            operating_scenario="LOCAL_RESEARCH",
            warnings=tuple(warnings),
        )


class StatisticalUncertaintyAnalyzer:
    @staticmethod
    def build(
        trades: tuple[dict[str, Any], ...],
        *,
        initial_cash: Decimal | None,
        minimum_samples: int,
        seed: int,
        resamples: int,
    ) -> StatisticalUncertaintyReport:
        outcomes = [_decimal(item.get("net_pnl")) or ZERO for item in trades]
        count = len(outcomes)
        wins = sum(item > ZERO for item in outcomes)
        win_rate = Decimal(wins) / count if count else None
        win_interval = None
        if count:
            p = wins / count
            z = 1.959963984540054
            denominator = 1 + z * z / count
            center = (p + z * z / (2 * count)) / denominator
            margin = z * math.sqrt((p * (1 - p) + z * z / (4 * count)) / count) / denominator
            win_interval = (
                Decimal(str(max(0.0, center - margin))),
                Decimal(str(min(1.0, center + margin))),
            )
        if count < minimum_samples:
            return StatisticalUncertaintyReport(
                status=UncertaintyStatus.INSUFFICIENT_FOR_RELIABLE_CI,
                sample_count=count,
                win_rate=win_rate,
                win_rate_interval_95=win_interval,
                expectancy_interval_95=None,
                total_return_interval_95=None,
                bootstrap_seed=seed,
                bootstrap_resamples=resamples,
                warning="INSUFFICIENT_FOR_RELIABLE_CI",
            )
        generator = random.Random(seed)
        expectations: list[Decimal] = []
        returns: list[Decimal] = []
        for _ in range(resamples):
            sample = [outcomes[generator.randrange(count)] for _ in range(count)]
            total = sum(sample, ZERO)
            expectations.append(total / count)
            if initial_cash not in (None, ZERO):
                returns.append(total / initial_cash)
        expectations.sort()
        returns.sort()
        low = max(0, int(resamples * 0.025))
        high = min(resamples - 1, int(resamples * 0.975))
        return StatisticalUncertaintyReport(
            status=UncertaintyStatus.AVAILABLE,
            sample_count=count,
            win_rate=win_rate,
            win_rate_interval_95=win_interval,
            expectancy_interval_95=(expectations[low], expectations[high]),
            total_return_interval_95=(returns[low], returns[high]) if returns else None,
            bootstrap_seed=seed,
            bootstrap_resamples=resamples,
            warning="NON_PARAMETRIC_BOOTSTRAP_IS_NOT_PROOF_OF_EDGE",
        )


def comparison_result(
    *,
    diagnostic_type: str,
    excluded_item: str,
    summary: dict[str, Any] | None,
    expected_config_hashes: tuple[tuple[str, str], ...],
    label_prefix: str = "without",
    ignored_config_hash_names: tuple[str, ...] = (),
) -> DiagnosticComparisonResult:
    if summary is None:
        return DiagnosticComparisonResult(
            label=f"{label_prefix}-{excluded_item}", diagnostic_type=diagnostic_type,
            excluded_item=excluded_item, run_id=None, source_hash_sha256=None,
            availability=DiagnosticAvailability.NOT_RUN,
            net_return=None, max_drawdown=None, closed_trades=None,
            expectancy=None, profit_factor=None, config_hashes_unchanged=True,
        )
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    costs = _status(summary, "costs", "summary") or {}
    expected = dict(expected_config_hashes)
    observed = {
        "risk": _status(summary, "risk", "config_hash"),
        "regime": _status(summary, "regime", "config_hash"),
        "policy": _status(summary, "regime", "policy_config_hash"),
        "portfolio": _status(summary, "portfolio", "config_hash"),
        "cost": _status(summary, "costs", "config_hash"),
    }
    ignored = set(ignored_config_hash_names)
    unchanged = all(
        observed.get(name) == expected.get(name)
        for name in observed
        if name not in ignored
    )
    return DiagnosticComparisonResult(
        label=f"{label_prefix}-{excluded_item}", diagnostic_type=diagnostic_type,
        excluded_item=excluded_item, run_id=str(summary.get("run_id")),
        source_hash_sha256=(
            str(summary["source_hash_sha256"])
            if summary.get("source_hash_sha256") is not None
            else None
        ),
        availability=DiagnosticAvailability.AVAILABLE,
        net_return=_decimal(costs.get("net_return_before_operating") or metrics.get("total_return")),
        max_drawdown=abs(_decimal(metrics.get("max_drawdown_pct")) or ZERO),
        closed_trades=int(metrics.get("number_of_trades", 0)),
        expectancy=_decimal(metrics.get("expectancy")),
        profit_factor=_decimal(metrics.get("profit_factor")),
        config_hashes_unchanged=unchanged,
    )


class RobustnessAnalyzer:
    """Build one deterministic report without feeding diagnostics into decisions."""

    report_name = "balanced-real-data-robustness"
    report_version = "1.0"

    def __init__(self, config: RobustnessConfig) -> None:
        self.config = config

    def analyze(
        self,
        *,
        summary: dict[str, Any],
        tables: dict[str, tuple[dict[str, Any], ...]],
        integrity_verified: bool,
        baseline: ResearchBaselineManifest,
        plan: RobustnessResearchPlan,
        period_classification: PeriodClassification,
        frozen_baseline_verified: bool | None = None,
        holdout_status: HoldoutStatus | None = None,
        validation_status: str = "UNAVAILABLE",
        leave_one_symbol_runs: Mapping[str, dict[str, Any]] | None = None,
        leave_one_strategy_runs: Mapping[str, dict[str, Any]] | None = None,
        single_strategy_runs: Mapping[str, dict[str, Any]] | None = None,
    ) -> RobustnessReport:
        is_baseline_run = str(summary.get("run_id")) == baseline.run_id
        baseline_reproduced = (
            BaselineReproducer.verify(
                baseline, summary=summary, integrity_verified=integrity_verified
            )
            if is_baseline_run
            else bool(frozen_baseline_verified)
        )
        coverage = HistoricalCoverageAnalyzer.build(summary)
        funnel = DecisionFunnelAnalyzer.build(summary, tables)
        drawdowns = DrawdownAnalyzer.build(
            tables, self.config.drawdown_episode_min_fraction
        )
        concentration = ConcentrationAnalyzer.build(
            tables, plan.symbols, self.config.concentration_warning_fraction
        )
        report_period = plan.period(period_classification)
        temporal = TemporalAnalyzer.build(tables, period=report_period)
        regimes = RegimeAnalyzer.build(tables)
        costs = CostRobustnessAnalyzer.build(summary, tables)
        uncertainty = StatisticalUncertaintyAnalyzer.build(
            tables.get("trades", ()),
            initial_cash=_decimal(summary.get("initial_cash")),
            minimum_samples=self.config.minimum_bootstrap_samples,
            seed=self.config.bootstrap_seed,
            resamples=self.config.bootstrap_resamples,
        )
        symbol_runs = leave_one_symbol_runs or {}
        strategy_runs = leave_one_strategy_runs or {}
        single_runs = single_strategy_runs or {}
        loso = tuple(
            comparison_result(
                diagnostic_type="LEAVE_ONE_SYMBOL_OUT",
                excluded_item=symbol,
                summary=symbol_runs.get(symbol),
                expected_config_hashes=plan.config_hashes,
            )
            for symbol in plan.symbols
        )
        lostrategy = tuple(
            comparison_result(
                diagnostic_type="LEAVE_ONE_STRATEGY_OUT",
                excluded_item=strategy,
                summary=strategy_runs.get(strategy),
                expected_config_hashes=plan.config_hashes,
            )
            for strategy in plan.strategies
        )
        singles = tuple(
            comparison_result(
                diagnostic_type="SINGLE_STRATEGY_COMPARISON",
                excluded_item=strategy,
                summary=single_runs.get(strategy),
                expected_config_hashes=plan.config_hashes,
                label_prefix="only",
                # The production CLI intentionally retains the legacy
                # single-strategy construction path; all common decision
                # assumptions stay frozen, while the multi-strategy
                # Portfolio hash is explicitly not comparable.
                ignored_config_hash_names=("portfolio",),
            )
            for strategy in plan.strategies
        )
        warnings = list(
            baseline.warnings
            if is_baseline_run
            else tuple(f"BASELINE_V1_{item}" for item in baseline.warnings)
        )
        if concentration.dominant_symbol_warning:
            warnings.append("SYMBOL_RESULT_CONCENTRATION")
        if uncertainty.status is UncertaintyStatus.INSUFFICIENT_FOR_RELIABLE_CI:
            warnings.append("INSUFFICIENT_FOR_RELIABLE_CI")
        if any(item.availability is DiagnosticAvailability.NOT_RUN for item in loso):
            warnings.append("LEAVE_ONE_SYMBOL_OUT_NOT_RUN")
        if any(item.availability is DiagnosticAvailability.NOT_RUN for item in lostrategy):
            warnings.append("LEAVE_ONE_STRATEGY_OUT_NOT_RUN")
        if any(item.availability is DiagnosticAvailability.NOT_RUN for item in singles):
            warnings.append("SINGLE_STRATEGY_COMPARISON_NOT_RUN")
        warnings.extend(costs.warnings)
        warnings.append("SURVIVORSHIP_BIAS_UNRESOLVED")
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        if period_classification is PeriodClassification.CONSUMED_DIAGNOSTIC:
            campaign_status = CampaignStatus.FAIL
        elif period_classification is PeriodClassification.FINAL_HOLDOUT:
            duration = (
                plan.period(PeriodClassification.FINAL_HOLDOUT).end
                - plan.period(PeriodClassification.FINAL_HOLDOUT).start
            ).days
            trades = int(metrics.get("number_of_trades", 0))
            minimum_trades = int(dict(plan.frozen_validation_criteria)["minimum_closed_trades"])
            if duration < self.config.minimum_holdout_calendar_days or trades < minimum_trades:
                campaign_status = CampaignStatus.INSUFFICIENT_HOLDOUT_EVIDENCE
            elif validation_status == "PASS":
                campaign_status = CampaignStatus.PASS
            elif validation_status == "FAIL":
                campaign_status = CampaignStatus.FAIL
                warnings.append("VALIDATION_GATE_FAIL")
            elif validation_status == "BLOCKED_EXTERNAL_DATA":
                campaign_status = CampaignStatus.BLOCKED
                warnings.append("VALIDATION_BLOCKED_EXTERNAL_DATA")
            else:
                campaign_status = CampaignStatus.WARNING
        else:
            campaign_status = CampaignStatus.WARNING
        semantic = {
            "report_version": self.report_version,
            "run_id": summary.get("run_id"),
            "baseline_manifest_hash": baseline.manifest_hash,
            "plan_hash": plan.plan_hash,
            "period_classification": period_classification,
            "campaign_status": campaign_status,
            "baseline_reproduced": baseline_reproduced,
            "coverage": coverage,
            "decision_funnel": funnel,
            "drawdowns": drawdowns,
            "concentration": concentration,
            "temporal": temporal,
            "regimes": regimes,
            "costs": costs,
            "uncertainty": uncertainty,
            "loso": loso,
            "lostrategy": lostrategy,
            "single_strategy": singles,
            "survivorship": SurvivorshipStatus.SURVIVORSHIP_BIAS_UNRESOLVED,
            "holdout_status": holdout_status,
            "validation_status": validation_status,
            "warnings": tuple(dict.fromkeys(warnings)),
        }
        digest = stable_hash(semantic)
        return RobustnessReport(
            report_id=f"robustness-{digest[:24]}",
            report_version=self.report_version,
            report_hash=digest,
            created_at=datetime.now(timezone.utc),
            run_id=str(summary.get("run_id")),
            baseline_manifest_hash=baseline.manifest_hash,
            plan_hash=plan.plan_hash,
            period_classification=period_classification,
            campaign_status=campaign_status,
            baseline_reproduced=baseline_reproduced,
            coverage=coverage,
            decision_funnel=funnel,
            drawdown_episodes=drawdowns,
            concentration=concentration,
            temporal_rows=temporal,
            regime_rows=regimes,
            cost_robustness=costs,
            uncertainty=uncertainty,
            leave_one_symbol_out=loso,
            leave_one_strategy_out=lostrategy,
            single_strategy_runs=singles,
            survivorship_status=SurvivorshipStatus.SURVIVORSHIP_BIAS_UNRESOLVED,
            holdout_status=holdout_status,
            warnings=tuple(dict.fromkeys(warnings)),
        )
