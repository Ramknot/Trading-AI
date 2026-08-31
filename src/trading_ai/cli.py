"""Safety, data, risk, regime, strategy, ML, and offline backtest commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trading_ai.backtesting.engine import BacktestEngine
from trading_ai.backtesting.exceptions import BacktestError
from trading_ai.backtesting.input import load_cached_dataset
from trading_ai.backtesting.models import (
    BacktestConfig,
    CommissionConfig,
    DataQualityPolicy,
)
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.backtesting.reproducibility import to_primitive
from trading_ai.backtesting.strategy import BuyAndHoldDemoStrategy
from trading_ai.core.config import inspect_profile, load_runtime_settings
from trading_ai.core.exceptions import TradingAIError
from trading_ai.core.health import HealthReport, doctor
from trading_ai.core.models import OrderSide
from trading_ai.costs import (
    BalancedTransactionCostEngine,
    CostError,
    EconomicGate,
    inspect_cost_config,
    load_balanced_cost_config,
)
from trading_ai.costs.models import PreTradeCostRequest
from trading_ai.data.engine import DataEngine
from trading_ai.data.exceptions import DataError
from trading_ai.data.models import (
    CacheMode,
    DataFetchResult,
    DatasetInspection,
    QualityStatus,
)
from trading_ai.data.providers import YahooFinanceProvider
from trading_ai.data.storage import ParquetDataStore
from trading_ai.risk.balanced import BalancedRiskEngine
from trading_ai.risk.config import (
    inspect_risk_config,
    load_asset_groups,
    load_balanced_risk_config,
    risk_config_hash,
)
from trading_ai.features import FeatureEngine
from trading_ai.ml import (
    InferenceEngine,
    LabelConfig,
    LocalModelRegistry,
    MLMode,
    ModelConfig,
    ModelFamily,
    ModelStatus,
    SignalMLScorer,
    SignalTrainingDatasetBuilder,
    TemporalSplitConfig,
    TimeRange,
    TrainingConfig,
    TrainingPipeline,
)
from trading_ai.ml.exceptions import MLError
from trading_ai.monitoring.exceptions import MonitoringError
from trading_ai.portfolio import (
    BalancedPortfolioEngine,
    inspect_portfolio_config,
    load_asset_currencies,
    load_balanced_portfolio_config,
    portfolio_config_hash,
)
from trading_ai.regimes import (
    BalancedRegimeDetector,
    BalancedStrategyActivationPolicy,
    RegimeError,
    build_regime_report,
    inspect_regime_config,
    inspect_strategy_policy_config,
    load_balanced_regime_config,
    load_balanced_strategy_policy_config,
    regime_config_hash,
    strategy_policy_config_hash,
)
from trading_ai.strategies.config import (
    BreakoutConfig,
    MeanReversionConfig,
    MomentumConfig,
    TrendConfig,
)
from trading_ai.strategies.registry import BASELINE_STRATEGIES
from trading_ai.robustness import (
    EvidenceClosureService,
    PeriodClassification,
    RobustnessService,
    load_historical_cost_evidence,
)
from trading_ai.robustness.exceptions import RobustnessError
from trading_ai.validation import (
    LocalValidationStore,
    ResearchValidationGate,
    ValidationError,
)


def _parse_datetime(value: str) -> datetime:
    """Parse ISO input; date-only values are explicit UTC day boundaries."""

    try:
        if len(value) == 10:
            return datetime.combine(
                datetime.fromisoformat(value).date(), time.min, tzinfo=timezone.utc
            )
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected YYYY-MM-DD or a timezone-aware ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            "datetime values must include a timezone; use Z or an explicit offset"
        )
    return parsed.astimezone(timezone.utc)


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_local"),
        help="local Parquet store (default: data_local)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")


def _parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("expected a decimal number") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("decimal values must be finite")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-ai",
        description=(
            "Trading AI diagnostics, historical data, governed ML, and offline backtests"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser(
        "doctor", help="validate environment, profile, and safety locks"
    )
    doctor_parser.add_argument(
        "--environment",
        default=os.getenv("TRADING_AI_ENV", "PAPER"),
        help="DEV, TEST, PAPER, or LIVE (default: PAPER)",
    )
    doctor_parser.add_argument(
        "--profile",
        default=os.getenv("TRADING_AI_PROFILE", "balanced"),
        help="balanced or aggressive (default: balanced)",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON",
    )

    data_parser = commands.add_parser("data", help="manage historical market data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)

    fetch_parser = data_commands.add_parser(
        "fetch", help="fetch or reuse one configured historical dataset"
    )
    fetch_parser.add_argument(
        "--profile",
        default=os.getenv("TRADING_AI_PROFILE", "balanced"),
        help="enabled profile providing the allowed universe",
    )
    selection = fetch_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--symbol", help="one symbol from the selected profile")
    selection.add_argument(
        "--all",
        action="store_true",
        dest="all_symbols",
        help="explicitly fetch every symbol in the selected profile",
    )
    fetch_parser.add_argument(
        "--timeframe", required=True, choices=("1h", "4h", "1d")
    )
    fetch_parser.add_argument("--start", required=True, type=_parse_datetime)
    fetch_parser.add_argument("--end", required=True, type=_parse_datetime)
    fetch_parser.add_argument(
        "--cache-mode",
        choices=tuple(mode.value for mode in CacheMode),
        default=CacheMode.CACHE_FIRST.value,
        help="CACHE_ONLY, CACHE_FIRST, or REFRESH",
    )
    _add_store_arguments(fetch_parser)

    for command_name in ("validate", "inspect"):
        command_parser = data_commands.add_parser(
            command_name, help=f"{command_name} the latest cached dataset"
        )
        command_parser.add_argument("--symbol", required=True)
        command_parser.add_argument(
            "--timeframe", required=True, choices=("1h", "4h", "1d")
        )
        _add_store_arguments(command_parser)

    backtest_parser = commands.add_parser(
        "backtest", help="run or inspect deterministic offline simulations"
    )
    backtest_commands = backtest_parser.add_subparsers(
        dest="backtest_command", required=True
    )
    run_parser = backtest_commands.add_parser(
        "run", help="run one technical demo or Lot 3 baseline from exact cached data"
    )
    run_parser.add_argument(
        "--strategy",
        choices=("buy-and-hold", *BASELINE_STRATEGIES.names),
        action="append",
        help=(
            "repeat for one shared multi-strategy portfolio; defaults to the "
            "buy-and-hold technical demo"
        ),
    )
    run_parser.add_argument(
        "--symbol",
        required=True,
        action="append",
        help="configured symbol; repeat for a multi-asset momentum run",
    )
    run_parser.add_argument(
        "--timeframe", required=True, choices=("1h", "4h", "1d")
    )
    run_parser.add_argument("--start", required=True, type=_parse_datetime)
    run_parser.add_argument("--end", required=True, type=_parse_datetime)
    run_parser.add_argument(
        "--profile", default=os.getenv("TRADING_AI_PROFILE", "balanced")
    )
    run_parser.add_argument(
        "--environment", default=os.getenv("TRADING_AI_ENV", "PAPER")
    )
    run_parser.add_argument("--quantity", type=_parse_decimal, default=Decimal("1"))
    run_parser.add_argument("--allocation-fraction", type=_parse_decimal)
    run_parser.add_argument("--fast-window", type=int)
    run_parser.add_argument("--slow-window", type=int)
    run_parser.add_argument("--slope-lookback", type=int)
    run_parser.add_argument("--momentum-lookback", type=int)
    run_parser.add_argument("--top-k", type=int)
    run_parser.add_argument("--rebalance-every", type=int)
    run_parser.add_argument("--minimum-return", type=_parse_decimal)
    run_parser.add_argument("--entry-window", type=int)
    run_parser.add_argument("--exit-window", type=int)
    run_parser.add_argument("--mean-reversion-lookback", type=int)
    run_parser.add_argument("--entry-zscore", type=_parse_decimal)
    run_parser.add_argument("--exit-zscore", type=_parse_decimal)
    run_parser.add_argument(
        "--starting-cash", type=_parse_decimal, default=Decimal("100000")
    )
    run_parser.add_argument(
        "--spread-bps", type=_parse_decimal, default=Decimal("0")
    )
    run_parser.add_argument(
        "--slippage-bps", type=_parse_decimal, default=Decimal("0")
    )
    run_parser.add_argument(
        "--commission-fixed", type=_parse_decimal, default=Decimal("0")
    )
    run_parser.add_argument(
        "--commission-bps", type=_parse_decimal, default=Decimal("0")
    )
    run_parser.add_argument(
        "--commission-minimum", type=_parse_decimal, default=Decimal("0")
    )
    run_parser.add_argument(
        "--cost-profile",
        choices=("ibkr_pro_fixed", "ibkr_pro_tiered"),
        default="ibkr_pro_fixed",
        help="explicit dated transaction-cost tariff (default: ibkr_pro_fixed)",
    )
    run_parser.add_argument(
        "--benchmark-symbol",
        help="optional Buy & Hold benchmark; defaults to the selected symbol",
    )
    run_parser.add_argument(
        "--allow-data-warnings",
        action="store_true",
        help="accept DataQuality WARNING datasets; FAIL is always rejected",
    )
    run_parser.add_argument(
        "--ml-mode",
        choices=("disabled", "score-only", "filter"),
        default="disabled",
        help="disabled (default), score-only, or APPROVED-model filter",
    )
    run_parser.add_argument(
        "--ml-model-id",
        action="append",
        help=(
            "explicit model ID; for multiple strategies use strategy=model-id "
            "once per strategy (no latest-model fallback)"
        ),
    )
    run_parser.add_argument(
        "--ml-threshold",
        type=float,
        default=0.55,
        help="fixed FILTER probability threshold (default: 0.55)",
    )
    _add_store_arguments(run_parser)

    inspect_backtest_parser = backtest_commands.add_parser(
        "inspect", help="verify hashes and display an exported backtest summary"
    )
    inspect_backtest_parser.add_argument("--run-id", required=True)
    _add_store_arguments(inspect_backtest_parser)

    strategy_parser = commands.add_parser(
        "strategy", help="inspect available research baselines"
    )
    strategy_commands = strategy_parser.add_subparsers(
        dest="strategy_command", required=True
    )
    strategy_list = strategy_commands.add_parser(
        "list", help="list baseline versions and non-optimized defaults"
    )
    strategy_list.add_argument("--json", action="store_true", dest="as_json")

    risk_parser = commands.add_parser(
        "risk", help="inspect offline risk configuration and safety limits"
    )
    risk_commands = risk_parser.add_subparsers(
        dest="risk_command", required=True
    )
    risk_inspect = risk_commands.add_parser(
        "inspect", help="validate and display one profile's risk configuration"
    )
    risk_inspect.add_argument(
        "--profile", default=os.getenv("TRADING_AI_PROFILE", "balanced")
    )
    risk_inspect.add_argument("--json", action="store_true", dest="as_json")

    regime_parser = commands.add_parser(
        "regime", help="inspect cached-data regimes and activation policy offline"
    )
    regime_commands = regime_parser.add_subparsers(
        dest="regime_command", required=True
    )
    regime_inspect = regime_commands.add_parser(
        "inspect", help="classify one exact cached dataset without downloading"
    )
    regime_inspect.add_argument(
        "--profile", default=os.getenv("TRADING_AI_PROFILE", "balanced")
    )
    regime_inspect.add_argument("--symbol", required=True)
    regime_inspect.add_argument(
        "--timeframe", required=True, choices=("1h", "4h", "1d")
    )
    regime_inspect.add_argument("--start", required=True, type=_parse_datetime)
    regime_inspect.add_argument("--end", required=True, type=_parse_datetime)
    _add_store_arguments(regime_inspect)

    regime_latest = regime_commands.add_parser(
        "latest", help="display the latest regime from the latest cached dataset"
    )
    regime_latest.add_argument(
        "--profile", default=os.getenv("TRADING_AI_PROFILE", "balanced")
    )
    regime_latest.add_argument("--symbol", required=True)
    regime_latest.add_argument(
        "--timeframe", required=True, choices=("1h", "4h", "1d")
    )
    _add_store_arguments(regime_latest)

    regime_policy = regime_commands.add_parser(
        "policy", help="display the configuration-driven Balanced policy matrix"
    )
    regime_policy.add_argument(
        "--profile", default=os.getenv("TRADING_AI_PROFILE", "balanced")
    )
    regime_policy.add_argument("--json", action="store_true", dest="as_json")

    ml_parser = commands.add_parser(
        "ml", help="train, inspect, and explicitly promote local statistical models"
    )
    ml_commands = ml_parser.add_subparsers(dest="ml_command", required=True)
    ml_train = ml_commands.add_parser(
        "train", help="train one fixed tabular baseline from exact cached data"
    )
    ml_train.add_argument("--strategy", required=True, choices=BASELINE_STRATEGIES.names)
    ml_train.add_argument("--timeframe", required=True, choices=("1h", "4h", "1d"))
    ml_train.add_argument(
        "--model", required=True, choices=tuple(item.value for item in ModelFamily)
    )
    ml_train.add_argument(
        "--symbol",
        action="append",
        help="configured symbol; repeat for multi-asset training (default: profile universe)",
    )
    ml_train.add_argument(
        "--profile", default=os.getenv("TRADING_AI_PROFILE", "balanced")
    )
    for name in (
        "train-start", "train-end", "validation-start", "validation-end",
        "test-start", "test-end",
    ):
        ml_train.add_argument(f"--{name}", required=True, type=_parse_datetime)
    ml_train.add_argument("--horizon-bars", type=int, default=5)
    ml_train.add_argument(
        "--minimum-forward-return-bps", type=_parse_decimal, default=Decimal("0")
    )
    ml_train.add_argument("--embargo-bars", type=int, default=1)
    ml_train.add_argument("--walk-forward-folds", type=int, default=3)
    ml_train.add_argument("--minimum-training-samples", type=int, default=100)
    ml_train.add_argument("--minimum-samples-per-class", type=int, default=20)
    _add_store_arguments(ml_train)

    ml_evaluate = ml_commands.add_parser(
        "evaluate", help="inspect stored temporal validation and final-test metrics"
    )
    ml_evaluate.add_argument("--model-id", required=True)
    _add_store_arguments(ml_evaluate)

    model_parser = ml_commands.add_parser("model", help="manage the local model registry")
    model_commands = model_parser.add_subparsers(dest="model_command", required=True)
    model_list = model_commands.add_parser("list", help="list registered model artifacts")
    _add_store_arguments(model_list)
    model_inspect = model_commands.add_parser("inspect", help="verify and inspect one model")
    model_inspect.add_argument("--model-id", required=True)
    _add_store_arguments(model_inspect)
    model_promote = model_commands.add_parser(
        "promote", help="perform one explicit audited lifecycle transition"
    )
    model_promote.add_argument("--model-id", required=True)
    model_promote.add_argument(
        "--to", required=True, choices=("VALIDATED", "APPROVED", "RETIRED")
    )
    model_promote.add_argument("--reason", required=True)
    _add_store_arguments(model_promote)
    model_rollback = model_commands.add_parser(
        "rollback", help="explicitly restore the prior APPROVED alias"
    )
    model_rollback.add_argument("--strategy", required=True, choices=BASELINE_STRATEGIES.names)
    model_rollback.add_argument("--timeframe", required=True, choices=("1h", "4h", "1d"))
    model_rollback.add_argument("--reason", required=True)
    _add_store_arguments(model_rollback)

    portfolio_parser = commands.add_parser(
        "portfolio", help="inspect offline Balanced portfolio construction"
    )
    portfolio_commands = portfolio_parser.add_subparsers(
        dest="portfolio_command", required=True
    )
    portfolio_inspect = portfolio_commands.add_parser(
        "inspect", help="validate fixed sleeves, caps, turnover, and currency policy"
    )
    portfolio_inspect.add_argument(
        "--profile", default=os.getenv("TRADING_AI_PROFILE", "balanced")
    )
    portfolio_inspect.add_argument("--json", action="store_true", dest="as_json")

    costs_parser = commands.add_parser(
        "costs", help="inspect and estimate configuration-driven transaction economics"
    )
    costs_commands = costs_parser.add_subparsers(dest="costs_command", required=True)
    for name in ("inspect", "verify-config"):
        item = costs_commands.add_parser(name, help=f"{name} one offline cost profile")
        item.add_argument("--profile", default="balanced")
        item.add_argument(
            "--cost-profile",
            choices=("ibkr_pro_fixed", "ibkr_pro_tiered"),
            default="ibkr_pro_fixed",
        )
        item.add_argument("--json", action="store_true", dest="as_json")
    costs_estimate = costs_commands.add_parser(
        "estimate", help="estimate one point-in-time order without future prices"
    )
    costs_estimate.add_argument("--profile", default="balanced")
    costs_estimate.add_argument("--cost-profile", choices=("ibkr_pro_fixed", "ibkr_pro_tiered"), default="ibkr_pro_fixed")
    costs_estimate.add_argument("--symbol", required=True)
    costs_estimate.add_argument("--side", choices=("BUY", "SELL"), required=True)
    costs_estimate.add_argument("--quantity", type=_parse_decimal, required=True)
    costs_estimate.add_argument("--price", type=_parse_decimal, required=True)
    costs_estimate.add_argument("--timeframe", choices=("1h", "4h", "1d"), required=True)
    costs_estimate.add_argument("--timestamp", type=_parse_datetime, required=True)
    costs_estimate.add_argument("--spread-bps", type=_parse_decimal, default=Decimal("0"))
    costs_estimate.add_argument("--slippage-bps", type=_parse_decimal, default=Decimal("0"))
    costs_estimate.add_argument("--json", action="store_true", dest="as_json")

    validation_parser = commands.add_parser(
        "validation", help="run or inspect the offline research validation gate"
    )
    validation_commands = validation_parser.add_subparsers(
        dest="validation_command", required=True
    )
    validation_run = validation_commands.add_parser(
        "run", help="validate one checksum-verified schema 1.6 backtest export"
    )
    validation_run.add_argument("--run-id", required=True)
    validation_run.add_argument("--final-oos-confirmed", action="store_true")
    validation_run.add_argument(
        "--no-training-edge-overlap-confirmed", action="store_true"
    )
    source_kind = validation_run.add_mutually_exclusive_group()
    source_kind.add_argument("--real-data", action="store_true")
    source_kind.add_argument("--synthetic-mechanics-only", action="store_true")
    _add_store_arguments(validation_run)
    validation_inspect = validation_commands.add_parser(
        "inspect", help="inspect a stored immutable validation report"
    )
    validation_inspect.add_argument("--validation-id", required=True)
    _add_store_arguments(validation_inspect)
    validation_holdout = validation_commands.add_parser(
        "holdout-status", help="inspect the frozen V2 final-holdout lifecycle"
    )
    _add_store_arguments(validation_holdout)
    validation_readiness = validation_commands.add_parser(
        "paper-readiness", help="inspect a read-only Paper readiness review"
    )
    validation_readiness.add_argument("--version", choices=("1", "2"), default="1")
    validation_readiness.add_argument("--report-id")
    validation_readiness.add_argument("--reassessment-id")
    _add_store_arguments(validation_readiness)
    validation_reassess = validation_commands.add_parser(
        "reassess-evidence",
        help="reassess a consumed holdout using offline evidence without freshening it",
    )
    validation_reassess.add_argument("--run-id", required=True)
    validation_reassess.add_argument("--report-id", required=True)
    validation_reassess.add_argument("--candidate-run-id")
    validation_reassess.add_argument(
        "--operating-scenario", default="PAPER_ESTIMATE_V1"
    )
    _add_store_arguments(validation_reassess)

    robustness_parser = commands.add_parser(
        "robustness", help="run frozen Lot 8.2 diagnostics without retuning"
    )
    robustness_commands = robustness_parser.add_subparsers(
        dest="robustness_command", required=True
    )
    robustness_plan = robustness_commands.add_parser(
        "plan", help="freeze and inspect the predeclared research plan"
    )
    _add_store_arguments(robustness_plan)
    robustness_run = robustness_commands.add_parser(
        "run", help="analyze one checksum-verified backtest export"
    )
    robustness_run.add_argument("--run-id", required=True)
    robustness_run.add_argument(
        "--period-classification",
        choices=("CONSUMED_DIAGNOSTIC", "DIAGNOSTIC", "FINAL_HOLDOUT"),
        required=True,
    )
    robustness_run.add_argument(
        "--without-symbol",
        action="append",
        default=[],
        metavar="SYMBOL=RUN_ID",
        help="attach one precomputed post-hoc leave-one-symbol-out run",
    )
    robustness_run.add_argument(
        "--without-strategy",
        action="append",
        default=[],
        metavar="STRATEGY=RUN_ID",
        help="attach one precomputed post-hoc leave-one-strategy-out run",
    )
    robustness_run.add_argument(
        "--single-strategy",
        action="append",
        default=[],
        metavar="STRATEGY=RUN_ID",
        help="attach one precomputed frozen single-strategy comparison run",
    )
    _add_store_arguments(robustness_run)
    robustness_inspect = robustness_commands.add_parser(
        "inspect", help="verify and inspect a stored robustness report"
    )
    robustness_inspect.add_argument("--report-id", required=True)
    _add_store_arguments(robustness_inspect)
    robustness_compare = robustness_commands.add_parser(
        "compare", help="report precomputed leave-one-out runs without selection"
    )
    robustness_compare.add_argument("--run-id", required=True)
    robustness_compare.add_argument("--without-symbol", action="append", default=[])
    robustness_compare.add_argument("--without-strategy", action="append", default=[])
    robustness_compare.add_argument("--single-strategy", action="append", default=[])
    _add_store_arguments(robustness_compare)

    evidence_parser = commands.add_parser(
        "evidence", help="inspect dated official evidence and immutable reassessments"
    )
    evidence_commands = evidence_parser.add_subparsers(
        dest="evidence_command", required=True
    )
    evidence_verify = evidence_commands.add_parser(
        "verify", help="verify the offline Evidence Registry V2"
    )
    _add_store_arguments(evidence_verify)
    evidence_inspect = evidence_commands.add_parser(
        "inspect", help="inspect one checksum-verified schema 1.8 reassessment"
    )
    evidence_inspect.add_argument("--reassessment-id", required=True)
    _add_store_arguments(evidence_inspect)
    evidence_compare = evidence_commands.add_parser(
        "compare", help="compare frozen holdout assumptions with dated evidence"
    )
    evidence_compare.add_argument("--run-id", required=True)
    evidence_compare.add_argument("--report-id", required=True)
    evidence_compare.add_argument("--candidate-run-id")
    evidence_compare.add_argument(
        "--operating-scenario", default="PAPER_ESTIMATE_V1"
    )
    _add_store_arguments(evidence_compare)

    dashboard_parser = commands.add_parser(
        "dashboard", help="serve or inspect the local read-only observability UI"
    )
    dashboard_commands = dashboard_parser.add_subparsers(
        dest="dashboard_command", required=True
    )
    dashboard_serve = dashboard_commands.add_parser(
        "serve", help="serve the local-only Dashboard"
    )
    dashboard_serve.add_argument("--host", default="127.0.0.1")
    dashboard_serve.add_argument("--port", type=int, default=8080)
    dashboard_serve.add_argument("--data-root", type=Path, default=Path("data_local"))
    dashboard_inspect = dashboard_commands.add_parser(
        "inspect", help="verify and inspect one run through monitoring view models"
    )
    dashboard_inspect.add_argument("--run-id", required=True)
    _add_store_arguments(dashboard_inspect)

    monitoring_parser = commands.add_parser(
        "monitoring", help="inspect local observability health"
    )
    monitoring_commands = monitoring_parser.add_subparsers(
        dest="monitoring_command", required=True
    )
    monitoring_health = monitoring_commands.add_parser(
        "health", help="inspect the local monitoring source and store"
    )
    monitoring_health.add_argument("--run-id")
    _add_store_arguments(monitoring_health)
    return parser


def format_report(report: HealthReport, as_json: bool) -> str:
    if as_json:
        return json.dumps(report.to_dict(), sort_keys=True)
    return "\n".join(
        (
            f"status: {report.status}",
            f"environment: {report.environment}",
            f"profile: {report.profile}",
            f"profile_enabled: {str(report.profile_enabled).lower()}",
            f"live_allowed: {str(report.live_allowed).lower()}",
            f"configuration_valid: {str(report.configuration_valid).lower()}",
            f"message: {report.message}",
        )
    )


def _fetch_payload(result: DataFetchResult) -> dict[str, Any]:
    return {
        "dataset_id": result.manifest.dataset_id,
        "symbol": result.manifest.symbol,
        "timeframe": result.manifest.timeframe,
        "row_count": result.manifest.row_count,
        "actual_start": (
            result.manifest.actual_start.isoformat()
            if result.manifest.actual_start is not None
            else None
        ),
        "actual_end": (
            result.manifest.actual_end.isoformat()
            if result.manifest.actual_end is not None
            else None
        ),
        "cache_hit": result.cache_hit,
        "quality_status": result.quality_report.quality_status.value,
        "corporate_action_count": len(result.corporate_actions),
        "file_path": result.manifest.file_path,
        "checksum_sha256": result.manifest.checksum_sha256,
    }


def _render_payload(payload: Any, as_json: bool) -> str:
    if as_json:
        return json.dumps(payload, indent=2, sort_keys=True)
    if isinstance(payload, list):
        return "\n\n".join(_render_payload(item, False) for item in payload)
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


def _build_data_engine(data_root: Path) -> DataEngine:
    return DataEngine(
        provider=YahooFinanceProvider(),
        store=ParquetDataStore(data_root),
    )


def _run_data(args: argparse.Namespace) -> int:
    engine = _build_data_engine(args.data_root)
    if args.data_command == "fetch":
        cache_mode = CacheMode(args.cache_mode)
        if args.all_symbols:
            results = engine.fetch_profile_universe(
                profile_name=args.profile,
                timeframe=args.timeframe,
                start=args.start,
                end=args.end,
                cache_mode=cache_mode,
            )
        else:
            results = (
                engine.fetch(
                    profile_name=args.profile,
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    start=args.start,
                    end=args.end,
                    cache_mode=cache_mode,
                ),
            )
        print(_render_payload([_fetch_payload(item) for item in results], args.as_json))
        return 0
    inspection: DatasetInspection
    if args.data_command == "validate":
        inspection = engine.validate_cached(args.symbol, args.timeframe)
    elif args.data_command == "inspect":
        inspection = engine.inspect_cached(args.symbol, args.timeframe)
    else:
        raise AssertionError(f"unhandled data command: {args.data_command}")
    payload = inspection.to_dict()
    if args.data_command == "validate":
        payload = {
            "integrity_valid": inspection.integrity_valid,
            "quality_report": inspection.quality_report.to_dict(),
        }
    print(_render_payload(payload, args.as_json))
    return 0


def _option(value: Any, default: Any) -> Any:
    return default if value is None else value


def _trend_cli_config(args: argparse.Namespace) -> TrendConfig:
    defaults = TrendConfig()
    return TrendConfig(
        fast_window=_option(args.fast_window, defaults.fast_window),
        slow_window=_option(args.slow_window, defaults.slow_window),
        slope_lookback=_option(args.slope_lookback, defaults.slope_lookback),
        allocation_fraction=_option(
            args.allocation_fraction, defaults.allocation_fraction
        ),
    )


def _momentum_cli_config(args: argparse.Namespace) -> MomentumConfig:
    defaults = MomentumConfig()
    return MomentumConfig(
        lookback=_option(args.momentum_lookback, defaults.lookback),
        top_k=_option(args.top_k, defaults.top_k),
        rebalance_every=_option(args.rebalance_every, defaults.rebalance_every),
        allocation_fraction=_option(
            args.allocation_fraction, defaults.allocation_fraction
        ),
        minimum_return=_option(args.minimum_return, defaults.minimum_return),
    )


def _breakout_cli_config(args: argparse.Namespace) -> BreakoutConfig:
    defaults = BreakoutConfig()
    return BreakoutConfig(
        entry_window=_option(args.entry_window, defaults.entry_window),
        exit_window=_option(args.exit_window, defaults.exit_window),
        allocation_fraction=_option(
            args.allocation_fraction, defaults.allocation_fraction
        ),
    )


def _mean_reversion_cli_config(args: argparse.Namespace) -> MeanReversionConfig:
    defaults = MeanReversionConfig()
    return MeanReversionConfig(
        lookback=_option(
            args.mean_reversion_lookback,
            defaults.lookback,
        ),
        entry_zscore=_option(args.entry_zscore, defaults.entry_zscore),
        exit_zscore=_option(args.exit_zscore, defaults.exit_zscore),
        allocation_fraction=_option(
            args.allocation_fraction, defaults.allocation_fraction
        ),
    )


_CLI_CONFIG_BUILDERS = {
    "trend": _trend_cli_config,
    "momentum": _momentum_cli_config,
    "breakout": _breakout_cli_config,
    "mean-reversion": _mean_reversion_cli_config,
}


def _run_strategy(args: argparse.Namespace) -> int:
    if args.strategy_command != "list":
        raise AssertionError(f"unhandled strategy command: {args.strategy_command}")
    payload = [
        {
            "name": descriptor.name,
            "version": descriptor.version,
            "description": descriptor.description,
            "default_parameters": dict(descriptor.default_parameters),
            "warning": "research baseline; defaults are not optimized",
        }
        for descriptor in BASELINE_STRATEGIES.descriptors
    ]
    payload.append(
        {
            "name": "buy-and-hold",
            "version": "technical-demo",
            "description": "Technical demo and benchmark compatibility command",
            "default_parameters": {"quantity": "1"},
            "warning": "not a Lot 3 quantitative strategy",
        }
    )
    print(_render_payload(payload, args.as_json))
    return 0


def _run_risk(args: argparse.Namespace) -> int:
    if args.risk_command != "inspect":
        raise AssertionError(f"unhandled risk command: {args.risk_command}")
    profile = inspect_profile(args.profile)
    config = inspect_risk_config(profile.name)
    groups = load_asset_groups()
    locked = profile.name.value == "aggressive" or not profile.enabled or not config.enabled
    if not locked:
        config, groups = load_balanced_risk_config(profile)
    payload = {
        "engine": config.engine_name,
        "version": config.engine_version,
        "enabled": config.enabled,
        "profile": profile.name.value,
        "profile_enabled": profile.enabled,
        "locked": locked,
        "config_hash": risk_config_hash(config, groups),
        "limits": {
            "max_positions": config.max_positions,
            "max_portfolio_exposure": str(config.max_portfolio_exposure),
            "max_single_position_exposure": str(
                config.max_single_position_exposure
            ),
            "max_group_exposure": str(config.max_group_exposure),
            "max_trade_risk_fraction": str(config.max_trade_risk_fraction),
        },
        "drawdown": {
            "soft_limit": str(config.soft_drawdown_limit),
            "hard_limit": str(config.hard_drawdown_limit),
            "reduced_multiplier": str(config.reduced_risk_multiplier),
        },
        "daily_loss": {
            "limit": str(config.daily_loss_limit),
            "risk_day_timezone": config.risk_day_timezone,
        },
        "correlation": {
            "threshold": str(config.high_correlation_threshold),
            "max_correlated_exposure": str(
                config.max_highly_correlated_exposure
            ),
            "minimum_observations": config.correlation_min_observations,
            "unknown_policy": config.correlation_unknown_policy.value,
        },
        "volatility": {
            "feature": config.volatility_feature_name,
            "missing_policy": config.missing_volatility_policy.value,
            "thresholds": {
                item.timeframe: {
                    "elevated": str(item.elevated),
                    "extreme": str(item.extreme),
                }
                for item in config.volatility_thresholds
            },
        },
        "asset_groups": {
            name: list(symbols) for name, symbols in groups.groups
        },
        "warning": (
            "research defaults reduce exposure; they do not guarantee safety or profitability"
        ),
    }
    print(_render_payload(payload, args.as_json))
    return 0


def _run_portfolio(args: argparse.Namespace) -> int:
    if args.portfolio_command != "inspect":
        raise AssertionError(f"unhandled portfolio command: {args.portfolio_command}")
    profile = inspect_profile(args.profile)
    config = inspect_portfolio_config(profile.name)
    currencies = load_asset_currencies()
    locked = profile.name.value == "aggressive" or not profile.enabled or not config.enabled
    if not locked:
        risk, _ = load_balanced_risk_config(profile)
        config, currencies = load_balanced_portfolio_config(profile, risk)
    payload = {
        "engine": config.engine_name,
        "version": config.engine_version,
        "enabled": config.enabled,
        "profile": profile.name.value,
        "profile_enabled": profile.enabled,
        "locked": locked,
        "config_hash": portfolio_config_hash(config, currencies),
        "base_currency": config.base_currency,
        "limits": {
            "max_target_exposure": str(config.max_target_exposure),
            "max_target_per_symbol": str(config.max_target_per_symbol),
            "max_unique_positions": config.max_unique_positions,
            "min_cash_fraction": str(config.min_cash_fraction),
        },
        "construction": {
            "allocation": "equal-weight-within-fixed-strategy-sleeve",
            "min_rebalance_weight": str(config.min_rebalance_weight),
            "max_entry_turnover_per_cycle": str(
                config.max_entry_turnover_per_cycle
            ),
            "soft_correlation_threshold": str(config.soft_correlation_threshold),
            "correlation_min_observations": config.correlation_min_observations,
            "unknown_correlation_policy": config.unknown_correlation_policy.value,
            "mixed_currency_policy": config.mixed_currency_policy.value,
        },
        "sleeves": {
            item.strategy_name: str(item.budget_weight)
            for item in config.strategy_sleeves
        },
        "unused_budget_policy": "CASH",
        "risk_authority": "BalancedRiskEngine",
        "network_access": False,
        "warning": (
            "Portfolio construction can diversify exposure but cannot eliminate "
            "market risk or guarantee profitability."
        ),
    }
    print(_render_payload(payload, args.as_json))
    return 0


def _run_costs(args: argparse.Namespace) -> int:
    profile = inspect_profile(args.profile)
    if args.costs_command in {"inspect", "verify-config"}:
        config = inspect_cost_config(
            profile.name, tariff_profile=args.cost_profile
        )
        payload: dict[str, Any] = {
            "profile": profile.name.value,
            "profile_enabled": profile.enabled,
            "enabled": config.enabled,
            "engine": config.engine_name,
            "version": config.engine_version,
            "tariff_profile": config.tariff_profile,
            "base_currency": config.base_currency,
            "cash_buffer_bps": str(config.cash_buffer_bps),
            "cash_buffer_absolute": str(config.cash_buffer_absolute),
            "minimum_net_edge_bps": str(config.minimum_net_edge_bps),
            "minimum_edge_to_cost_ratio": str(config.minimum_edge_to_cost_ratio),
            "locked": profile.name.value == "aggressive" or not config.enabled,
        }
        if profile.name.value == "balanced" and profile.enabled and config.enabled:
            bundle = load_balanced_cost_config(
                profile, tariff_profile=args.cost_profile
            )
            payload.update(
                {
                    "config_hash": bundle.config_hash,
                    "tariff_status": bundle.tariff.status.value,
                    "tariff_effective_from": bundle.tariff.effective_from.isoformat(),
                    "tariff_effective_to": (
                        bundle.tariff.effective_to.isoformat()
                        if bundle.tariff.effective_to else None
                    ),
                    "tariff_source": bundle.tariff.source_reference,
                    "tariff_config_hash": bundle.tariff.config_hash,
                    "instrument_metadata_count": len(bundle.instruments),
                    "tax_rule_count": len(bundle.taxes),
                    "critical_variable_components": list(
                        bundle.config.critical_variable_components
                    ),
                }
            )
            evidence = load_historical_cost_evidence()
            payload["historical_evidence"] = {
                "registry_hash": evidence.registry_hash,
                "broker_tariffs": [
                    {
                        "profile": item.subject,
                        "status": item.status.value,
                        "evidence_kind": item.evidence_kind.value,
                        "source": item.source_reference,
                        "warning": item.warning,
                    }
                    for item in evidence.broker_tariffs
                ],
                "tax_rate_periods": len(evidence.tax_rates),
                "annual_tax_eligibility_records": len(evidence.tax_eligibility),
                "exchange_fees": evidence.exchange_fee_status.value,
                "fx_cost": evidence.fx_cost_status.value,
                "operating_scenarios": [name for name, _ in evidence.operating_scenarios],
            }
        print(_render_payload(payload, args.as_json))
        return 0
    if args.costs_command == "estimate":
        settings = load_runtime_settings("PAPER", args.profile)
        engine = BalancedTransactionCostEngine.from_profile(
            settings.profile, tariff_profile=args.cost_profile
        )
        estimate = engine.estimate(
            PreTradeCostRequest(
                timestamp=args.timestamp,
                symbol=args.symbol,
                side=OrderSide(args.side),
                quantity=args.quantity,
                reference_price=args.price,
                timeframe=args.timeframe,
                spread_bps=args.spread_bps,
                slippage_bps=args.slippage_bps,
                order_id="cli-cost-estimate",
            )
        )
        print(_render_payload(to_primitive(estimate), args.as_json))
        return 0
    raise AssertionError(f"unhandled costs command: {args.costs_command}")


def _run_validation(args: argparse.Namespace) -> int:
    store = LocalValidationStore(args.data_root / "validation")
    if args.validation_command == "holdout-status":
        print(
            _render_payload(
                RobustnessService(args.data_root).holdout_status(), args.as_json
            )
        )
        return 0
    if args.validation_command == "paper-readiness":
        if args.version == "2":
            if not args.reassessment_id or args.report_id:
                raise ValueError(
                    "Paper Readiness V2 requires --reassessment-id and no --report-id"
                )
            payload = EvidenceClosureService(args.data_root).inspect(
                args.reassessment_id
            )["paper_readiness_v2"]
        else:
            if not args.report_id or args.reassessment_id:
                raise ValueError(
                    "Paper Readiness V1 requires --report-id and no --reassessment-id"
                )
            payload = RobustnessService(args.data_root).inspect(args.report_id)[
                "paper_readiness"
            ]
        print(_render_payload(payload, args.as_json))
        return 0
    if args.validation_command == "reassess-evidence":
        reassessment, readiness = EvidenceClosureService(args.data_root).reassess(
            run_id=args.run_id,
            robustness_report_id=args.report_id,
            candidate_run_id=args.candidate_run_id,
            operating_scenario_id=args.operating_scenario,
        )
        print(
            _render_payload(
                {
                    "reassessment": to_primitive(reassessment),
                    "paper_readiness_v2": to_primitive(readiness),
                },
                args.as_json,
            )
        )
        return 0
    if args.validation_command == "inspect":
        print(_render_payload(store.inspect(args.validation_id), args.as_json))
        return 0
    if args.validation_command == "run":
        from trading_ai.monitoring.source import BacktestMonitoringSource

        data = BacktestMonitoringSource(args.data_root / "backtests").load_run(
            args.run_id
        )
        report = ResearchValidationGate().evaluate_export(
            summary=data.summary,
            tables=data.tables,
            integrity_verified=data.integrity_verified,
            final_oos=args.final_oos_confirmed,
            no_training_or_edge_overlap_confirmed=(
                args.no_training_edge_overlap_confirmed
            ),
            real_data_available=args.real_data,
            synthetic_mechanics_only=args.synthetic_mechanics_only,
        )
        path = store.save(report)
        payload = to_primitive(report)
        payload["report_path"] = str(path)
        print(_render_payload(payload, args.as_json))
        return 0 if report.status.value in {"PASS", "WARNING"} else 3
    raise AssertionError(
        f"unhandled validation command: {args.validation_command}"
    )


def _parse_run_mapping(values: Sequence[str], name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{name} values must use ITEM=RUN_ID")
        item, run_id = raw.split("=", 1)
        if not item or not run_id or item in result:
            raise ValueError(f"invalid or duplicate {name} value")
        result[item] = run_id
    return result


def _run_robustness(args: argparse.Namespace) -> int:
    service = RobustnessService(args.data_root)
    if args.robustness_command == "plan":
        print(_render_payload(service.freeze_plan(), args.as_json))
        return 0
    if args.robustness_command == "inspect":
        print(_render_payload(service.inspect(args.report_id), args.as_json))
        return 0
    if args.robustness_command in {"run", "compare"}:
        classification = (
            PeriodClassification(args.period_classification)
            if args.robustness_command == "run"
            else PeriodClassification.DIAGNOSTIC
        )
        payload = service.run(
            args.run_id,
            period_classification=classification,
            leave_one_symbol_run_ids=_parse_run_mapping(
                args.without_symbol, "--without-symbol"
            ),
            leave_one_strategy_run_ids=_parse_run_mapping(
                args.without_strategy, "--without-strategy"
            ),
            single_strategy_run_ids=_parse_run_mapping(
                args.single_strategy, "--single-strategy"
            ),
        )
        print(_render_payload(payload, args.as_json))
        return 0
    raise AssertionError(f"unhandled robustness command: {args.robustness_command}")


def _run_evidence(args: argparse.Namespace) -> int:
    service = EvidenceClosureService(args.data_root)
    if args.evidence_command == "verify":
        print(_render_payload(service.verify_registry(), args.as_json))
        return 0
    if args.evidence_command == "inspect":
        print(_render_payload(service.inspect(args.reassessment_id), args.as_json))
        return 0
    if args.evidence_command == "compare":
        reassessment, readiness = service.reassess(
            run_id=args.run_id,
            robustness_report_id=args.report_id,
            candidate_run_id=args.candidate_run_id,
            operating_scenario_id=args.operating_scenario,
        )
        print(
            _render_payload(
                {
                    "reassessment": to_primitive(reassessment),
                    "paper_readiness_v2": to_primitive(readiness),
                },
                args.as_json,
            )
        )
        return 0
    raise AssertionError(f"unhandled evidence command: {args.evidence_command}")


def _regime_snapshot_payload(snapshot) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "symbol": snapshot.symbol,
        "timestamp": snapshot.timestamp.isoformat(),
        "timeframe": snapshot.timeframe,
        "structure_regime": snapshot.structure_regime.value,
        "volatility_regime": snapshot.volatility_regime.value,
        "bars_in_current_structure_regime": (
            snapshot.bars_in_current_structure_regime
        ),
        "candidate_structure_regime": snapshot.candidate_structure_regime.value,
        "confirmation_progress": snapshot.confirmation_progress,
        "reason_codes": list(snapshot.reason_codes),
        "evidence": dict(snapshot.evidence),
    }


def _run_regime(args: argparse.Namespace) -> int:
    if args.regime_command == "policy":
        profile = inspect_profile(args.profile)
        config = inspect_strategy_policy_config(profile.name)
        locked = (
            profile.name.value == "aggressive"
            or not profile.enabled
            or not config.enabled
        )
        if not locked:
            config = load_balanced_strategy_policy_config(profile)
        matrix: dict[str, dict[str, dict[str, str]]] = {}
        for rule in config.structure_rules:
            matrix.setdefault(rule.structure.value, {})[rule.strategy_name] = {
                "status": rule.status.value,
                "multiplier": str(rule.multiplier),
            }
        overlays: dict[str, dict[str, dict[str, str]]] = {}
        for overlay in config.volatility_overlays:
            overlays.setdefault(overlay.volatility.value, {})[
                overlay.strategy_name
            ] = {
                "status": overlay.status.value,
                "multiplier": str(overlay.multiplier),
            }
        payload = {
            "profile": profile.name.value,
            "profile_enabled": profile.enabled,
            "enabled": config.enabled,
            "locked": locked,
            "policy_name": config.policy_name,
            "policy_version": config.policy_version,
            "config_hash": strategy_policy_config_hash(config),
            "structure": matrix,
            "volatility_overlays": overlays,
            "warning": "eligibility never overrides BalancedRiskEngine",
        }
        print(_render_payload(payload, args.as_json))
        return 0

    settings = load_runtime_settings("PAPER", args.profile)
    if args.symbol not in settings.profile.asset_universe:
        raise ValueError("symbol must come from the active profile configuration")
    config = load_balanced_regime_config(settings.profile)
    store = ParquetDataStore(args.data_root)
    if args.regime_command == "inspect":
        start = args.start
        end = args.end
    elif args.regime_command == "latest":
        manifest = store.find_latest(args.symbol, args.timeframe)
        if manifest is None:
            raise ValueError(
                f"no cached dataset for {args.symbol} {args.timeframe}; regime commands never download"
            )
        start = manifest.requested_start
        end = manifest.requested_end
    else:
        raise AssertionError(f"unhandled regime command: {args.regime_command}")
    dataset = load_cached_dataset(
        store,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
    )
    if dataset.quality_report.quality_status is QualityStatus.FAIL:
        raise ValueError("DataQuality FAIL cannot be classified by the Regime Detector")
    detector = BalancedRegimeDetector(config)
    feature_engine = FeatureEngine()
    history = []
    snapshots = []
    for bar in dataset.bars:
        history.append(bar)
        features = feature_engine.compute(
            history,
            detector.feature_request,
            as_of=bar.timestamp,
        )
        snapshots.append(detector.evaluate(features))
    report = build_regime_report(snapshots, detector.transitions, ())
    payload = {
        "detector_name": detector.detector_name,
        "detector_version": detector.detector_version,
        "config_hash": regime_config_hash(config),
        "dataset_id": dataset.reference.dataset_id,
        "dataset_checksum": dataset.reference.checksum_sha256,
        "data_quality": dataset.quality_report.quality_status.value,
        "latest": _regime_snapshot_payload(snapshots[-1]),
        "report": {
            "bars_by_structure_regime": dict(report.bars_by_structure_regime),
            "bars_by_volatility_regime": dict(report.bars_by_volatility_regime),
            "transition_count": report.transition_count,
        },
        "transitions": [
            {
                "transition_id": item.transition_id,
                "timestamp": item.timestamp.isoformat(),
                "from_structure": item.from_structure.value,
                "to_structure": item.to_structure.value,
                "from_volatility": item.from_volatility.value,
                "to_volatility": item.to_volatility.value,
                "reason": item.reason,
            }
            for item in detector.transitions
        ],
        "network_access": False,
    }
    print(_render_payload(payload, args.as_json))
    return 0


def _run_dashboard(args: argparse.Namespace) -> int:
    from trading_ai.monitoring.dashboard import (
        build_monitoring_service,
        serve_dashboard,
    )

    if args.dashboard_command == "serve":
        serve_dashboard(data_root=args.data_root, host=args.host, port=args.port)
        return 0
    if args.dashboard_command == "inspect":
        payload = build_monitoring_service(args.data_root).inspect(args.run_id)
        print(_render_payload(payload, args.as_json))
        return 0
    raise AssertionError(f"unhandled dashboard command: {args.dashboard_command}")


def _run_monitoring(args: argparse.Namespace) -> int:
    from trading_ai.monitoring.dashboard import build_monitoring_service

    if args.monitoring_command != "health":
        raise AssertionError(f"unhandled monitoring command: {args.monitoring_command}")
    service = build_monitoring_service(args.data_root)
    payload = (
        service.section(args.run_id, "health")
        if args.run_id is not None
        else service.health_without_run()
    )
    print(_render_payload(payload, args.as_json))
    return 0


def _run_ml(args: argparse.Namespace) -> int:
    registry = LocalModelRegistry(args.data_root / "ml")
    if args.ml_command == "evaluate":
        payload = registry.inspect(args.model_id)
        print(_render_payload(payload["evaluation"], args.as_json))
        return 0
    if args.ml_command == "model":
        if args.model_command == "list":
            payload = [to_primitive(artifact) for artifact in registry.list()]
        elif args.model_command == "inspect":
            payload = registry.inspect(args.model_id)
        elif args.model_command == "promote":
            payload = to_primitive(
                registry.promote(
                    args.model_id,
                    ModelStatus(args.to),
                    reason=args.reason,
                )
            )
        elif args.model_command == "rollback":
            payload = to_primitive(
                registry.rollback(
                    args.strategy,
                    args.timeframe,
                    reason=args.reason,
                )
            )
        else:
            raise AssertionError(f"unhandled model command: {args.model_command}")
        print(_render_payload(payload, args.as_json))
        return 0
    if args.ml_command != "train":
        raise AssertionError(f"unhandled ML command: {args.ml_command}")

    settings = load_runtime_settings("PAPER", args.profile)
    symbols = tuple(dict.fromkeys(args.symbol or settings.profile.asset_universe))
    if any(symbol not in settings.profile.asset_universe for symbol in symbols):
        raise ValueError("ML training symbols must come from profile configuration")
    store = ParquetDataStore(args.data_root)
    datasets = tuple(
        load_cached_dataset(
            store,
            symbol=symbol,
            timeframe=args.timeframe,
            start=args.train_start,
            end=args.test_end,
        )
        for symbol in sorted(symbols)
    )
    strategy = BASELINE_STRATEGIES.create(
        args.strategy,
        symbols=symbols,
        timeframe=args.timeframe,
        config=None,
    )
    feature_engine = FeatureEngine()
    quant_result = BacktestEngine(
        risk_engine=BalancedRiskEngine.from_profile(settings.profile),
        feature_engine=feature_engine,
        regime_detector=BalancedRegimeDetector.from_profile(settings.profile),
        activation_policy=BalancedStrategyActivationPolicy.from_profile(
            settings.profile
        ),
    ).run(
        strategy,
        datasets,
        settings.context,
        BacktestConfig(
            starting_cash=Decimal("100000"),
            primary_timeframe=args.timeframe,
            benchmark_symbol=symbols[0],
            data_quality_policy=DataQualityPolicy.STRICT,
        ),
    )
    label_config = LabelConfig(
        horizon_bars=args.horizon_bars,
        minimum_forward_return_bps=args.minimum_forward_return_bps,
    )
    build_result = SignalTrainingDatasetBuilder(
        label_config=label_config
    ).build(quant_result, datasets)
    split_config = TemporalSplitConfig(
        training=TimeRange(args.train_start, args.train_end),
        validation=TimeRange(args.validation_start, args.validation_end),
        final_test=TimeRange(args.test_start, args.test_end),
        embargo_bars=args.embargo_bars,
        walk_forward_folds=args.walk_forward_folds,
    )
    outcome = TrainingPipeline(
        config=TrainingConfig(
            minimum_training_samples=args.minimum_training_samples,
            minimum_samples_per_class=args.minimum_samples_per_class,
        )
    ).run(
        build_result.dataset,
        split_config=split_config,
        model_config=ModelConfig(family=ModelFamily(args.model)),
    )
    artifact = registry.save(outcome)
    payload = {
        "artifact": to_primitive(artifact),
        "dataset_build": to_primitive(build_result.report),
        "evaluation": to_primitive(outcome.evaluation),
        "network_access": False,
        "automatic_promotion": False,
    }
    print(_render_payload(payload, args.as_json))
    return 0


def _run_backtest(args: argparse.Namespace) -> int:
    result_store = BacktestResultStore(args.data_root / "backtests")
    if args.backtest_command == "inspect":
        print(_render_payload(result_store.inspect(args.run_id), args.as_json))
        return 0
    if args.backtest_command != "run":
        raise AssertionError(f"unhandled backtest command: {args.backtest_command}")
    settings = load_runtime_settings(args.environment, args.profile)
    strategy_names = tuple(args.strategy or ("buy-and-hold",))
    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError("each --strategy may be supplied at most once")
    if "buy-and-hold" in strategy_names and len(strategy_names) != 1:
        raise ValueError("buy-and-hold cannot be combined with portfolio baselines")
    symbols = tuple(dict.fromkeys(args.symbol))
    if any(symbol not in settings.profile.asset_universe for symbol in symbols):
        invalid = sorted(set(symbols) - set(settings.profile.asset_universe))
        raise ValueError(
            "symbols must come from the active profile configuration: "
            + ", ".join(invalid)
        )
    if strategy_names == ("buy-and-hold",) and len(symbols) != 1:
        raise ValueError("buy-and-hold accepts exactly one --symbol")
    benchmark_symbol = args.benchmark_symbol or symbols[0]
    if benchmark_symbol not in settings.profile.asset_universe:
        raise ValueError("benchmark symbol must come from the active profile configuration")
    store = ParquetDataStore(args.data_root)
    dataset_symbols = tuple(sorted(set((*symbols, benchmark_symbol))))
    datasets = tuple(
        load_cached_dataset(
            store,
            symbol=symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
        )
        for symbol in dataset_symbols
    )
    regime_detector = None
    activation_policy = None
    ml_scorer = None
    ml_scorers = None
    portfolio_engine = None
    shared_feature_engine = FeatureEngine()
    if strategy_names == ("buy-and-hold",):
        strategy = BuyAndHoldDemoStrategy(symbols[0], args.quantity)
    else:
        strategies = tuple(
            BASELINE_STRATEGIES.create(
                name,
                symbols=symbols,
                timeframe=args.timeframe,
                config=_CLI_CONFIG_BUILDERS[name](args),
            )
            for name in strategy_names
        )
        strategy = strategies[0] if len(strategies) == 1 else strategies
        regime_detector = BalancedRegimeDetector.from_profile(settings.profile)
        activation_policy = BalancedStrategyActivationPolicy.from_profile(
            settings.profile
        )
        if len(strategies) > 1:
            portfolio_engine = BalancedPortfolioEngine.from_profile(settings.profile)
    ml_mode = MLMode(args.ml_mode.upper().replace("-", "_"))
    model_arguments = tuple(args.ml_model_id or ())
    if ml_mode is MLMode.DISABLED:
        if model_arguments:
            raise ValueError("--ml-model-id requires score-only or filter mode")
    else:
        if strategy_names == ("buy-and-hold",):
            raise ValueError("ML scoring applies to quantitative baseline signals only")
        if not model_arguments:
            raise ValueError("active ML mode requires explicit --ml-model-id")
        registry = LocalModelRegistry(args.data_root / "ml")
        if len(strategy_names) == 1:
            if len(model_arguments) != 1:
                raise ValueError("single-strategy ML requires exactly one model ID")
            model_id = model_arguments[0].split("=", 1)[-1]
            assert not isinstance(strategy, tuple)
            artifact, adapter, _ = registry.load(
                model_id,
                strategy_name=strategy.name,
                strategy_version=strategy.version,
                timeframe=args.timeframe,
            )
            ml_scorer = SignalMLScorer(
                mode=ml_mode,
                inference_engine=InferenceEngine(artifact, adapter),
                threshold=args.ml_threshold,
            )
        else:
            assignments: dict[str, str] = {}
            for raw in model_arguments:
                if "=" not in raw:
                    raise ValueError(
                        "multi-strategy ML model IDs must use strategy=model-id"
                    )
                name, model_id = raw.split("=", 1)
                if name not in strategy_names or name in assignments or not model_id:
                    raise ValueError("invalid or duplicate strategy=model-id assignment")
                assignments[name] = model_id
            if set(assignments) != set(strategy_names):
                raise ValueError("multi-strategy ML requires one model ID per strategy")
            assert isinstance(strategy, tuple)
            ml_scorers = {}
            for item in strategy:
                artifact, adapter, _ = registry.load(
                    assignments[item.name],
                    strategy_name=item.name,
                    strategy_version=item.version,
                    timeframe=args.timeframe,
                )
                ml_scorers[item.name] = SignalMLScorer(
                    mode=ml_mode,
                    inference_engine=InferenceEngine(artifact, adapter),
                    threshold=args.ml_threshold,
                )
    config = BacktestConfig(
        starting_cash=args.starting_cash,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        commission=CommissionConfig(
            fixed=args.commission_fixed,
            percentage_bps=args.commission_bps,
            minimum=args.commission_minimum,
        ),
        allow_short=False,
        data_quality_policy=(
            DataQualityPolicy.ALLOW_WARNINGS
            if args.allow_data_warnings
            else DataQualityPolicy.STRICT
        ),
        primary_timeframe=args.timeframe,
        benchmark_symbol=benchmark_symbol,
    )
    if any(
        value != Decimal("0")
        for value in (
            args.commission_fixed,
            args.commission_bps,
            args.commission_minimum,
        )
    ):
        raise ValueError(
            "legacy commission flags cannot be combined with the explicit Lot 8.1 cost profile"
        )
    cost_engine = BalancedTransactionCostEngine.from_profile(
        settings.profile, tariff_profile=args.cost_profile
    )
    economic_gate = EconomicGate(
        cost_engine.bundle.config, cost_engine.config_hash
    )
    risk_engine = BalancedRiskEngine.from_profile(settings.profile)
    result = BacktestEngine(
        risk_engine=risk_engine,
        feature_engine=shared_feature_engine,
        regime_detector=regime_detector,
        activation_policy=activation_policy,
        ml_scorer=ml_scorer,
        ml_scorers=ml_scorers,
        portfolio_engine=portfolio_engine,
        cost_engine=cost_engine,
        economic_gate=economic_gate,
    ).run(
        strategy,
        datasets,
        settings.context,
        config,
    )
    export_directory = result_store.export(result)
    payload = {
        "run_id": result.run_id,
        "status": result.status,
        "strategy": result.strategy_name,
        "dataset_ids": [
            reference.dataset_id for reference in result.dataset_references
        ],
        "initial_cash": str(result.initial_cash),
        "final_equity": str(result.final_equity),
        "number_of_trades": result.metrics.number_of_trades,
        "number_of_signals": len(result.signals),
        "total_return": result.metrics.total_return,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "fees": str(result.metrics.total_commission),
        "spread_cost": str(result.metrics.total_spread_cost),
        "slippage_cost": str(result.metrics.total_slippage_cost),
        "total_transaction_costs": str(
            result.metrics.total_commission
            + result.metrics.total_spread_cost
            + result.metrics.total_slippage_cost
        ),
        "benchmark": (
            {
                "symbol": result.benchmark.symbol,
                "total_return": result.benchmark.total_return,
                "excess_return": result.benchmark.excess_return,
            }
            if result.benchmark is not None
            else None
        ),
        "result_hash": result.result_hash,
        "risk": {
            "engine": result.risk_engine_name,
            "version": result.risk_engine_version,
            "config_hash": result.risk_config_hash,
            "approved": result.risk_summary.approved_orders,
            "reduced": result.risk_summary.reduced_orders,
            "rejected": result.risk_summary.rejected_orders,
            "state_transitions": len(result.risk_state_transitions),
            "max_portfolio_exposure": result.risk_summary.max_portfolio_exposure,
            "max_drawdown": result.risk_summary.max_observed_drawdown,
            "max_daily_loss": result.risk_summary.max_daily_loss,
        },
        "regime": (
            {
                "detector": result.regime_detector_name,
                "version": result.regime_detector_version,
                "config_hash": result.regime_config_hash,
                "policy": result.strategy_policy_name,
                "policy_version": result.strategy_policy_version,
                "policy_config_hash": result.strategy_policy_config_hash,
                "transitions": len(result.regime_transitions),
                "activation_allow": result.regime_report.activation_allow,
                "activation_reduce": result.regime_report.activation_reduce,
                "activation_block": result.regime_report.activation_block,
            }
            if result.regime_report is not None
            else {"status": "unavailable"}
        ),
        "ml": {
            "mode": result.ml_mode,
            "model_id": result.ml_model_id,
            "model_family": result.ml_model_family,
            "model_status": result.ml_model_status,
            "threshold": result.ml_threshold,
            "predictions": len(result.ml_predictions),
            "pass": sum(
                decision.status.value == "PASS" for decision in result.ml_decisions
            ),
            "block": sum(
                decision.status.value == "BLOCK" for decision in result.ml_decisions
            ),
            "unavailable": sum(
                decision.status.value == "UNAVAILABLE"
                for decision in result.ml_decisions
            ),
        },
        "portfolio": (
            {
                "engine": result.portfolio_engine_name,
                "version": result.portfolio_engine_version,
                "config_hash": result.portfolio_config_hash,
                "opportunities": len(result.portfolio_opportunities),
                "selected": result.portfolio_metrics.opportunities_selected,
                "deferred": result.portfolio_metrics.opportunities_deferred,
                "rejected": result.portfolio_metrics.opportunities_rejected,
                "plans": len(result.portfolio_plans),
                "targets": len(result.portfolio_targets),
                "max_exposure": result.portfolio_metrics.max_gross_exposure,
                "planned_turnover": result.portfolio_metrics.planned_turnover,
                "executed_turnover": result.portfolio_metrics.executed_turnover,
                "sleeves": dict(
                    result.portfolio_metrics.targets_by_strategy_sleeve
                ),
            }
            if result.portfolio_metrics is not None
            else {"status": "unavailable / legacy sizing"}
        ),
        "costs": (
            {
                "engine": result.cost_engine_name,
                "version": result.cost_engine_version,
                "config_hash": result.cost_config_hash,
                "tariff_profile": result.tariff_profile_id,
                "tariff_status": result.tariff_status,
                "coverage": result.cost_summary.cost_coverage.value,
                "gross_trading_pnl": (
                    str(result.cost_summary.gross_trading_pnl)
                    if result.cost_summary.gross_trading_pnl is not None else None
                ),
                "variable_trading_costs": (
                    str(result.cost_summary.total_variable_cost)
                    if result.cost_summary.total_variable_cost is not None else None
                ),
                "net_before_operating": (
                    str(result.cost_summary.net_trading_pnl_before_operating)
                    if result.cost_summary.net_trading_pnl_before_operating is not None else None
                ),
                "net_economic": (
                    str(result.cost_summary.net_economic_pnl)
                    if result.cost_summary.net_economic_pnl is not None else None
                ),
                "estimates": len(result.cost_estimates),
                "actuals": len(result.cost_actuals),
                "economic_decisions": len(result.economic_decisions),
            }
            if result.cost_summary is not None else {"status": "UNAVAILABLE"}
        ),
        "strategy_parameters": dict(result.strategy_parameters),
        "export_path": str(export_directory),
        "warnings": list(result.warnings),
    }
    print(_render_payload(payload, args.as_json))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = doctor(args.environment, args.profile)
        print(format_report(report, args.as_json))
        return 0 if report.configuration_valid else 2
    try:
        if args.command == "data":
            return _run_data(args)
        if args.command == "backtest":
            return _run_backtest(args)
        if args.command == "strategy":
            return _run_strategy(args)
        if args.command == "risk":
            return _run_risk(args)
        if args.command == "regime":
            return _run_regime(args)
        if args.command == "ml":
            return _run_ml(args)
        if args.command == "portfolio":
            return _run_portfolio(args)
        if args.command == "costs":
            return _run_costs(args)
        if args.command == "validation":
            return _run_validation(args)
        if args.command == "robustness":
            return _run_robustness(args)
        if args.command == "evidence":
            return _run_evidence(args)
        if args.command == "dashboard":
            return _run_dashboard(args)
        if args.command == "monitoring":
            return _run_monitoring(args)
    except (
        BacktestError,
        DataError,
        RegimeError,
        MLError,
        MonitoringError,
        CostError,
        ValidationError,
        RobustnessError,
        TradingAIError,
        ValueError,
    ) as exc:
        if getattr(args, "as_json", False):
            print(json.dumps({"status": "ERROR", "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
