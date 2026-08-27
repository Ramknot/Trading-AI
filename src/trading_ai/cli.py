"""Safety diagnostics, historical-data, and offline backtest commands."""

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
from trading_ai.backtesting.strategy import BuyAndHoldDemoStrategy
from trading_ai.core.config import load_runtime_settings
from trading_ai.core.exceptions import TradingAIError
from trading_ai.core.health import HealthReport, doctor
from trading_ai.data.engine import DataEngine
from trading_ai.data.exceptions import DataError
from trading_ai.data.models import CacheMode, DataFetchResult, DatasetInspection
from trading_ai.data.providers import YahooFinanceProvider
from trading_ai.data.storage import ParquetDataStore
from trading_ai.strategies.config import (
    BreakoutConfig,
    MomentumConfig,
    TrendConfig,
)
from trading_ai.strategies.registry import BASELINE_STRATEGIES


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
        description="Trading AI diagnostics, historical data, and offline backtests",
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
        default="buy-and-hold",
        help="buy-and-hold demo, trend, momentum, or breakout",
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
        "--benchmark-symbol",
        help="optional Buy & Hold benchmark; defaults to the selected symbol",
    )
    run_parser.add_argument(
        "--allow-data-warnings",
        action="store_true",
        help="accept DataQuality WARNING datasets; FAIL is always rejected",
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


_CLI_CONFIG_BUILDERS = {
    "trend": _trend_cli_config,
    "momentum": _momentum_cli_config,
    "breakout": _breakout_cli_config,
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


def _run_backtest(args: argparse.Namespace) -> int:
    result_store = BacktestResultStore(args.data_root / "backtests")
    if args.backtest_command == "inspect":
        print(_render_payload(result_store.inspect(args.run_id), args.as_json))
        return 0
    if args.backtest_command != "run":
        raise AssertionError(f"unhandled backtest command: {args.backtest_command}")
    settings = load_runtime_settings(args.environment, args.profile)
    symbols = tuple(dict.fromkeys(args.symbol))
    if any(symbol not in settings.profile.asset_universe for symbol in symbols):
        invalid = sorted(set(symbols) - set(settings.profile.asset_universe))
        raise ValueError(
            "symbols must come from the active profile configuration: "
            + ", ".join(invalid)
        )
    if args.strategy == "buy-and-hold" and len(symbols) != 1:
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
    if args.strategy == "buy-and-hold":
        strategy = BuyAndHoldDemoStrategy(symbols[0], args.quantity)
    else:
        config_builder = _CLI_CONFIG_BUILDERS[args.strategy]
        strategy = BASELINE_STRATEGIES.create(
            args.strategy,
            symbols=symbols,
            timeframe=args.timeframe,
            config=config_builder(args),
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
    result = BacktestEngine().run(
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
    except (BacktestError, DataError, TradingAIError, ValueError) as exc:
        if getattr(args, "as_json", False):
            print(json.dumps({"status": "ERROR", "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
