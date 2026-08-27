"""Safety diagnostics and historical-data commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

from trading_ai.core.exceptions import TradingAIError
from trading_ai.core.health import HealthReport, doctor
from trading_ai.data.engine import DataEngine
from trading_ai.data.exceptions import DataError
from trading_ai.data.models import CacheMode, DataFetchResult, DatasetInspection
from trading_ai.data.providers import YahooFinanceProvider
from trading_ai.data.storage import ParquetDataStore


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-ai",
        description="Trading AI diagnostics and historical market data",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = doctor(args.environment, args.profile)
        print(format_report(report, args.as_json))
        return 0 if report.configuration_valid else 2
    try:
        if args.command == "data":
            return _run_data(args)
    except (DataError, TradingAIError, ValueError) as exc:
        if getattr(args, "as_json", False):
            print(json.dumps({"status": "ERROR", "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
