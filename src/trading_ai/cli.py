"""Diagnostic command-line interface for safe Lot 0 startup checks."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from trading_ai.core.health import HealthReport, doctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-ai",
        description="Trading AI safety diagnostics",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = doctor(args.environment, args.profile)
        print(format_report(report, args.as_json))
        return 0 if report.configuration_valid else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
