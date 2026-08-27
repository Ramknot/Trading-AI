"""Local JSON/Parquet exports and SHA-256 verification for backtest results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from trading_ai.backtesting.exceptions import BacktestStorageError
from trading_ai.backtesting.reproducibility import to_primitive
from trading_ai.core.models import BacktestResult


RESULT_SCHEMA_VERSION = "1.0"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_EXPORTED_FILES = frozenset(
    {
        "summary.json",
        "equity.parquet",
        "orders.parquet",
        "fills.parquet",
        "trades.parquet",
        "ledger.parquet",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BacktestResultStore:
    """Export deterministic results below the Git-ignored data_local tree."""

    def __init__(
        self, root: Path | str = Path("data_local") / "backtests"
    ) -> None:
        self.root = Path(root)

    def _directory(self, run_id: str) -> Path:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise BacktestStorageError("invalid backtest run_id")
        root = self.root.resolve()
        directory = (self.root / run_id).resolve()
        if root not in directory.parents:
            raise BacktestStorageError("backtest path escapes the result store")
        return directory

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _write_parquet(
        path: Path,
        records: Iterable[dict[str, Any]],
        schema,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(records)
        table = pa.Table.from_pylist(rows, schema=schema)
        temporary = path.with_suffix(path.suffix + ".tmp")
        parquet.write_table(table, temporary, compression="zstd")
        temporary.replace(path)

    @staticmethod
    def _equity_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("timestamp", pa.string()),
                ("cash", pa.string()),
                ("positions_value", pa.string()),
                ("equity", pa.string()),
                ("realized_pnl", pa.string()),
                ("unrealized_pnl", pa.string()),
            ]
        )

    @staticmethod
    def _fill_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("fill_id", pa.string()),
                ("order_id", pa.string()),
                ("symbol", pa.string()),
                ("side", pa.string()),
                ("quantity", pa.string()),
                ("reference_price", pa.string()),
                ("price", pa.string()),
                ("timestamp", pa.string()),
                ("commission", pa.string()),
                ("slippage_cost", pa.string()),
                ("spread_cost", pa.string()),
            ]
        )

    @staticmethod
    def _trade_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("trade_id", pa.string()),
                ("symbol", pa.string()),
                ("entry_time", pa.string()),
                ("exit_time", pa.string()),
                ("entry_price", pa.string()),
                ("exit_price", pa.string()),
                ("quantity", pa.string()),
                ("gross_pnl", pa.string()),
                ("fees", pa.string()),
                ("spread_cost", pa.string()),
                ("slippage_cost", pa.string()),
                ("net_pnl", pa.string()),
                ("return_pct", pa.string()),
                ("holding_period_seconds", pa.float64()),
            ]
        )

    @staticmethod
    def _order_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("order_id", pa.string()),
                ("symbol", pa.string()),
                ("timeframe", pa.string()),
                ("side", pa.string()),
                ("quantity", pa.string()),
                ("order_type", pa.string()),
                ("created_at", pa.string()),
                ("status", pa.string()),
                ("limit_price", pa.string()),
                ("status_reason", pa.string()),
                ("completed_at", pa.string()),
                ("eligible_bar_count", pa.int64()),
            ]
        )

    @staticmethod
    def _ledger_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("entry_id", pa.string()),
                ("timestamp", pa.string()),
                ("entry_type", pa.string()),
                ("symbol", pa.string()),
                ("cash_change", pa.string()),
                ("quantity_change", pa.string()),
                ("amount", pa.string()),
                ("reference_id", pa.string()),
                ("message", pa.string()),
            ]
        )

    @staticmethod
    def _summary(result: BacktestResult) -> dict[str, Any]:
        benchmark = None
        if result.benchmark is not None:
            benchmark = {
                "symbol": result.benchmark.symbol,
                "initial_equity": result.benchmark.initial_equity,
                "final_equity": result.benchmark.final_equity,
                "total_return": result.benchmark.total_return,
                "max_drawdown_pct": result.benchmark.max_drawdown_pct,
                "excess_return": result.benchmark.excess_return,
            }
        return to_primitive(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "run_id": result.run_id,
                "status": result.status,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "created_at": result.created_at,
                "strategy_name": result.strategy_name,
                "strategy_version": result.strategy_version,
                "strategy_parameters": result.strategy_parameters,
                "dataset_references": result.dataset_references,
                "config": result.config,
                "initial_cash": result.initial_cash,
                "final_equity": result.final_equity,
                "metrics": result.metrics,
                "warnings": result.warnings,
                "benchmark": benchmark,
                "result_hash": result.result_hash,
                "code_version": result.code_version,
                "source_hash_sha256": result.source_hash_sha256,
                "counts": {
                    "equity_points": len(result.equity_curve),
                    "orders": len(result.orders),
                    "fills": len(result.fills),
                    "trades": len(result.trades),
                    "ledger_entries": len(result.ledger_entries),
                },
            }
        )

    def export(self, result: BacktestResult) -> Path:
        directory = self._directory(result.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            self._write_json(directory / "summary.json", self._summary(result))
            self._write_parquet(
                directory / "equity.parquet",
                (to_primitive(point) for point in result.equity_curve),
                self._equity_schema(),
            )
            self._write_parquet(
                directory / "fills.parquet",
                (to_primitive(fill) for fill in result.fills),
                self._fill_schema(),
            )
            self._write_parquet(
                directory / "trades.parquet",
                (to_primitive(trade) for trade in result.trades),
                self._trade_schema(),
            )
            self._write_parquet(
                directory / "orders.parquet",
                (to_primitive(order) for order in result.orders),
                self._order_schema(),
            )
            self._write_parquet(
                directory / "ledger.parquet",
                (to_primitive(entry) for entry in result.ledger_entries),
                self._ledger_schema(),
            )
            checksums = {
                name: _sha256_file(directory / name)
                for name in sorted(_EXPORTED_FILES)
            }
            self._write_json(
                directory / "checksums.json",
                {
                    "algorithm": "SHA-256",
                    "files": checksums,
                    "result_hash": result.result_hash,
                },
            )
        except BacktestStorageError:
            raise
        except Exception as exc:
            raise BacktestStorageError(
                f"unable to export backtest {result.run_id}"
            ) from exc
        return directory

    def verify_integrity(self, run_id: str) -> bool:
        directory = self._directory(run_id)
        try:
            payload = json.loads(
                (directory / "checksums.json").read_text(encoding="utf-8")
            )
            if set(payload["files"]) != _EXPORTED_FILES:
                raise BacktestStorageError(
                    "backtest checksum manifest has an unexpected file set"
                )
            for name, expected in payload["files"].items():
                if len(expected) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in expected.lower()
                ):
                    raise BacktestStorageError(
                        f"invalid SHA-256 metadata for backtest file {name}"
                    )
                path = directory / name
                if not path.is_file() or _sha256_file(path) != expected:
                    raise BacktestStorageError(
                        f"SHA-256 mismatch for backtest file {name}"
                    )
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            if summary.get("result_hash") != payload.get("result_hash"):
                raise BacktestStorageError("backtest result hash metadata mismatch")
        except BacktestStorageError:
            raise
        except FileNotFoundError as exc:
            raise BacktestStorageError(f"backtest result not found: {run_id}") from exc
        except Exception as exc:
            raise BacktestStorageError(
                f"invalid backtest export: {run_id}"
            ) from exc
        return True

    def inspect(self, run_id: str) -> dict[str, Any]:
        self.verify_integrity(run_id)
        return json.loads(
            (self._directory(run_id) / "summary.json").read_text(encoding="utf-8")
        )
