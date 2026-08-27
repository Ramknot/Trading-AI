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


RESULT_SCHEMA_VERSION = "1.3"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_EXPORTED_FILES = frozenset(
    {
        "summary.json",
        "equity.parquet",
        "orders.parquet",
        "fills.parquet",
        "trades.parquet",
        "signals.parquet",
        "ledger.parquet",
        "risk_decisions.parquet",
        "risk_states.parquet",
        "regime_snapshots.parquet",
        "regime_transitions.parquet",
        "activation_decisions.parquet",
    }
)
_LOT4_EXPORTED_FILES = _EXPORTED_FILES - {
    "regime_snapshots.parquet",
    "regime_transitions.parquet",
    "activation_decisions.parquet",
}
_LOT3_EXPORTED_FILES = _LOT4_EXPORTED_FILES - {
    "risk_decisions.parquet",
    "risk_states.parquet",
}
_LOT2_EXPORTED_FILES = _LOT3_EXPORTED_FILES - {"signals.parquet"}


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
                ("signal_id", pa.string()),
                ("activation_decision_id", pa.string()),
                ("risk_decision_id", pa.string()),
            ]
        )

    @staticmethod
    def _regime_snapshot_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("snapshot_id", pa.string()),
                ("symbol", pa.string()),
                ("timestamp", pa.string()),
                ("timeframe", pa.string()),
                ("structure_regime", pa.string()),
                ("volatility_regime", pa.string()),
                ("detector_name", pa.string()),
                ("detector_version", pa.string()),
                ("config_hash", pa.string()),
                ("bars_in_current_structure_regime", pa.int64()),
                ("evidence", pa.list_(pa.list_(pa.string(), 2))),
                ("reason_codes", pa.list_(pa.string())),
                ("candidate_structure_regime", pa.string()),
                ("confirmation_progress", pa.int64()),
                ("transition_from", pa.string()),
                ("transition_reason", pa.string()),
            ]
        )

    @staticmethod
    def _regime_transition_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("transition_id", pa.string()),
                ("symbol", pa.string()),
                ("timestamp", pa.string()),
                ("timeframe", pa.string()),
                ("from_structure", pa.string()),
                ("to_structure", pa.string()),
                ("from_volatility", pa.string()),
                ("to_volatility", pa.string()),
                ("reason", pa.string()),
            ]
        )

    @staticmethod
    def _activation_decision_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("decision_id", pa.string()),
                ("timestamp", pa.string()),
                ("symbol", pa.string()),
                ("strategy_name", pa.string()),
                ("strategy_version", pa.string()),
                ("signal_id", pa.string()),
                ("regime_snapshot_id", pa.string()),
                ("structure_regime", pa.string()),
                ("volatility_regime", pa.string()),
                ("status", pa.string()),
                ("allocation_multiplier", pa.string()),
                ("proposed_quantity", pa.string()),
                ("adjusted_quantity", pa.string()),
                ("reason_codes", pa.list_(pa.string())),
                ("human_readable_reasons", pa.list_(pa.string())),
                ("policy_name", pa.string()),
                ("policy_version", pa.string()),
                ("policy_config_hash", pa.string()),
            ]
        )

    @staticmethod
    def _risk_decision_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("decision_id", pa.string()),
                ("order_id", pa.string()),
                ("status", pa.string()),
                ("reason", pa.string()),
                ("risk_engine", pa.string()),
                ("timestamp", pa.string()),
                ("engine_version", pa.string()),
                ("requested_quantity", pa.string()),
                ("approved_quantity", pa.string()),
                ("reason_codes", pa.list_(pa.string())),
                ("human_readable_reasons", pa.list_(pa.string())),
                ("risk_state", pa.string()),
                ("config_hash", pa.string()),
                ("equity", pa.string()),
                ("cash", pa.string()),
                ("gross_exposure_before", pa.float64()),
                ("gross_exposure_after", pa.float64()),
                ("position_exposure_before", pa.float64()),
                ("position_exposure_after", pa.float64()),
                ("daily_loss_pct", pa.float64()),
                ("drawdown_pct", pa.float64()),
                ("volatility_metric", pa.float64()),
                ("correlation_metric", pa.float64()),
            ]
        )

    @staticmethod
    def _risk_state_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("transition_id", pa.string()),
                ("timestamp", pa.string()),
                ("previous_state", pa.string()),
                ("new_state", pa.string()),
                ("reason", pa.string()),
                ("equity", pa.string()),
                ("daily_loss_pct", pa.float64()),
                ("drawdown_pct", pa.float64()),
            ]
        )

    @staticmethod
    def _signal_schema():
        import pyarrow as pa

        return pa.schema(
            [
                ("signal_id", pa.string()),
                ("strategy_name", pa.string()),
                ("strategy_version", pa.string()),
                ("symbol", pa.string()),
                ("timeframe", pa.string()),
                ("timestamp", pa.string()),
                ("action", pa.string()),
                ("strength", pa.float64()),
                ("reason", pa.string()),
                ("features_used", pa.list_(pa.list_(pa.string(), 2))),
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
                "risk": {
                    "engine_name": result.risk_engine_name,
                    "engine_version": result.risk_engine_version,
                    "config": result.risk_config,
                    "config_hash": result.risk_config_hash,
                    "summary": result.risk_summary,
                },
                "regime": {
                    "detector_name": result.regime_detector_name,
                    "detector_version": result.regime_detector_version,
                    "config": result.regime_config,
                    "config_hash": result.regime_config_hash,
                    "policy_name": result.strategy_policy_name,
                    "policy_version": result.strategy_policy_version,
                    "policy_config": result.strategy_policy_config,
                    "policy_config_hash": result.strategy_policy_config_hash,
                    "report": result.regime_report,
                },
                "counts": {
                    "equity_points": len(result.equity_curve),
                    "orders": len(result.orders),
                    "fills": len(result.fills),
                    "trades": len(result.trades),
                    "signals": len(result.signals),
                    "ledger_entries": len(result.ledger_entries),
                    "risk_decisions": len(result.risk_decisions),
                    "risk_state_transitions": len(result.risk_state_transitions),
                    "regime_snapshots": len(result.regime_snapshots),
                    "regime_transitions": len(result.regime_transitions),
                    "activation_decisions": len(result.activation_decisions),
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
                directory / "signals.parquet",
                (to_primitive(signal) for signal in result.signals),
                self._signal_schema(),
            )
            self._write_parquet(
                directory / "ledger.parquet",
                (to_primitive(entry) for entry in result.ledger_entries),
                self._ledger_schema(),
            )
            self._write_parquet(
                directory / "risk_decisions.parquet",
                (to_primitive(decision) for decision in result.risk_decisions),
                self._risk_decision_schema(),
            )
            self._write_parquet(
                directory / "risk_states.parquet",
                (
                    to_primitive(transition)
                    for transition in result.risk_state_transitions
                ),
                self._risk_state_schema(),
            )
            self._write_parquet(
                directory / "regime_snapshots.parquet",
                (to_primitive(snapshot) for snapshot in result.regime_snapshots),
                self._regime_snapshot_schema(),
            )
            self._write_parquet(
                directory / "regime_transitions.parquet",
                (to_primitive(transition) for transition in result.regime_transitions),
                self._regime_transition_schema(),
            )
            self._write_parquet(
                directory / "activation_decisions.parquet",
                (to_primitive(decision) for decision in result.activation_decisions),
                self._activation_decision_schema(),
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
            exported_files = frozenset(payload["files"])
            if exported_files not in {
                _EXPORTED_FILES,
                _LOT4_EXPORTED_FILES,
                _LOT3_EXPORTED_FILES,
                _LOT2_EXPORTED_FILES,
            }:
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
            expected_files = {
                "1.0": _LOT2_EXPORTED_FILES,
                "1.1": _LOT3_EXPORTED_FILES,
                "1.2": _LOT4_EXPORTED_FILES,
                RESULT_SCHEMA_VERSION: _EXPORTED_FILES,
            }.get(summary.get("schema_version"))
            if expected_files is None or exported_files != expected_files:
                raise BacktestStorageError(
                    "backtest schema version does not match its exported files"
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
        payload = json.loads(
            (self._directory(run_id) / "summary.json").read_text(encoding="utf-8")
        )
        if "regime" not in payload:
            payload["regime"] = {"status": "unavailable"}
        return payload
