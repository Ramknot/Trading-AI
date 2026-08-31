"""Checksum-verified observability source for local backtest exports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_ai.backtesting.exceptions import BacktestStorageError
from trading_ai.backtesting.reproducibility import stable_hash, to_primitive
from trading_ai.backtesting.storage import BacktestResultStore
from trading_ai.monitoring.base import MonitoringSource
from trading_ai.monitoring.exceptions import (
    MonitoringIntegrityError,
    MonitoringNotFoundError,
)
from trading_ai.monitoring.models import MonitoringEvent, MonitoringEventType
from trading_ai.monitoring.security import redact_sensitive
from trading_ai.validation.storage import LocalValidationStore


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_TABLES = (
    "equity",
    "orders",
    "fills",
    "trades",
    "signals",
    "ledger",
    "risk_decisions",
    "risk_states",
    "regime_snapshots",
    "regime_transitions",
    "activation_decisions",
    "ml_predictions",
    "ml_decisions",
    "portfolio_opportunities",
    "portfolio_decisions",
    "portfolio_targets",
    "portfolio_sleeves",
    "cost_estimates",
    "cost_actuals",
    "economic_decisions",
    "cost_reconciliation",
)


@dataclass(frozen=True, slots=True)
class BacktestMonitoringData:
    run_id: str
    schema_version: str
    source_fingerprint: str
    cache_token: str
    summary: dict[str, Any]
    tables: dict[str, tuple[dict[str, Any], ...]]
    integrity_verified: bool = True


@dataclass(slots=True)
class _CacheEntry:
    fingerprint: str
    cache_token: str
    data: BacktestMonitoringData


class BacktestMonitoringSource(MonitoringSource):
    """Read backtest 1.0--1.6 and linked robustness/evidence 1.7--1.8.

    The backtest artifact itself keeps the Lot 8.1 schema.  Lot 8.2 reports
    live in a separate, checksum-verified local store and are attached to the
    monitoring summary without rewriting the immutable backtest export.
    """

    def __init__(self, root: Path | str = Path("data_local/backtests")) -> None:
        self.root = Path(root)
        self.result_store = BacktestResultStore(self.root)
        self.validation_store = LocalValidationStore(self.root.parent / "validation")
        # Imported lazily to avoid coupling monitoring module import order to
        # the analytical Lot 8.2 application service.
        from trading_ai.robustness.storage import LocalRobustnessStore

        self.robustness_store = LocalRobustnessStore(self.root.parent / "robustness")
        self._cache: dict[str, _CacheEntry] = {}
        self.parquet_parse_count = 0

    def _directory(self, run_id: str) -> Path:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise MonitoringNotFoundError("invalid backtest run_id")
        root = self.root.resolve()
        directory = (self.root / run_id).resolve()
        if root not in directory.parents:
            raise MonitoringNotFoundError("invalid backtest run_id")
        return directory

    def list_runs(self) -> tuple[dict[str, Any], ...]:
        if not self.root.is_dir():
            return ()
        records: list[dict[str, Any]] = []
        for directory in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or not _SAFE_RUN_ID.fullmatch(directory.name):
                continue
            try:
                payload = self.result_store.inspect(directory.name)
            except BacktestStorageError:
                records.append(
                    {
                        "run_id": directory.name,
                        "integrity": "ERROR",
                        "status": "UNTRUSTED",
                        "schema_version": "UNAVAILABLE",
                    }
                )
                continue
            records.append(
                {
                    "run_id": directory.name,
                    "integrity": "VERIFIED",
                    "status": payload.get("status", "UNAVAILABLE"),
                    "schema_version": payload.get("schema_version", "UNAVAILABLE"),
                    "strategy_name": payload.get("strategy_name", "UNAVAILABLE"),
                    "completed_at": payload.get("completed_at"),
                }
            )
        return tuple(records)

    def load_run(self, run_id: str) -> BacktestMonitoringData:
        directory = self._directory(run_id)
        try:
            summary = redact_sensitive(self.result_store.inspect(run_id))
            latest_validation = self.validation_store.latest_for_run(run_id)
            if latest_validation is not None:
                summary["validation"] = redact_sensitive(latest_validation)
            latest_robustness = self.robustness_store.latest_for_run(run_id)
            if latest_robustness is not None:
                summary["robustness"] = redact_sensitive(latest_robustness)
            latest_evidence = self.robustness_store.latest_evidence_for_run(run_id)
            if latest_evidence is not None:
                summary["evidence"] = redact_sensitive(latest_evidence)
            checksums_bytes = (directory / "checksums.json").read_bytes()
            fingerprint = hashlib.sha256(checksums_bytes).hexdigest()
            cache_token = self.cache_token(run_id)
            cached = self._cache.get(run_id)
            if (
                cached is not None
                and cached.fingerprint == fingerprint
                and cached.cache_token == cache_token
            ):
                return cached.data
            checksums = json.loads(checksums_bytes.decode("utf-8"))["files"]
            tables: dict[str, tuple[dict[str, Any], ...]] = {}
            for name in _TABLES:
                filename = f"{name}.parquet"
                if filename not in checksums:
                    tables[name] = ()
                    continue
                tables[name] = self._read_parquet(directory / filename)
                self.parquet_parse_count += 1
        except MonitoringNotFoundError:
            raise
        except BacktestStorageError as exc:
            message = str(exc).lower()
            if "not found" in message:
                raise MonitoringNotFoundError("backtest run not found") from exc
            raise MonitoringIntegrityError(
                "backtest integrity verification failed; data is untrusted"
            ) from exc
        except FileNotFoundError as exc:
            raise MonitoringNotFoundError("backtest run not found") from exc
        except Exception as exc:
            raise MonitoringIntegrityError(
                "backtest export cannot be decoded safely"
            ) from exc
        data = BacktestMonitoringData(
            run_id=run_id,
            schema_version=str(summary.get("schema_version", "UNAVAILABLE")),
            source_fingerprint=fingerprint,
            cache_token=cache_token,
            summary=summary,
            tables=tables,
        )
        self._cache[run_id] = _CacheEntry(fingerprint, cache_token, data)
        return data

    def cache_token(self, run_id: str) -> str:
        """Cheap invalidation token; changed state triggers full SHA-256 verification."""

        directory = self._directory(run_id)
        try:
            checksum_path = directory / "checksums.json"
            checksum_bytes = checksum_path.read_bytes()
            manifest = json.loads(checksum_bytes.decode("utf-8"))
            names = tuple(sorted(str(name) for name in manifest["files"]))
            metadata = []
            for name in ("checksums.json", *names):
                stat = (directory / name).stat()
                metadata.append((name, stat.st_size, stat.st_mtime_ns))
            return stable_hash(
                {
                    "checksum_manifest": hashlib.sha256(checksum_bytes).hexdigest(),
                    "files": metadata,
                    "validation": self.validation_store.latest_for_run(run_id),
                    "robustness": self.robustness_store.latest_for_run(run_id),
                    "evidence": self.robustness_store.latest_evidence_for_run(run_id),
                }
            )
        except FileNotFoundError as exc:
            if not directory.is_dir():
                raise MonitoringNotFoundError("backtest run not found") from exc
            raise MonitoringIntegrityError("backtest export files are missing") from exc
        except Exception as exc:
            if isinstance(exc, (MonitoringNotFoundError, MonitoringIntegrityError)):
                raise
            raise MonitoringIntegrityError("backtest checksum metadata is invalid") from exc

    @staticmethod
    def _read_parquet(path: Path) -> tuple[dict[str, Any], ...]:
        import pyarrow.parquet as parquet

        return tuple(redact_sensitive(item) for item in parquet.read_table(path).to_pylist())

    def events_for_run(self, run_id: str) -> tuple[MonitoringEvent, ...]:
        return self.events_for_data(self.load_run(run_id))

    @staticmethod
    def events_for_data(data: BacktestMonitoringData) -> tuple[MonitoringEvent, ...]:
        summary = data.summary
        risk = summary.get("risk") if isinstance(summary.get("risk"), dict) else {}
        regime = summary.get("regime") if isinstance(summary.get("regime"), dict) else {}
        ml = summary.get("ml") if isinstance(summary.get("ml"), dict) else {}
        portfolio = summary.get("portfolio") if isinstance(summary.get("portfolio"), dict) else {}
        costs = summary.get("costs") if isinstance(summary.get("costs"), dict) else {}
        specs = (
            ("equity", MonitoringEventType.EQUITY_UPDATE, "EquityCurve", "1", "timestamp", None),
            ("ledger", MonitoringEventType.POSITION_UPDATE, "PortfolioLedger", "1", "timestamp", "entry_id"),
            ("signals", MonitoringEventType.SIGNAL, "Strategy", str(summary.get("strategy_version", "UNAVAILABLE")), "timestamp", "signal_id"),
            ("regime_snapshots", MonitoringEventType.REGIME, str(regime.get("detector_name", "RegimeDetector")), str(regime.get("detector_version", "UNAVAILABLE")), "timestamp", "snapshot_id"),
            ("ml_predictions", MonitoringEventType.ML_PREDICTION, "MLScorer", str(ml.get("model_version", "UNAVAILABLE")), "timestamp", "prediction_id"),
            ("ml_decisions", MonitoringEventType.ML_DECISION, "MLFilter", "1", "timestamp", "decision_id"),
            ("activation_decisions", MonitoringEventType.ACTIVATION_DECISION, str(regime.get("policy_name", "StrategyActivationPolicy")), str(regime.get("policy_version", "UNAVAILABLE")), "timestamp", "decision_id"),
            ("portfolio_decisions", MonitoringEventType.PORTFOLIO_DECISION, str(portfolio.get("engine_name", "PortfolioEngine")), str(portfolio.get("engine_version", "UNAVAILABLE")), "timestamp", "decision_id"),
            ("cost_estimates", MonitoringEventType.COST_ESTIMATE, str(costs.get("engine_name", "TransactionCostEngine")), str(costs.get("engine_version", "UNAVAILABLE")), "timestamp", "estimate_id"),
            ("economic_decisions", MonitoringEventType.ECONOMIC_DECISION, "EconomicGate", "1.0", "timestamp", "decision_id"),
            ("risk_decisions", MonitoringEventType.RISK_DECISION, str(risk.get("engine_name", "RiskEngine")), str(risk.get("engine_version", "UNAVAILABLE")), "timestamp", "decision_id"),
            ("orders", MonitoringEventType.ORDER_INTENT, "BacktestExecution", "1", "created_at", "order_id"),
            ("fills", MonitoringEventType.FILL, "BarExecutionModel", "1", "timestamp", "fill_id"),
            ("cost_actuals", MonitoringEventType.COST_ACTUAL, str(costs.get("engine_name", "TransactionCostEngine")), str(costs.get("engine_version", "UNAVAILABLE")), "timestamp", "actual_cost_id"),
            ("cost_reconciliation", MonitoringEventType.COST_RECONCILIATION, str(costs.get("engine_name", "TransactionCostEngine")), str(costs.get("engine_version", "UNAVAILABLE")), "timestamp", "reconciliation_id"),
        )
        provenance = tuple(
            sorted(
                (
                    ("result_hash", str(summary.get("result_hash", "UNAVAILABLE"))),
                    ("schema_version", data.schema_version),
                    ("source_fingerprint", data.source_fingerprint),
                )
            )
        )
        events: list[MonitoringEvent] = []
        for table_name, event_type, component, version, timestamp_field, id_field in specs:
            for row in data.tables.get(table_name, ()):
                raw_timestamp = row.get(timestamp_field)
                if raw_timestamp is None:
                    continue
                timestamp = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    continue
                timestamp = timestamp.astimezone(timezone.utc)
                entity_id = str(row.get(id_field)) if id_field and row.get(id_field) else None
                event_id = entity_id or (
                    "event-" + stable_hash(
                        {"run_id": data.run_id, "table": table_name, "row": row}
                    )[:24]
                )
                related = []
                for key, value in row.items():
                    if key.endswith("_id") and value and key != id_field:
                        related.append((str(key), str(value)))
                related_ids = tuple(sorted(set(related)))
                payload = json.dumps(
                    to_primitive(row), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                events.append(
                    MonitoringEvent(
                        event_id=f"{data.run_id}:{event_type.value}:{event_id}",
                        timestamp=timestamp,
                        event_type=event_type,
                        run_id=data.run_id,
                        session_id=data.run_id,
                        source_component=component,
                        component_version=version,
                        related_ids=related_ids,
                        provenance=provenance,
                        payload_json=payload,
                        symbol=str(row.get("symbol")) if row.get("symbol") else None,
                        strategy_name=str(row.get("strategy_name")) if row.get("strategy_name") else None,
                        status=str(row.get("status") or row.get("action")) if row.get("status") or row.get("action") else None,
                    )
                )
        validation = summary.get("validation")
        if isinstance(validation, dict) and validation.get("validation_id"):
            timestamp = datetime.fromisoformat(
                str(validation.get("created_at")).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            events.append(
                MonitoringEvent(
                    event_id=f"{data.run_id}:VALIDATION_RESULT:{validation['validation_id']}",
                    timestamp=timestamp,
                    event_type=MonitoringEventType.VALIDATION_RESULT,
                    run_id=data.run_id,
                    session_id=data.run_id,
                    source_component=str(validation.get("gate_name", "ValidationGate")),
                    component_version=str(validation.get("gate_version", "UNAVAILABLE")),
                    related_ids=(("validation_id", str(validation["validation_id"])),),
                    provenance=provenance,
                    payload_json=json.dumps(
                        to_primitive(validation), sort_keys=True,
                        separators=(",", ":"), allow_nan=False,
                    ),
                    status=str(validation.get("status", "UNAVAILABLE")),
                )
            )
        evidence = summary.get("evidence")
        if isinstance(evidence, dict) and evidence.get("reassessment_id"):
            timestamp = datetime.fromisoformat(
                str(evidence.get("created_at")).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            event_specs = (
                (
                    MonitoringEventType.EVIDENCE_CONFLICT
                    if (evidence.get("evidence_registry") or {}).get("conflicts")
                    else MonitoringEventType.EVIDENCE_VERIFIED,
                    str((evidence.get("evidence_registry") or {}).get("registry_hash", "UNAVAILABLE")),
                    "CONFLICT"
                    if (evidence.get("evidence_registry") or {}).get("conflicts")
                    else "VERIFIED",
                    evidence.get("evidence_registry") or {},
                ),
                (
                    MonitoringEventType.EVIDENCE_REASSESSMENT,
                    str(evidence["reassessment_id"]),
                    str(evidence.get("strict_validation_evidence_status", "UNAVAILABLE")),
                    evidence,
                ),
                (
                    MonitoringEventType.PAPER_READINESS_REVIEW,
                    str((evidence.get("paper_readiness_v2") or {}).get("review_id", "UNAVAILABLE")),
                    str((evidence.get("paper_readiness_v2") or {}).get("status", "UNAVAILABLE")),
                    evidence.get("paper_readiness_v2") or {},
                ),
            )
            for event_type, entity_id, status, payload_value in event_specs:
                events.append(
                    MonitoringEvent(
                        event_id=f"{data.run_id}:{event_type.value}:{entity_id}",
                        timestamp=timestamp,
                        event_type=event_type,
                        run_id=data.run_id,
                        session_id=data.run_id,
                        source_component="EvidenceClosure",
                        component_version="2.0",
                        related_ids=(("reassessment_id", str(evidence["reassessment_id"])),),
                        provenance=provenance,
                        payload_json=json.dumps(
                            to_primitive(payload_value), sort_keys=True,
                            separators=(",", ":"), allow_nan=False,
                        ),
                        status=status,
                    )
                )
        return tuple(sorted(events, key=lambda item: (item.timestamp, item.event_id)))
