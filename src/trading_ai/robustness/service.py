"""Offline application service for frozen plans and verified diagnostics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from trading_ai.core.hashing import to_primitive
from trading_ai.monitoring.source import BacktestMonitoringSource
from trading_ai.robustness.config import (
    load_research_baseline_manifest,
    load_research_plan,
    load_robustness_config,
)
from trading_ai.robustness.diagnostics import RobustnessAnalyzer
from trading_ai.robustness.exceptions import (
    HoldoutGovernanceError,
    RobustnessStorageError,
)
from trading_ai.robustness.governance import (
    BaselineReproducer,
    HoldoutAccessPolicy,
    HoldoutConsumer,
    consume_holdout,
    decision_core_hash,
    make_untouched_holdout,
    observed_decision_config_hashes,
)
from trading_ai.robustness.models import (
    HoldoutRecord,
    HoldoutStatus,
    PeriodClassification,
    ResearchPeriod,
)
from trading_ai.robustness.readiness import PaperReadinessReviewer
from trading_ai.robustness.storage import LocalRobustnessStore
from trading_ai.validation import LocalValidationStore


class RobustnessService:
    """Join trusted exports and analytics; it never imports a trading engine."""

    def __init__(self, data_root: Path | str = Path("data_local")) -> None:
        self.data_root = Path(data_root)
        self.source = BacktestMonitoringSource(self.data_root / "backtests")
        self.store = LocalRobustnessStore(self.data_root / "robustness")
        self.config = load_robustness_config()
        self.baseline = load_research_baseline_manifest(
            self.config.baseline_manifest_path
        )

    def freeze_plan(self) -> dict[str, Any]:
        plan = load_research_plan(
            self.config.research_plan_path, baseline=self.baseline
        )
        record = self._current_holdout(plan)
        self._require_frozen_plan_match(record, plan)
        plan_path = self.store.save_plan(plan)
        holdout_path = self.store.root / "holdouts" / record.holdout_id
        return {
            "baseline": to_primitive(self.baseline),
            "plan": to_primitive(plan),
            "holdout": to_primitive(record),
            "plan_path": str(plan_path),
            "holdout_path": str(holdout_path),
            "network_access": False,
            "thresholds_relaxed": False,
        }

    def run(
        self,
        run_id: str,
        *,
        period_classification: PeriodClassification,
        leave_one_symbol_run_ids: Mapping[str, str] | None = None,
        leave_one_strategy_run_ids: Mapping[str, str] | None = None,
        single_strategy_run_ids: Mapping[str, str] | None = None,
        holdout_status: HoldoutStatus | None = None,
    ) -> dict[str, Any]:
        plan = load_research_plan(
            self.config.research_plan_path, baseline=self.baseline
        )
        record: HoldoutRecord | None = None
        if period_classification is PeriodClassification.FINAL_HOLDOUT:
            record = self._current_holdout(plan)
            self._require_frozen_plan_match(record, plan)
            HoldoutAccessPolicy.authorize(
                record,
                HoldoutConsumer.FINAL_EVALUATION
                if record.status is HoldoutStatus.UNTOUCHED
                else HoldoutConsumer.REPRODUCIBILITY,
            )
        data = self.source.load_run(run_id)
        baseline_data = (
            data
            if run_id == self.baseline.run_id
            else self.source.load_run(self.baseline.run_id)
        )
        frozen_baseline_verified = BaselineReproducer.verify(
            self.baseline,
            summary=baseline_data.summary,
            integrity_verified=baseline_data.integrity_verified,
        )
        validation = LocalValidationStore(
            self.data_root / "validation"
        ).latest_for_run(run_id)
        validation_status = (
            str(validation.get("status", "UNAVAILABLE"))
            if isinstance(validation, dict)
            else "UNAVAILABLE"
        )
        if record is not None:
            self._verify_holdout_period(data.summary, record.period)
            record, _ = consume_holdout(
                record,
                result_hash=str(data.summary.get("result_hash")),
                core_hash=decision_core_hash(),
                config_hashes=observed_decision_config_hashes(data.summary, plan),
            )
            self.store.save_holdout(record)
            holdout_status = record.status
        self.store.save_plan(plan)
        symbol_runs = self._load_summaries(leave_one_symbol_run_ids or {})
        strategy_runs = self._load_summaries(leave_one_strategy_run_ids or {})
        single_runs = self._load_summaries(single_strategy_run_ids or {})
        self._validate_leave_one_runs(
            symbol_runs,
            expected_items=plan.symbols,
            expected_symbols=plan.symbols,
            expected_strategies=plan.strategies,
            expected_config_hashes=self.baseline.config_hashes,
            reference_strategy_parameters=baseline_data.summary.get(
                "strategy_parameters", ()
            ),
            period=plan.period(PeriodClassification.CONSUMED_DIAGNOSTIC),
            kind="symbol",
        )
        self._validate_leave_one_runs(
            strategy_runs,
            expected_items=plan.strategies,
            expected_symbols=plan.symbols,
            expected_strategies=plan.strategies,
            expected_config_hashes=self.baseline.config_hashes,
            reference_strategy_parameters=baseline_data.summary.get(
                "strategy_parameters", ()
            ),
            period=plan.period(PeriodClassification.CONSUMED_DIAGNOSTIC),
            kind="strategy",
        )
        self._validate_single_strategy_runs(
            single_runs,
            expected_strategies=plan.strategies,
            expected_symbols=plan.symbols,
            expected_config_hashes=self.baseline.config_hashes,
            reference_strategy_parameters=baseline_data.summary.get(
                "strategy_parameters", ()
            ),
            period=plan.period(PeriodClassification.CONSUMED_DIAGNOSTIC),
        )
        report = RobustnessAnalyzer(self.config).analyze(
            summary=data.summary,
            tables=data.tables,
            integrity_verified=data.integrity_verified,
            baseline=self.baseline,
            plan=plan,
            period_classification=period_classification,
            frozen_baseline_verified=frozen_baseline_verified,
            holdout_status=holdout_status,
            validation_status=validation_status,
            leave_one_symbol_runs=symbol_runs,
            leave_one_strategy_runs=strategy_runs,
            single_strategy_runs=single_runs,
        )
        readiness = PaperReadinessReviewer().review(
            report, validation_status=validation_status
        )
        path = self.store.save_report(
            report, baseline=self.baseline, plan=plan, readiness=readiness
        )
        return {
            "report": to_primitive(report),
            "paper_readiness": to_primitive(readiness),
            "report_path": str(path),
            "network_access": False,
            "trading_pipeline_mutated": False,
        }

    def inspect(self, report_id: str) -> dict[str, Any]:
        return self.store.inspect_report(report_id)

    def latest_for_run(self, run_id: str) -> dict[str, Any] | None:
        return self.store.latest_for_run(run_id)

    def holdout_status(self) -> dict[str, Any]:
        plan = load_research_plan(
            self.config.research_plan_path, baseline=self.baseline
        )
        return to_primitive(self._current_holdout(plan))

    def _current_holdout(self, plan) -> HoldoutRecord:
        current_core_hash = decision_core_hash()
        record = make_untouched_holdout(plan, core_hash=current_core_hash)
        existing_for_period = self.store.find_holdout_for_period(record.period)
        if existing_for_period is not None:
            existing = deserialize_holdout(existing_for_period)
            if (
                existing.status is HoldoutStatus.CONSUMED
                and existing.expected_core_hash != current_core_hash
            ):
                assert existing.result_hash is not None
                invalidated, _ = consume_holdout(
                    existing,
                    result_hash=existing.result_hash,
                    core_hash=current_core_hash,
                    config_hashes=existing.expected_config_hashes,
                )
                self.store.save_holdout(invalidated)
                return invalidated
            return existing
        try:
            return deserialize_holdout(self.store.inspect_holdout(record.holdout_id))
        except RobustnessStorageError:
            self.store.save_plan(plan)
            self.store.save_holdout(record)
            return record

    @staticmethod
    def _require_frozen_plan_match(record: HoldoutRecord, plan) -> None:
        if record.plan_hash != plan.plan_hash:
            raise HoldoutGovernanceError(
                "the holdout period belongs to a different frozen plan hash; "
                "it cannot become new FINAL evidence after retuning"
            )
        if (
            record.status is HoldoutStatus.UNTOUCHED
            and record.expected_core_hash != decision_core_hash()
        ):
            raise HoldoutGovernanceError(
                "the decision core changed after the untouched holdout was frozen"
            )

    @staticmethod
    def _verify_holdout_period(summary: dict[str, Any], period: ResearchPeriod) -> None:
        references = summary.get("dataset_references")
        if not isinstance(references, list) or not references:
            raise ValueError("FINAL_HOLDOUT requires exact dataset provenance")
        starts = {str(item.get("requested_start")) for item in references}
        ends = {str(item.get("requested_end")) for item in references}
        expected_start = period.start.isoformat()
        expected_end = period.end.isoformat()
        if starts != {expected_start} or ends != {expected_end}:
            raise ValueError(
                "run period does not exactly match the frozen FINAL_HOLDOUT"
            )

    def _load_summaries(self, mapping: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        return {
            item: self.source.load_run(run_id).summary
            for item, run_id in sorted(mapping.items())
        }

    @staticmethod
    def _strategy_names(summary: dict[str, Any]) -> set[str]:
        result: set[str] = set()
        for item in summary.get("strategy_parameters", ()):
            if not isinstance(item, (list, tuple)) or not item:
                continue
            key = str(item[0])
            if key.startswith("strategy.") and key.count(".") >= 2:
                result.add(key.split(".", 2)[1])
        if not result:
            strategy_name = str(summary.get("strategy_name", ""))
            if strategy_name and strategy_name != "multi-strategy-portfolio":
                result.add(strategy_name)
        return result

    @staticmethod
    def _strategy_parameters(raw: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for item in raw if isinstance(raw, (list, tuple)) else ():
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            result[str(item[0])] = str(item[1])
        return result

    @classmethod
    def _validate_leave_one_runs(
        cls,
        summaries: Mapping[str, dict[str, Any]],
        *,
        expected_items: tuple[str, ...],
        expected_symbols: tuple[str, ...],
        expected_strategies: tuple[str, ...],
        expected_config_hashes: tuple[tuple[str, str], ...] | None = None,
        reference_strategy_parameters: Any = (),
        period: ResearchPeriod,
        kind: str,
    ) -> None:
        """Require one exact post-hoc exclusion on the consumed V1 period."""

        allowed = set(expected_items)
        for excluded, summary in summaries.items():
            if excluded not in allowed:
                raise ValueError(f"unknown leave-one-{kind} item: {excluded}")
            references = summary.get("dataset_references")
            if not isinstance(references, list) or not references:
                raise ValueError(f"leave-one-{kind} run lacks exact dataset provenance")
            starts = {str(item.get("requested_start")) for item in references}
            ends = {str(item.get("requested_end")) for item in references}
            if starts != {period.start.isoformat()} or ends != {period.end.isoformat()}:
                raise ValueError(
                    f"leave-one-{kind} run must use the frozen consumed diagnostic period"
                )
            symbols = {str(item.get("symbol")) for item in references}
            strategies = cls._strategy_names(summary)
            expected_hashes = dict(expected_config_hashes or ())
            shared_hashes = {
                "cost": (summary.get("costs") or {}).get("config_hash"),
                "portfolio": (summary.get("portfolio") or {}).get("config_hash"),
                "regime": (summary.get("regime") or {}).get("config_hash"),
                "policy": (summary.get("regime") or {}).get("policy_config_hash"),
                "risk": (summary.get("risk") or {}).get("config_hash"),
            }
            if expected_hashes and any(
                shared_hashes[name] != expected_hashes.get(name)
                for name in shared_hashes
            ):
                raise ValueError(
                    f"leave-one-{kind} run changed a frozen shared configuration"
                )
            observed_parameters = cls._strategy_parameters(
                summary.get("strategy_parameters", ())
            )
            reference_parameters = cls._strategy_parameters(
                reference_strategy_parameters
            )
            if kind == "symbol":
                if symbols != allowed - {excluded}:
                    raise ValueError(
                        "leave-one-symbol run must remove exactly its declared symbol"
                    )
                if strategies != set(expected_strategies):
                    raise ValueError(
                        "leave-one-symbol run must preserve every frozen strategy"
                    )
                if reference_parameters:
                    if set(observed_parameters) != set(reference_parameters):
                        raise ValueError(
                            "leave-one-symbol run changed strategy parameter keys"
                        )
                    for key, reference_value in reference_parameters.items():
                        observed_value = observed_parameters[key]
                        if key.endswith(".symbols"):
                            if set(observed_value.split(",")) != set(expected_symbols) - {excluded}:
                                raise ValueError(
                                    "leave-one-symbol strategy universe is inconsistent"
                                )
                        elif observed_value != reference_value:
                            raise ValueError(
                                "leave-one-symbol run changed a strategy parameter"
                            )
            elif kind == "strategy":
                if symbols != set(expected_symbols):
                    raise ValueError(
                        "leave-one-strategy run must preserve the complete frozen universe"
                    )
                if strategies != allowed - {excluded}:
                    raise ValueError(
                        "leave-one-strategy run must remove exactly its declared strategy"
                    )
                if reference_parameters:
                    expected_parameters = {
                        key: value
                        for key, value in reference_parameters.items()
                        if not key.startswith(f"strategy.{excluded}.")
                    }
                    if observed_parameters != expected_parameters:
                        raise ValueError(
                            "leave-one-strategy run changed remaining strategy parameters"
                        )
            else:  # pragma: no cover - private caller supplies the literal kinds
                raise AssertionError(f"unsupported leave-one kind: {kind}")

    @classmethod
    def _validate_single_strategy_runs(
        cls,
        summaries: Mapping[str, dict[str, Any]],
        *,
        expected_strategies: tuple[str, ...],
        expected_symbols: tuple[str, ...],
        expected_config_hashes: tuple[tuple[str, str], ...],
        reference_strategy_parameters: Any,
        period: ResearchPeriod,
    ) -> None:
        """Require an exact one-strategy run on the consumed V1 period."""

        allowed = set(expected_strategies)
        reference_parameters = cls._strategy_parameters(reference_strategy_parameters)
        expected_hashes = dict(expected_config_hashes)
        for strategy, summary in summaries.items():
            if strategy not in allowed:
                raise ValueError(f"unknown single-strategy item: {strategy}")
            references = summary.get("dataset_references")
            if not isinstance(references, list) or not references:
                raise ValueError("single-strategy run lacks exact dataset provenance")
            starts = {str(item.get("requested_start")) for item in references}
            ends = {str(item.get("requested_end")) for item in references}
            if starts != {period.start.isoformat()} or ends != {period.end.isoformat()}:
                raise ValueError(
                    "single-strategy run must use the frozen consumed diagnostic period"
                )
            symbols = {str(item.get("symbol")) for item in references}
            if symbols != set(expected_symbols):
                raise ValueError("single-strategy run must preserve the frozen universe")
            if cls._strategy_names(summary) != {strategy}:
                raise ValueError("single-strategy run must contain exactly its declared strategy")
            shared_hashes = {
                "cost": (summary.get("costs") or {}).get("config_hash"),
                "regime": (summary.get("regime") or {}).get("config_hash"),
                "policy": (summary.get("regime") or {}).get("policy_config_hash"),
                "risk": (summary.get("risk") or {}).get("config_hash"),
            }
            if any(
                shared_hashes[name] != expected_hashes.get(name)
                for name in shared_hashes
            ):
                raise ValueError("single-strategy run changed a frozen common configuration")
            prefix = f"strategy.{strategy}."
            observed = {
                key.removeprefix(prefix): value
                for key, value in cls._strategy_parameters(
                    summary.get("strategy_parameters", ())
                ).items()
            }
            expected = {
                key.removeprefix(prefix): value
                for key, value in reference_parameters.items()
                if key.startswith(prefix)
            }
            if observed != expected:
                raise ValueError("single-strategy run changed frozen strategy parameters")


def deserialize_holdout(payload: dict[str, Any]) -> HoldoutRecord:
    """Typed conversion used by explicit holdout lifecycle commands/tests."""

    period_raw = payload["period"]
    period = ResearchPeriod(
        name=str(period_raw["name"]),
        classification=PeriodClassification(str(period_raw["classification"])),
        start=datetime.fromisoformat(str(period_raw["start"]).replace("Z", "+00:00")),
        end=datetime.fromisoformat(str(period_raw["end"]).replace("Z", "+00:00")),
        label=str(period_raw["label"]),
    )
    return HoldoutRecord(
        holdout_id=str(payload["holdout_id"]),
        plan_hash=str(payload["plan_hash"]),
        period=period,
        status=HoldoutStatus(str(payload["status"])),
        expected_core_hash=str(payload["expected_core_hash"]),
        expected_config_hashes=tuple(
            (str(item[0]), str(item[1])) for item in payload["expected_config_hashes"]
        ),
        record_hash=str(payload["record_hash"]),
        consumed_at=(
            datetime.fromisoformat(str(payload["consumed_at"]).replace("Z", "+00:00"))
            if payload.get("consumed_at") else None
        ),
        result_hash=str(payload["result_hash"]) if payload.get("result_hash") else None,
        invalidated_at=(
            datetime.fromisoformat(str(payload["invalidated_at"]).replace("Z", "+00:00"))
            if payload.get("invalidated_at") else None
        ),
        invalidation_reason=(
            str(payload["invalidation_reason"])
            if payload.get("invalidation_reason") else None
        ),
    )
