from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.experiments.baselines import heuristic_prediction_records
from vipragsent.experiments.constants import (
    ABLATION_ROWS,
    COST_BREAKDOWN,
    LOW_RESOURCE_BUDGETS,
    METRIC_FIELDS,
    ORDINARY_DATASETS,
    Q1_SYSTEMS,
    SMOKE_SYSTEM,
    SYSTEM_LABELS,
    SYSTEM_REQUIRES_FINETUNING,
)
from vipragsent.experiments.evaluator import (
    ExperimentDataError,
    aggregate_seed_reports,
    evaluate_calibration_records,
    evaluate_confusion_records,
    evaluate_polarity_records,
    evaluate_pragmatic_records,
    load_gold_records,
)
from vipragsent.experiments.predictions import (
    PredictionError,
    join_gold_predictions,
    load_prediction_records,
    validate_pragmatic_predictions,
)
from vipragsent.utils.io import dump_json, load_json


def run_no_finetune_suite(
    *,
    root: str | Path = ".",
    predictions_dir: str | Path | None = None,
    vipragsent_test: str | Path | None = None,
    public_data: str | Path | None = None,
    output_dir: str | Path | None = None,
    include_heuristic: bool = False,
    allow_silver_gold: bool = False,
    bootstrap_resamples: int = 1000,
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(root)
    predictions_dir = Path(predictions_dir) if predictions_dir else root / "results" / "predictions"
    output_dir = Path(output_dir) if output_dir else root / "results"
    vipragsent_test = Path(vipragsent_test) if vipragsent_test else _default_vipragsent_test(root)
    public_data = Path(public_data) if public_data else root / "data" / "processed" / "all_unified.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"status": "ok", "outputs": {}}

    main_result, main_joined = build_main_pragmatic_result(
        gold_path=vipragsent_test,
        predictions_dir=predictions_dir,
        include_heuristic=include_heuristic,
        allow_silver_gold=allow_silver_gold,
        bootstrap_resamples=bootstrap_resamples,
        limit=limit,
    )
    _add_run_context(main_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit, gold_path=vipragsent_test)
    dump_json(output_dir / "main_pragmatic.json", main_result)
    summary["outputs"]["main_pragmatic"] = str(output_dir / "main_pragmatic.json")

    ordinary_result = build_ordinary_sentiment_result(
        public_data_path=public_data,
        predictions_dir=predictions_dir,
        allow_silver_gold=allow_silver_gold,
        bootstrap_resamples=bootstrap_resamples,
        limit=limit,
    )
    _add_run_context(ordinary_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit, gold_path=public_data)
    dump_json(output_dir / "ordinary_sentiment.json", ordinary_result)
    summary["outputs"]["ordinary_sentiment"] = str(output_dir / "ordinary_sentiment.json")

    ablation_result = build_ablation_result(predictions_dir=predictions_dir)
    _add_run_context(ablation_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit)
    dump_json(output_dir / "multitask_ablation.json", ablation_result)
    summary["outputs"]["multitask_ablation"] = str(output_dir / "multitask_ablation.json")

    low_resource_result = build_low_resource_result(
        gold_path=vipragsent_test,
        predictions_dir=predictions_dir,
        allow_silver_gold=allow_silver_gold,
        bootstrap_resamples=bootstrap_resamples,
        limit=limit,
    )
    _add_run_context(low_resource_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit, gold_path=vipragsent_test)
    dump_json(output_dir / "low_resource_sarcasm.json", low_resource_result)
    summary["outputs"]["low_resource_sarcasm"] = str(output_dir / "low_resource_sarcasm.json")

    calibration_result = build_calibration_result(main_joined)
    _add_run_context(calibration_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit, gold_path=vipragsent_test)
    dump_json(output_dir / "calibration.json", calibration_result)
    summary["outputs"]["calibration"] = str(output_dir / "calibration.json")

    confusion_result = build_confusion_result(main_joined)
    _add_run_context(confusion_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit, gold_path=vipragsent_test)
    dump_json(output_dir / "error_confusion.json", confusion_result)
    summary["outputs"]["error_confusion"] = str(output_dir / "error_confusion.json")

    learning_result = build_learning_curves_result(predictions_dir=predictions_dir)
    _add_run_context(learning_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit)
    dump_json(output_dir / "learning_curves.json", learning_result)
    summary["outputs"]["learning_curves"] = str(output_dir / "learning_curves.json")

    cost_result = _result_envelope("cost_breakdown", status="complete")
    _add_run_context(cost_result, predictions_dir, allow_silver_gold=allow_silver_gold, limit=limit)
    cost_result["cost"] = COST_BREAKDOWN
    dump_json(output_dir / "cost_breakdown.json", cost_result)
    summary["outputs"]["cost_breakdown"] = str(output_dir / "cost_breakdown.json")

    return summary


def build_main_pragmatic_result(
    *,
    gold_path: str | Path,
    predictions_dir: str | Path,
    include_heuristic: bool,
    allow_silver_gold: bool,
    bootstrap_resamples: int,
    limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    result = _result_envelope("q1_main_pragmatic")
    result["metric"] = "binary_macro_f1_per_phenomenon"
    result["systems"] = {}
    joined_by_system: dict[str, list[dict[str, Any]]] = {}
    try:
        gold_records = load_gold_records(
            gold_path,
            task="pragmatic",
            require_adjudicated=True,
            allow_silver=allow_silver_gold,
            limit=limit,
        )
    except ExperimentDataError as exc:
        result["status"] = "blocked"
        result["blocked_reason"] = str(exc)
        for system in Q1_SYSTEMS:
            result["systems"][system.system_id] = _blocked_system(system.system_id, predictions_dir, str(exc))
        return result, joined_by_system

    systems = list(Q1_SYSTEMS)
    if include_heuristic:
        systems.append(SMOKE_SYSTEM)
    for system in systems:
        if system.system_id == SMOKE_SYSTEM.system_id:
            prediction_records = heuristic_prediction_records(gold_records, system=system.system_id)
            try:
                joined = join_gold_predictions(gold_records, prediction_records)
            except PredictionError as exc:
                result["systems"][system.system_id] = _failed_system(system.system_id, str(exc))
                continue
            report = evaluate_pragmatic_records(joined, bootstrap_resamples=bootstrap_resamples)
            result["systems"][system.system_id] = _complete_system(system.system_id, [report], [None])
            joined_by_system[system.system_id] = joined
            continue

        files = _prediction_files(predictions_dir, "main_pragmatic", system.system_id)
        if not files:
            result["systems"][system.system_id] = _blocked_system(
                system.system_id,
                predictions_dir,
                "missing prediction JSONL",
                experiment="main_pragmatic",
            )
            continue
        system_result, joined = _evaluate_pragmatic_files(
            system.system_id,
            files,
            gold_records,
            bootstrap_resamples=bootstrap_resamples,
        )
        result["systems"][system.system_id] = system_result
        if joined:
            joined_by_system[system.system_id] = joined
    result["status"] = _overall_status(result["systems"])
    return result, joined_by_system


def build_ordinary_sentiment_result(
    *,
    public_data_path: str | Path,
    predictions_dir: str | Path,
    allow_silver_gold: bool,
    bootstrap_resamples: int,
    limit: int | None = None,
) -> dict[str, Any]:
    result = _result_envelope("q1_ordinary_sentiment_retention")
    result["metric"] = "polarity_macro_f1"
    result["datasets"] = {}
    for dataset in ORDINARY_DATASETS:
        dataset_result = {"systems": {}, "status": "pending"}
        try:
            all_records = load_gold_records(
                public_data_path,
                split="test",
                task="polarity",
                require_adjudicated=False,
                allow_silver=allow_silver_gold,
                limit=None,
            )
            gold_records = [
                record for record in all_records if (record.get("source") or {}).get("dataset") == dataset
            ]
            if limit is not None:
                gold_records = gold_records[:limit]
            if not gold_records:
                raise ExperimentDataError(f"no polarity test records for dataset {dataset}")
        except ExperimentDataError as exc:
            dataset_result["status"] = "blocked"
            dataset_result["blocked_reason"] = str(exc)
            result["datasets"][dataset] = dataset_result
            continue
        for system in Q1_SYSTEMS:
            files = _prediction_files(predictions_dir, f"ordinary_sentiment/{dataset}", system.system_id)
            if not files:
                dataset_result["systems"][system.system_id] = _blocked_system(
                    system.system_id,
                    predictions_dir,
                    "missing prediction JSONL",
                    experiment=f"ordinary_sentiment/{dataset}",
                )
                continue
            seed_reports = []
            errors = []
            for path in files:
                try:
                    joined = join_gold_predictions(gold_records, load_prediction_records(path))
                    report = evaluate_polarity_records(joined, bootstrap_resamples=bootstrap_resamples)
                    report["prediction_file"] = str(path)
                    seed_reports.append(report)
                except (PredictionError, ValueError) as exc:
                    errors.append({"prediction_file": str(path), "error": str(exc)})
            dataset_result["systems"][system.system_id] = _system_from_seed_reports(
                system.system_id,
                seed_reports,
                ["polarity_macro_f1"],
                errors=errors,
            )
        dataset_result["status"] = _overall_status(dataset_result["systems"])
        result["datasets"][dataset] = dataset_result
    result["status"] = _overall_status({name: value for name, value in result["datasets"].items()})
    return result


def build_ablation_result(*, predictions_dir: str | Path) -> dict[str, Any]:
    result = _result_envelope("q2_multitask_ablation")
    summary_path = Path(predictions_dir) / "multitask_ablation" / "summary.json"
    if summary_path.exists():
        payload = load_json(summary_path)
        result["status"] = "complete"
        result["rows"] = payload.get("rows", payload)
        result["provenance"]["source_file"] = str(summary_path)
        return result
    result["status"] = "blocked"
    result["blocked_reason"] = "Ablation rows require fine-tuned run summaries or imported prediction metrics."
    result["rows"] = {
        row: {
            "status": "blocked",
            "reason": "missing multitask ablation summary",
            "expected_source": str(summary_path),
        }
        for row in ABLATION_ROWS
    }
    return result


def build_low_resource_result(
    *,
    gold_path: str | Path,
    predictions_dir: str | Path,
    allow_silver_gold: bool,
    bootstrap_resamples: int,
    limit: int | None = None,
) -> dict[str, Any]:
    result = _result_envelope("q3_low_resource_sarcasm")
    result["metric"] = "sarcasm_binary_macro_f1"
    result["budgets"] = {}
    try:
        gold_records = load_gold_records(
            gold_path,
            task="pragmatic",
            require_adjudicated=True,
            allow_silver=allow_silver_gold,
            limit=limit,
        )
    except ExperimentDataError as exc:
        result["status"] = "blocked"
        result["blocked_reason"] = str(exc)
        return result
    systems = ["phobert_finetune", "xlmr_large", "vistral_7b_sft", "gpt41_mini_8_shot", "vipragsent_full"]
    for budget in LOW_RESOURCE_BUDGETS:
        budget_key = str(budget)
        result["budgets"][budget_key] = {"systems": {}, "positive_budget": budget}
        for system_id in systems:
            files = _prediction_files(predictions_dir, f"low_resource_sarcasm/{budget}", system_id)
            if not files:
                result["budgets"][budget_key]["systems"][system_id] = _blocked_system(
                    system_id,
                    predictions_dir,
                    "missing prediction JSONL",
                    experiment=f"low_resource_sarcasm/{budget}",
                )
                continue
            system_result, _ = _evaluate_pragmatic_files(
                system_id,
                files,
                gold_records,
                bootstrap_resamples=bootstrap_resamples,
                metric_names=["sarcasm"],
            )
            result["budgets"][budget_key]["systems"][system_id] = system_result
        result["budgets"][budget_key]["status"] = _overall_status(result["budgets"][budget_key]["systems"])
    result["status"] = _overall_status({key: value for key, value in result["budgets"].items()})
    return result


def build_calibration_result(joined_by_system: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = _result_envelope("q4_calibration")
    result["target"] = "pragmatic_polarity"
    result["systems"] = {}
    for system_id, joined in joined_by_system.items():
        report = evaluate_calibration_records(joined, target="pragmatic_polarity", bins=10)
        status = "complete" if report["n"] > 0 else "blocked"
        result["systems"][system_id] = {
            "label": SYSTEM_LABELS.get(system_id, system_id),
            "status": status,
            "ece": round(report["ece"], 6),
            "bins": report["bins"],
            "n": report["n"],
            "missing_confidence": report["missing_confidence"],
        }
    if not result["systems"]:
        result["status"] = "blocked"
        result["blocked_reason"] = "No evaluated system with confidence scores is available."
    else:
        result["status"] = _overall_status(result["systems"])
    return result


def build_confusion_result(joined_by_system: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = _result_envelope("q4_error_confusion")
    preferred = "vipragsent_full" if "vipragsent_full" in joined_by_system else None
    if preferred is None and joined_by_system:
        preferred = sorted(joined_by_system)[0]
    if preferred is None:
        result["status"] = "blocked"
        result["blocked_reason"] = "No evaluated system is available for confusion matrix."
        return result
    result.update(evaluate_confusion_records(joined_by_system[preferred]))
    result["status"] = "complete"
    result["system"] = preferred
    result["label"] = SYSTEM_LABELS.get(preferred, preferred)
    return result


def build_learning_curves_result(*, predictions_dir: str | Path) -> dict[str, Any]:
    result = _result_envelope("q4_learning_curves")
    summary_path = Path(predictions_dir) / "learning_curves" / "summary.json"
    if summary_path.exists():
        payload = load_json(summary_path)
        result["status"] = "complete"
        result["curves"] = payload.get("curves", payload)
        result["provenance"]["source_file"] = str(summary_path)
        return result
    result["status"] = "blocked"
    result["blocked_reason"] = "Learning curves require per-epoch logs from fine-tuned runs."
    result["expected_source"] = str(summary_path)
    return result


def _evaluate_pragmatic_files(
    system_id: str,
    files: list[Path],
    gold_records: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    metric_names: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    seed_reports = []
    errors = []
    first_joined = None
    for path in files:
        try:
            prediction_records = load_prediction_records(path)
            validation_errors = validate_pragmatic_predictions(prediction_records)
            if validation_errors:
                raise PredictionError("; ".join(validation_errors[:5]))
            joined = join_gold_predictions(gold_records, prediction_records)
            report = evaluate_pragmatic_records(joined, bootstrap_resamples=bootstrap_resamples)
            report["prediction_file"] = str(path)
            report["seed"] = _seed_from_report(prediction_records, path)
            seed_reports.append(report)
            if first_joined is None:
                first_joined = joined
        except (PredictionError, ValueError) as exc:
            errors.append({"prediction_file": str(path), "error": str(exc)})
    metrics = metric_names or METRIC_FIELDS
    return _system_from_seed_reports(system_id, seed_reports, metrics, errors=errors), first_joined


def _system_from_seed_reports(
    system_id: str,
    seed_reports: list[dict[str, Any]],
    metric_names: list[str],
    *,
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not seed_reports:
        return _failed_system(system_id, "no usable prediction files", errors=errors or [])
    out = _complete_system(system_id, seed_reports, [report.get("prediction_file") for report in seed_reports])
    out["metrics"] = aggregate_seed_reports(seed_reports, metric_names)
    if errors:
        out["status"] = "partial"
        out["errors"] = errors
    return out


def _complete_system(system_id: str, seed_reports: list[dict[str, Any]], prediction_files: list[str | None]) -> dict[str, Any]:
    return {
        "label": SYSTEM_LABELS.get(system_id, system_id),
        "status": "complete",
        "requires_finetuning": SYSTEM_REQUIRES_FINETUNING.get(system_id, False),
        "prediction_files": [path for path in prediction_files if path],
        "runs": seed_reports,
        "metrics": aggregate_seed_reports(seed_reports, METRIC_FIELDS),
    }


def _blocked_system(
    system_id: str,
    predictions_dir: str | Path,
    reason: str,
    *,
    experiment: str = "main_pragmatic",
) -> dict[str, Any]:
    return {
        "label": SYSTEM_LABELS.get(system_id, system_id),
        "status": "blocked",
        "requires_finetuning": SYSTEM_REQUIRES_FINETUNING.get(system_id, False),
        "reason": reason,
        "expected_prediction_dir": str(Path(predictions_dir) / experiment / system_id),
    }


def _failed_system(system_id: str, reason: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    out = {
        "label": SYSTEM_LABELS.get(system_id, system_id),
        "status": "failed",
        "requires_finetuning": SYSTEM_REQUIRES_FINETUNING.get(system_id, False),
        "reason": reason,
    }
    if errors:
        out["errors"] = errors
    return out


def _prediction_files(predictions_dir: str | Path, experiment: str, system_id: str) -> list[Path]:
    base = Path(predictions_dir)
    patterns = [
        base / experiment / system_id / "*.jsonl",
        base / experiment / f"{system_id}*.jsonl",
        base / f"{experiment}_{system_id}*.jsonl",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(pattern.parent.glob(pattern.name))
    return sorted({path.resolve() for path in files})


def _seed_from_report(prediction_records: list[dict[str, Any]], path: Path) -> int | str | None:
    for record in prediction_records:
        if record.get("seed") is not None:
            return record["seed"]
    digits = "".join(char for char in path.stem if char.isdigit())
    if len(digits) >= 6:
        try:
            return int(digits[-8:])
        except ValueError:
            return path.stem
    return path.stem


def _overall_status(items: dict[str, Any]) -> str:
    statuses = {str(item.get("status")) for item in items.values() if isinstance(item, dict)}
    if not statuses:
        return "blocked"
    if statuses == {"complete"}:
        return "complete"
    if "complete" in statuses or "partial" in statuses:
        return "partial"
    if "failed" in statuses:
        return "failed"
    return "blocked"


def _default_vipragsent_test(root: Path) -> Path:
    candidates = [
        root / "data" / "processed" / "vipragsent_test.jsonl",
        root / "data" / "processed" / "vipragsent" / "test.jsonl",
        root / "data" / "processed" / "all_unified.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _result_envelope(experiment: str, *, status: str = "pending") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment": experiment,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "runner": "vipragsent.experiments.runner",
            "fine_tuning": "excluded",
            "note": "This runner evaluates imported predictions and no-train baselines only.",
        },
    }


def _add_run_context(
    result: dict[str, Any],
    predictions_dir: str | Path,
    *,
    allow_silver_gold: bool,
    limit: int | None,
    gold_path: str | Path | None = None,
) -> None:
    provenance = result.setdefault("provenance", {})
    provenance["predictions_dir"] = str(predictions_dir)
    provenance["allow_silver_gold"] = allow_silver_gold
    provenance["record_limit"] = limit
    if gold_path is not None:
        provenance["gold_path"] = str(gold_path)
    if allow_silver_gold:
        provenance["warning"] = (
            "This result may use silver/reviewed labels as gold for smoke testing; "
            "do not cite it as a final paper metric."
        )
