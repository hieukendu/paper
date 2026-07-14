from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

from vipragsent.data.schema import (
    EMOTION_LABELS,
    POLARITY_LABELS,
    PRAGMATIC_LABELS,
    available_labels_from,
    assert_valid_record,
)
from vipragsent.evaluation.bootstrap import bootstrap_ci
from vipragsent.evaluation.calibration import expected_calibration_error
from vipragsent.evaluation.confusion import confusion_matrix, row_normalise
from vipragsent.evaluation.metrics import multiclass_macro_f1, pragmatic_f1
from vipragsent.experiments.constants import PRAGMATIC_POLARITY_CLASSES
from vipragsent.experiments.predictions import confidence_for_class, pragmatic_polarity_class
from vipragsent.utils.io import read_jsonl


class ExperimentDataError(ValueError):
    """Raised when a gold or prediction dataset cannot be evaluated."""


def load_gold_records(
    path: str | Path,
    *,
    split: str | None = None,
    task: str = "pragmatic",
    require_adjudicated: bool = True,
    allow_silver: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ExperimentDataError(f"gold file does not exist: {path}")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for record in read_jsonl(path):
        if split and record.get("split") != split and record.get("type") != split:
            continue
        if require_adjudicated and not allow_silver and record.get("label_status") != "adjudicated":
            continue
        labels = record.get("labels") or {}
        available = available_labels_from(labels)
        if task == "pragmatic" and not all(available.values()):
            continue
        if task == "polarity" and not available["polarity"]:
            continue
        if task == "emotion" and not available["emotion"]:
            continue
        try:
            assert_valid_record(
                record,
                require_adjudicated_gold=require_adjudicated and not allow_silver,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if errors and not records:
        raise ExperimentDataError("; ".join(errors[:5]))
    if not records:
        adjudication_note = "adjudicated " if require_adjudicated and not allow_silver else ""
        raise ExperimentDataError(f"no {adjudication_note}{task} gold records found in {path}")
    return records


def evaluate_pragmatic_records(joined_records: list[dict[str, Any]], *, bootstrap_resamples: int = 1000) -> dict[str, Any]:
    golds = [record["gold"] for record in joined_records]
    preds = [record["predictions"] for record in joined_records]
    point = pragmatic_f1(golds, preds)
    metrics = {}
    for field in [*PRAGMATIC_LABELS, "macro_pragmatic_f1"]:
        metric_fn = _pragmatic_metric_fn(field)
        lo, hi = bootstrap_ci(joined_records, metric_fn=metric_fn, resamples=bootstrap_resamples)
        metrics[field] = {
            "value": round(point[field], 4),
            "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
        }
    return {"n": len(joined_records), "metrics": metrics}


def evaluate_polarity_records(joined_records: list[dict[str, Any]], *, bootstrap_resamples: int = 1000) -> dict[str, Any]:
    labels = ["positive", "neutral", "negative"]
    point = _multiclass_metric(joined_records, labels=labels, field="polarity")
    lo, hi = bootstrap_ci(
        joined_records,
        metric_fn=lambda records: _multiclass_metric(records, labels=labels, field="polarity"),
        resamples=bootstrap_resamples,
    )
    return {
        "n": len(joined_records),
        "metrics": {
            "polarity_macro_f1": {
                "value": round(point, 4),
                "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            }
        },
    }


def evaluate_emotion_records(joined_records: list[dict[str, Any]], *, bootstrap_resamples: int = 1000) -> dict[str, Any]:
    labels = sorted(EMOTION_LABELS)
    point = _multiclass_metric(joined_records, labels=labels, field="emotion")
    lo, hi = bootstrap_ci(
        joined_records,
        metric_fn=lambda records: _multiclass_metric(records, labels=labels, field="emotion"),
        resamples=bootstrap_resamples,
    )
    return {
        "n": len(joined_records),
        "metrics": {
            "emotion_macro_f1": {
                "value": round(point, 4),
                "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            }
        },
    }


def evaluate_confusion_records(joined_records: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [pragmatic_polarity_class(record["gold"]) for record in joined_records]
    pred = [pragmatic_polarity_class(record["predictions"]) for record in joined_records]
    matrix = confusion_matrix(PRAGMATIC_POLARITY_CLASSES, gold, pred)
    return {
        "labels": PRAGMATIC_POLARITY_CLASSES,
        "counts": matrix,
        "row_normalised": row_normalise(matrix),
    }


def evaluate_calibration_records(
    joined_records: list[dict[str, Any]],
    *,
    target: str = "pragmatic_polarity",
    bins: int = 10,
) -> dict[str, Any]:
    confidences: list[float] = []
    correct: list[int] = []
    missing_confidence = 0
    for record in joined_records:
        confidence = confidence_for_class(record, target=target)
        if confidence is None:
            missing_confidence += 1
            continue
        if target == "pragmatic_polarity":
            gold = pragmatic_polarity_class(record["gold"])
            pred = pragmatic_polarity_class(record["predictions"])
        else:
            gold = record["gold"].get(target)
            pred = record["predictions"].get(target)
        confidences.append(confidence)
        correct.append(int(gold == pred))
    report = expected_calibration_error(confidences, correct, bins=bins)
    report["n"] = len(confidences)
    report["missing_confidence"] = missing_confidence
    report["target"] = target
    return report


def aggregate_seed_reports(seed_reports: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in metric_names:
        values = [_metric_value(report, metric) for report in seed_reports]
        values = [value for value in values if value is not None]
        if not values:
            continue
        ci = _normal_ci(values)
        out[metric] = {
            "mean": round(mean(values), 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "seed_values": [round(value, 4) for value in values],
            "n_seeds": len(values),
        }
    return out


def _metric_value(report: dict[str, Any], metric_name: str) -> float | None:
    metrics = report.get("metrics") or {}
    value = metrics.get(metric_name)
    if isinstance(value, dict):
        raw = value.get("value")
        if raw is not None:
            return float(raw)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normal_ci(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        value = values[0]
        return value, value
    spread = 1.96 * stdev(values) / math.sqrt(len(values))
    center = mean(values)
    return center - spread, center + spread


def _pragmatic_metric_fn(field: str) -> Callable[[list[dict[str, Any]]], float]:
    def metric(records: list[dict[str, Any]]) -> float:
        if not records:
            return 0.0
        golds = [record["gold"] for record in records]
        preds = [record["predictions"] for record in records]
        return pragmatic_f1(golds, preds)[field]

    return metric


def _multiclass_metric(records: list[dict[str, Any]], *, labels: list[str], field: str) -> float:
    y_true = [record["gold"].get(field) for record in records if record["gold"].get(field) in labels]
    y_pred = [
        record["predictions"].get(field)
        for record in records
        if record["gold"].get(field) in labels
    ]
    y_pred = [str(value) if value in labels else "__invalid__" for value in y_pred]
    return multiclass_macro_f1([str(value) for value in y_true], y_pred, labels) * 100
