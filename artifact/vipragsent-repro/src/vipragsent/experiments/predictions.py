from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from vipragsent.data.schema import (
    EMOTION_LABELS,
    POLARITY_LABELS,
    PRAGMATIC_LABELS,
    canonicalize_labels,
)
from vipragsent.experiments.constants import PRAGMATIC_POLARITY_CLASSES
from vipragsent.utils.io import read_jsonl, write_jsonl


class PredictionError(ValueError):
    """Raised when prediction files cannot be joined or evaluated."""


def normalize_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    record_id = record.get("id") or record.get("input_id") or record.get("record_id")
    if not record_id:
        raise PredictionError("prediction record is missing id/input_id")
    predictions = (
        record.get("predictions")
        or record.get("prediction")
        or record.get("predicted_labels")
        or record.get("labels_pred")
    )
    if not isinstance(predictions, dict):
        raise PredictionError(f"{record_id}: prediction record is missing predictions object")
    out = {
        "id": str(record_id),
        "predictions": canonicalize_labels(predictions),
        "probabilities": record.get("probabilities") or record.get("probs") or record.get("scores") or {},
        "logits": record.get("logits") or {},
        "system": record.get("system"),
        "seed": record.get("seed"),
        "raw": record,
    }
    if isinstance(record.get("labels"), dict):
        out["labels"] = canonicalize_labels(record["labels"])
    return out


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    return [normalize_prediction_record(record) for record in read_jsonl(path)]


def dump_prediction_records(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    return write_jsonl(path, records)


def validate_pragmatic_predictions(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for record in records:
        labels = record["predictions"]
        for label in PRAGMATIC_LABELS:
            if labels.get(label) not in (0, 1, False, True):
                errors.append(f"{record['id']}: predictions.{label} must be 0/1")
        if labels.get("polarity") not in POLARITY_LABELS:
            errors.append(f"{record['id']}: predictions.polarity must be one of {sorted(POLARITY_LABELS)}")
        if labels.get("emotion") not in EMOTION_LABELS and labels.get("emotion") is not None:
            errors.append(f"{record['id']}: predictions.emotion must be one of {sorted(EMOTION_LABELS)} or null")
    return errors


def join_gold_predictions(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in prediction_records}
    duplicate_count = len(prediction_records) - len(by_id)
    if duplicate_count:
        raise PredictionError(f"prediction file contains {duplicate_count} duplicate id(s)")
    missing = [record["id"] for record in gold_records if record["id"] not in by_id]
    if missing:
        sample = ", ".join(missing[:5])
        raise PredictionError(f"prediction file is missing {len(missing)} gold id(s), e.g. {sample}")
    joined = []
    for gold in gold_records:
        pred = by_id[gold["id"]]
        joined.append(
            {
                "id": gold["id"],
                "text": gold.get("text"),
                "gold": canonicalize_labels(gold.get("labels")),
                "predictions": pred["predictions"],
                "probabilities": pred.get("probabilities") or {},
                "logits": pred.get("logits") or {},
                "system": pred.get("system"),
                "seed": pred.get("seed"),
                "source": gold.get("source"),
                "split": gold.get("split"),
            }
        )
    return joined


def pragmatic_polarity_class(labels: dict[str, Any]) -> str:
    labels = canonicalize_labels(labels)
    if labels.get("mocking") in (1, True):
        return "mocking"
    if labels.get("sarcasm") in (1, True) and labels.get("polarity") == "negative":
        return "sarcastic-negative"
    if labels.get("irony") in (1, True) and labels.get("polarity") == "positive":
        return "ironic-positive"
    polarity = labels.get("polarity")
    if polarity in {"positive", "negative", "neutral"}:
        return str(polarity)
    return "neutral"


def confidence_for_class(record: dict[str, Any], *, target: str = "pragmatic_polarity") -> float | None:
    probabilities = record.get("probabilities") or {}
    predictions = record.get("predictions") or {}
    if target == "pragmatic_polarity":
        predicted = pragmatic_polarity_class(predictions)
        values = probabilities.get("pragmatic_polarity") or probabilities.get("pragmatic_polarity_class")
        if isinstance(values, dict) and predicted in values:
            return _as_probability(values[predicted])
        polarity_values = probabilities.get("polarity")
        if isinstance(polarity_values, dict):
            polarity = predictions.get("polarity")
            if polarity in polarity_values:
                return _as_probability(polarity_values[polarity])
    if target in PRAGMATIC_LABELS:
        value = probabilities.get(target)
        if isinstance(value, dict):
            pred = int(predictions.get(target) in (1, True))
            key = "1" if pred == 1 else "0"
            return _as_probability(value.get(key, value.get(pred)))
        if value is not None:
            positive_prob = _as_probability(value)
            if positive_prob is None:
                return None
            pred = int(predictions.get(target) in (1, True))
            return positive_prob if pred == 1 else 1.0 - positive_prob
    generic = record.get("raw", {}).get("confidence") if isinstance(record.get("raw"), dict) else None
    if generic is not None:
        return _as_probability(generic)
    return None


def _as_probability(value: Any) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if prob < 0.0 or prob > 1.0:
        return None
    return prob


def all_pragmatic_polarity_classes() -> list[str]:
    return list(PRAGMATIC_POLARITY_CLASSES)
