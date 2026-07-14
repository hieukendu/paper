from __future__ import annotations

from typing import Any, Iterable

from vipragsent.annotation.llm_labeler import heuristic_silver_labels
from vipragsent.data.schema import PRAGMATIC_LABELS


def heuristic_probability_payload(predictions: dict[str, Any]) -> dict[str, Any]:
    probabilities: dict[str, Any] = {}
    for label in PRAGMATIC_LABELS:
        pred = int(predictions.get(label) in (1, True))
        positive_prob = 0.78 if pred == 1 else 0.22
        probabilities[label] = positive_prob
    polarity = predictions.get("polarity") or "neutral"
    probabilities["polarity"] = {
        "positive": 0.12,
        "neutral": 0.12,
        "negative": 0.12,
    }
    probabilities["polarity"][polarity] = 0.76
    probabilities["pragmatic_polarity"] = {
        "positive": 0.10,
        "negative": 0.10,
        "neutral": 0.10,
        "ironic-positive": 0.10,
        "sarcastic-negative": 0.10,
        "mocking": 0.10,
    }
    if predictions.get("mocking") in (1, True):
        probabilities["pragmatic_polarity"]["mocking"] = 0.70
    elif predictions.get("sarcasm") in (1, True) and polarity == "negative":
        probabilities["pragmatic_polarity"]["sarcastic-negative"] = 0.70
    elif predictions.get("irony") in (1, True) and polarity == "positive":
        probabilities["pragmatic_polarity"]["ironic-positive"] = 0.70
    else:
        probabilities["pragmatic_polarity"][polarity] = 0.70
    return probabilities


def heuristic_prediction_records(
    gold_records: Iterable[dict[str, Any]],
    *,
    system: str = "heuristic_local",
    seed: int = 20260520,
) -> list[dict[str, Any]]:
    records = []
    for record in gold_records:
        predictions = heuristic_silver_labels(record.get("text", ""))
        records.append(
            {
                "id": record["id"],
                "system": system,
                "seed": seed,
                "predictions": predictions,
                "probabilities": heuristic_probability_payload(predictions),
                "provenance": {
                    "source": "vipragsent.annotation.llm_labeler.heuristic_silver_labels",
                    "note": "Smoke-test baseline only; not a paper result.",
                },
            }
        )
    return records
