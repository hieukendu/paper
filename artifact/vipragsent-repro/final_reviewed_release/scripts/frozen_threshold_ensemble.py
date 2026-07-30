#!/usr/bin/env python3
"""Frozen-weight probability ensembling and development-only threshold fitting.

This utility deliberately never instantiates, changes, or saves neural model
parameters.  ``fit`` reads only a development gold file; ``apply`` reads only
saved thresholds and model prediction JSONL files.  It is therefore suitable
for the frozen-weight ViPragSent protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from vipragsent.data.schema import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, canonicalize_labels
from vipragsent.evaluation.metrics import binary_macro_f1, pragmatic_f1
from vipragsent.utils.io import read_jsonl, write_jsonl


GRID = [round(value / 100, 2) for value in range(1, 100)]


def _records_by_id(paths: list[Path]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in paths:
        for row in read_jsonl(path):
            record_id = str(row["id"])
            grouped.setdefault(record_id, []).append(row)
    expected = len(paths)
    incomplete = [record_id for record_id, rows in grouped.items() if len(rows) != expected]
    if incomplete:
        raise ValueError(f"prediction ensemble has {len(incomplete)} incomplete record(s)")
    return grouped


def _majority(values: list[str], valid: set[str]) -> str:
    choices = sorted(valid)
    counts = Counter(value for value in values if value in valid)
    if not counts:
        return choices[0]
    return max(choices, key=lambda value: (counts[value], -choices.index(value)))


def _ensemble_rows(paths: list[Path], *, thresholds: dict[str, float] | None = None) -> list[dict]:
    rows: list[dict] = []
    for record_id, members in sorted(_records_by_id(paths).items()):
        probabilities = {
            label: mean(float(member["probabilities"][label]) for member in members)
            for label in PRAGMATIC_LABELS
        }
        selected = thresholds or {label: 0.5 for label in PRAGMATIC_LABELS}
        predictions = {label: int(probabilities[label] >= selected[label]) for label in PRAGMATIC_LABELS}
        predictions["polarity"] = _majority(
            [member["predictions"].get("polarity") for member in members], POLARITY_LABELS
        )
        predictions["emotion"] = _majority(
            [member["predictions"].get("emotion") for member in members], EMOTION_LABELS
        )
        rows.append(
            {
                "id": record_id,
                "system": "vipragsent_frozen_threshold_ensemble",
                "seed": "ensemble_20260520_20260521_20260522",
                "predictions": predictions,
                "probabilities": probabilities,
                "members": [str(member.get("seed")) for member in members],
            }
        )
    return rows


def _gold_by_id(path: Path) -> dict[str, dict]:
    return {str(row["id"]): canonicalize_labels(row["labels"]) for row in read_jsonl(path)}


def _fit_thresholds(gold_path: Path, prediction_paths: list[Path]) -> tuple[dict[str, float], dict[str, float]]:
    gold = _gold_by_id(gold_path)
    rows = _ensemble_rows(prediction_paths)
    if set(gold) != {row["id"] for row in rows}:
        raise ValueError("development gold IDs and ensemble prediction IDs differ")
    thresholds: dict[str, float] = {}
    scores: dict[str, float] = {}
    for label in PRAGMATIC_LABELS:
        truth = [int(gold[row["id"]][label]) for row in rows]
        probabilities = [float(row["probabilities"][label]) for row in rows]
        ranked = []
        for threshold in GRID:
            score = binary_macro_f1(truth, [int(value >= threshold) for value in probabilities]) * 100
            ranked.append((score, -abs(threshold - 0.5), -threshold, threshold))
        best = max(ranked)
        scores[label] = round(best[0], 4)
        thresholds[label] = best[3]
    return thresholds, scores


def fit(args: argparse.Namespace) -> None:
    prediction_paths = sorted(args.predictions.glob("*.jsonl"))
    if not prediction_paths:
        raise SystemExit(f"no prediction files in {args.predictions}")
    thresholds, scores = _fit_thresholds(args.gold, prediction_paths)
    payload = {
        "method": "mean_probability_ensemble_with_label_thresholds",
        "selection_split": "development",
        "gold_path": str(args.gold),
        "prediction_files": [str(path) for path in prediction_paths],
        "threshold_grid": {"start": 0.01, "stop": 0.99, "step": 0.01},
        "thresholds": thresholds,
        "development_binary_macro_f1": scores,
        "frozen_weight_compliance": {
            "neural_weight_updates": False,
            "selection_uses_test_labels": False,
            "optimizer_or_backward_called": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prediction_paths = sorted(args.predictions.glob("*.jsonl"))
    if not prediction_paths:
        raise SystemExit(f"no prediction files in {args.predictions}")
    rows = _ensemble_rows(prediction_paths, thresholds=config["thresholds"])
    write_jsonl(args.output, rows)
    print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def score(args: argparse.Namespace) -> None:
    gold = _gold_by_id(args.gold)
    predictions = {str(row["id"]): row["predictions"] for row in read_jsonl(args.predictions)}
    if set(gold) != set(predictions):
        raise SystemExit("evaluation gold IDs and prediction IDs differ")
    report = pragmatic_f1([gold[record_id] for record_id in sorted(gold)], [predictions[record_id] for record_id in sorted(gold)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows({"metric": metric, "value": f"{value:.4f}"} for metric, value in report.items())
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    fit_parser = command.add_parser("fit", help="Select label thresholds on development gold only")
    fit_parser.add_argument("--gold", type=Path, required=True)
    fit_parser.add_argument("--predictions", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.set_defaults(func=fit)
    apply_parser = command.add_parser("apply", help="Apply saved thresholds without reading gold labels")
    apply_parser.add_argument("--config", type=Path, required=True)
    apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.set_defaults(func=apply)
    score_parser = command.add_parser("score", help="Score an already frozen configuration")
    score_parser.add_argument("--gold", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
