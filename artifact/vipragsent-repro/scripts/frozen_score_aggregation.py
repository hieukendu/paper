#!/usr/bin/env python3
"""Development-selected nonlinear aggregation of frozen score candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_score_blend import THRESHOLDS, metric_grid, parse_candidate, load, values
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl, write_jsonl

METHODS = ["minimum", "maximum", "median", "lower_quartile", "upper_quartile", "geometric_mean"]


def aggregate(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "minimum": return matrix.min(axis=1)
    if method == "maximum": return matrix.max(axis=1)
    if method == "median": return np.median(matrix, axis=1)
    if method == "lower_quartile": return np.quantile(matrix, 0.25, axis=1)
    if method == "upper_quartile": return np.quantile(matrix, 0.75, axis=1)
    if method == "geometric_mean": return np.exp(np.log(np.clip(matrix, 1e-8, 1)).mean(axis=1))
    raise ValueError(f"unknown aggregation method: {method}")


def fit(args: argparse.Namespace) -> None:
    gold = {str(row["id"]): row["labels"] for row in read_jsonl(args.gold)}
    ids = sorted(gold); inputs = dict(args.candidate)
    candidates = {name: (load(path), suffix) for name, (path, suffix) in inputs.items()}
    if any(set(rows) != set(gold) for rows, _ in candidates.values()):
        raise SystemExit("candidate prediction IDs must exactly match development gold IDs")
    selected = {}
    for label in PRAGMATIC_LABELS:
        truth = np.asarray([int(gold[record_id][label]) for record_id in ids])
        matrix = np.column_stack([values(rows, suffix, label, ids) for rows, suffix in candidates.values()])
        best = (-1.0, None, None)
        for method in METHODS:
            outcome = metric_grid(truth, aggregate(matrix, method))
            position = int(outcome.argmax())
            candidate = (float(outcome[position]), method, float(THRESHOLDS[position]))
            if candidate[0] > best[0] + 1e-12:
                best = candidate
        selected[label] = {"method": best[1], "threshold": best[2], "development_binary_macro_f1": round(best[0], 4)}
    payload = {
        "method": "development_selected_nonlinear_aggregation_of_frozen_scores",
        "selection_split": "development", "gold_path": str(args.gold),
        "candidates": {name: {"path": str(path), "probability_suffix": suffix} for name, (path, suffix) in inputs.items()},
        "methods": METHODS, "labels": selected,
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: rule["development_binary_macro_f1"] for label, rule in selected.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = dict(args.candidate); candidates = {name: (load(path), suffix) for name, (path, suffix) in inputs.items()}
    reference = next(iter(candidates.values()))[0]
    if any(set(rows) != set(reference) for rows, _ in candidates.values()):
        raise SystemExit("candidate prediction IDs differ")
    output = []
    for record_id in sorted(reference):
        row = json.loads(json.dumps(reference[record_id]))
        for label in PRAGMATIC_LABELS:
            rule = config["labels"][label]
            matrix = np.asarray([[values(rows, suffix, label, [record_id])[0] for rows, suffix in candidates.values()]])
            score = float(aggregate(matrix, rule["method"])[0])
            row["probabilities"][f"{label}_score_aggregate"] = score
            row["predictions"][label] = int(score >= rule["threshold"])
        row["system"] = "vipragsent_frozen_score_aggregation"
        output.append(row)
    write_jsonl(args.output, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--gold", type=Path, required=True); fit_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    fit_parser.add_argument("--output", type=Path, required=True); fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--config", type=Path, required=True); apply_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
