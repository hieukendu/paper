#!/usr/bin/env python3
"""Development-selected per-label convex ensemble of frozen ViPragSent seeds."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_code_switch_rules import metric_grid
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl, write_jsonl

STEPS = 10
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100


def load(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def inputs(path: Path) -> tuple[list[str], list[dict[str, dict]]]:
    files = sorted(path.glob("reproduction_*.jsonl"))
    if len(files) < 2:
        raise SystemExit("at least two seed prediction files are required")
    rows = [load(file) for file in files]
    reference = set(rows[0])
    if any(set(seed) != reference for seed in rows[1:]):
        raise SystemExit("seed prediction IDs differ")
    return [file.stem.removeprefix("reproduction_") for file in files], rows


def weights(count: int) -> list[list[float]]:
    if count != 3:
        raise SystemExit("the current frozen protocol expects exactly three archived seeds")
    return [[first / STEPS, second / STEPS, (STEPS - first - second) / STEPS] for first in range(STEPS + 1) for second in range(STEPS - first + 1)]


def fit(args: argparse.Namespace) -> None:
    gold = {str(row["id"]): row["labels"] for row in read_jsonl(args.gold)}
    seeds, predictions = inputs(args.predictions)
    if set(gold) != set(predictions[0]):
        raise SystemExit("development gold and seed prediction IDs differ")
    ids = sorted(gold); combinations = weights(len(predictions)); selected = {}
    for label in PRAGMATIC_LABELS:
        truth = np.asarray([int(gold[record_id][label]) for record_id in ids])
        values = np.asarray([[float(seed[record_id]["probabilities"][label]) for seed in predictions] for record_id in ids])
        best = (-1.0, None, None)
        for combination in combinations:
            outcome = metric_grid(truth, values @ np.asarray(combination))
            position = int(outcome.argmax())
            candidate = (float(outcome[position]), combination, float(THRESHOLDS[position]))
            if candidate[0] > best[0] + 1e-12:
                best = candidate
        selected[label] = {"weights": best[1], "threshold": best[2], "development_binary_macro_f1": round(best[0], 4)}
    payload = {
        "method": "development_selected_per_label_convex_ensemble_of_frozen_vipragsent_seeds",
        "selection_split": "development", "gold_path": str(args.gold), "seeds": seeds,
        "weight_grid": {"step": 1 / STEPS, "count": len(combinations)}, "labels": selected,
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: rule["development_binary_macro_f1"] for label, rule in selected.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds, predictions = inputs(args.predictions)
    if seeds != config["seeds"]:
        raise SystemExit("seed IDs do not match the development-selected configuration")
    output = []
    for record_id in sorted(predictions[0]):
        row = json.loads(json.dumps(predictions[0][record_id]))
        for label in PRAGMATIC_LABELS:
            rule = config["labels"][label]
            value = sum(weight * float(seed[record_id]["probabilities"][label]) for weight, seed in zip(rule["weights"], predictions))
            row["probabilities"][f"{label}_seed_weighted"] = value
            row["predictions"][label] = int(value >= rule["threshold"])
        row["system"] = "vipragsent_frozen_seed_weighted_ensemble"
        output.append(row)
    write_jsonl(args.output, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--gold", type=Path, required=True); fit_parser.add_argument("--predictions", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True); fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--config", type=Path, required=True); apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
