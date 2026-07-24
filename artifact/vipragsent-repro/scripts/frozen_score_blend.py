#!/usr/bin/env python3
"""Select pairwise score blends over frozen retrieval outputs on development data.

The script is deliberately non-parametric: it combines outputs already
produced by frozen encoders, then selects a convex mixing weight and a
threshold using development labels only.  It never constructs a trainable
module or an optimizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl, write_jsonl

ALPHAS = np.arange(0, 11, dtype=np.float64) / 10
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100


def parse_candidate(value: str) -> tuple[str, tuple[Path, str]]:
    name, separator, raw = value.partition("=")
    path, suffix_separator, suffix = raw.rpartition(":")
    if not separator or not name or not suffix_separator or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH:PROBABILITY_SUFFIX")
    return name, (Path(path), suffix)


def load(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def values(rows: dict[str, dict], suffix: str, label: str, record_ids: list[str]) -> np.ndarray:
    key = label if suffix == "raw" else f"{label}{suffix}"
    try:
        return np.asarray([float(rows[record_id]["probabilities"][key]) for record_id in record_ids])
    except KeyError as error:
        raise SystemExit(f"missing score field {key!r} in candidate predictions") from error


def metric_grid(truth: np.ndarray, scores: np.ndarray) -> np.ndarray:
    prediction = scores[:, None] >= THRESHOLDS[None, :]
    true = truth[:, None].astype(bool)
    tp = np.sum(prediction & true, axis=0)
    fp = np.sum(prediction & ~true, axis=0)
    fn = np.sum(~prediction & true, axis=0)
    tn = np.sum(~prediction & ~true, axis=0)
    f1_positive = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    f1_negative = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    return (f1_positive + f1_negative) * 50


def fit(args: argparse.Namespace) -> None:
    gold = {str(row["id"]): row["labels"] for row in read_jsonl(args.gold)}
    record_ids = sorted(gold)
    inputs = dict(args.candidate)
    candidates = {name: (load(path), suffix) for name, (path, suffix) in inputs.items()}
    if any(set(rows) != set(gold) for rows, _ in candidates.values()):
        raise SystemExit("candidate prediction IDs must exactly match development gold IDs")
    names = list(candidates)
    selected, development_scores = {}, {}
    for label in PRAGMATIC_LABELS:
        truth = np.asarray([int(gold[record_id][label]) for record_id in record_ids])
        score_by_name = {name: values(rows, suffix, label, record_ids) for name, (rows, suffix) in candidates.items()}
        best = (-1.0, None, None, None, None)
        for first_position, first in enumerate(names):
            for second in names[first_position:]:
                for alpha in ALPHAS:
                    score = alpha * score_by_name[first] + (1 - alpha) * score_by_name[second]
                    outcome = metric_grid(truth, score)
                    threshold_index = int(outcome.argmax())
                    candidate = (float(outcome[threshold_index]), first, second, float(alpha), float(THRESHOLDS[threshold_index]))
                    # Preserve a deterministic preference for simpler single-source choices in ties.
                    if candidate[0] > best[0] + 1e-12 or (abs(candidate[0] - best[0]) <= 1e-12 and first == second and best[1] != best[2]):
                        best = candidate
        selected[label] = {
            "first": best[1], "second": best[2], "first_weight": best[3],
            "threshold": best[4], "development_binary_macro_f1": round(best[0], 4),
        }
        development_scores[label] = round(best[0], 4)
    payload = {
        "method": "development_only_pairwise_convex_blend_of_frozen_scores",
        "selection_split": "development", "gold_path": str(args.gold),
        "candidates": {name: {"path": str(path), "probability_suffix": suffix} for name, (path, suffix) in inputs.items()},
        "labels": selected, "development_binary_macro_f1": development_scores,
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(development_scores, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = dict(args.candidate)
    candidates = {name: (load(path), suffix) for name, (path, suffix) in inputs.items()}
    needed = {choice for rule in config["labels"].values() for choice in (rule["first"], rule["second"])}
    if needed - set(candidates):
        raise SystemExit(f"missing selected candidate(s): {sorted(needed - set(candidates))}")
    reference = candidates[next(iter(needed))][0]
    if any(set(rows) != set(reference) for rows, _ in candidates.values()):
        raise SystemExit("candidate prediction IDs differ")
    output = []
    for record_id in sorted(reference):
        row = json.loads(json.dumps(reference[record_id]))
        for label, rule in config["labels"].items():
            left_rows, left_suffix = candidates[rule["first"]]
            right_rows, right_suffix = candidates[rule["second"]]
            left = values(left_rows, left_suffix, label, [record_id])[0]
            right = values(right_rows, right_suffix, label, [record_id])[0]
            score = rule["first_weight"] * left + (1 - rule["first_weight"]) * right
            row["probabilities"][f"{label}_score_blend"] = float(score)
            row["predictions"][label] = int(score >= rule["threshold"])
        row["system"] = "vipragsent_frozen_score_blend"
        output.append(row)
    write_jsonl(args.output, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--gold", type=Path, required=True)
    fit_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--config", type=Path, required=True)
    apply_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
