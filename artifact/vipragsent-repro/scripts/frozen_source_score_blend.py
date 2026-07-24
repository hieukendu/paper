#!/usr/bin/env python3
"""Source-conditioned pairwise blends of frozen retrieval scores on dev only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_score_blend import ALPHAS, THRESHOLDS, parse_candidate, metric_grid, load, values
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl, write_jsonl


def source(row: dict) -> str:
    return str((row.get("source") or {}).get("dataset") or "unknown")


def choose(truth: np.ndarray, scores: dict[str, np.ndarray], names: list[str]) -> dict:
    best = (-1.0, None, None, None, None)
    for first_position, first in enumerate(names):
        for second in names[first_position:]:
            for alpha in ALPHAS:
                blended = alpha * scores[first] + (1 - alpha) * scores[second]
                outcome = metric_grid(truth, blended)
                position = int(outcome.argmax())
                candidate = (float(outcome[position]), first, second, float(alpha), float(THRESHOLDS[position]))
                if candidate[0] > best[0] + 1e-12 or (abs(candidate[0] - best[0]) <= 1e-12 and first == second and best[1] != best[2]):
                    best = candidate
    return {"first": best[1], "second": best[2], "first_weight": best[3], "threshold": best[4], "source_development_binary_macro_f1": round(best[0], 4)}


def fit(args: argparse.Namespace) -> None:
    gold = {str(row["id"]): row for row in read_jsonl(args.gold)}
    ids = sorted(gold); inputs = dict(args.candidate)
    candidates = {name: (load(path), suffix) for name, (path, suffix) in inputs.items()}
    if any(set(rows) != set(gold) for rows, _ in candidates.values()):
        raise SystemExit("candidate prediction IDs must exactly match development gold IDs")
    names = list(candidates); groups = sorted({source(row) for row in gold.values()})
    selected, combined = {}, {}
    for label in PRAGMATIC_LABELS:
        selected[label] = {}
        predictions = np.zeros(len(ids), dtype=np.int64)
        for group in groups:
            positions = [index for index, record_id in enumerate(ids) if source(gold[record_id]) == group]
            group_ids = [ids[index] for index in positions]
            truth = np.asarray([int(gold[record_id]["labels"][label]) for record_id in group_ids])
            score_by_name = {name: values(rows, suffix, label, group_ids) for name, (rows, suffix) in candidates.items()}
            rule = choose(truth, score_by_name, names)
            selected[label][group] = rule
            blended = rule["first_weight"] * score_by_name[rule["first"]] + (1 - rule["first_weight"]) * score_by_name[rule["second"]]
            predictions[positions] = (blended >= rule["threshold"]).astype(int)
        truth_all = [int(gold[record_id]["labels"][label]) for record_id in ids]
        # Equivalent to the shared evaluator's binary macro-F1, retained as a score only.
        from vipragsent.evaluation.metrics import binary_macro_f1
        combined[label] = round(binary_macro_f1(truth_all, predictions.tolist()) * 100, 4)
    payload = {
        "method": "development_only_source_conditioned_pairwise_convex_blend_of_frozen_scores",
        "selection_split": "development", "gold_path": str(args.gold),
        "candidates": {name: {"path": str(path), "probability_suffix": suffix} for name, (path, suffix) in inputs.items()},
        "labels": selected, "development_binary_macro_f1": combined,
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = {str(row["id"]): row for row in read_jsonl(args.data)}
    inputs = dict(args.candidate); candidates = {name: (load(path), suffix) for name, (path, suffix) in inputs.items()}
    expected = {choice for label_rules in config["labels"].values() for rule in label_rules.values() for choice in (rule["first"], rule["second"])}
    if expected - set(candidates):
        raise SystemExit(f"missing selected candidate(s): {sorted(expected - set(candidates))}")
    reference = candidates[next(iter(expected))][0]
    if set(reference) != set(data) or any(set(rows) != set(reference) for rows, _ in candidates.values()):
        raise SystemExit("candidate prediction IDs differ from requested data")
    output = []
    for record_id in sorted(data):
        row = json.loads(json.dumps(reference[record_id]))
        group = source(data[record_id])
        for label in PRAGMATIC_LABELS:
            rules = config["labels"][label]
            if group not in rules:
                raise SystemExit(f"no development rule for source {group!r}")
            rule = rules[group]
            left_rows, left_suffix = candidates[rule["first"]]
            right_rows, right_suffix = candidates[rule["second"]]
            score = rule["first_weight"] * values(left_rows, left_suffix, label, [record_id])[0] + (1 - rule["first_weight"]) * values(right_rows, right_suffix, label, [record_id])[0]
            row["probabilities"][f"{label}_source_score_blend"] = float(score)
            row["predictions"][label] = int(score >= rule["threshold"])
        row["system"] = "vipragsent_frozen_source_score_blend"
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
    apply_parser.add_argument("--config", type=Path, required=True); apply_parser.add_argument("--data", type=Path, required=True)
    apply_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True); apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
