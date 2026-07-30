#!/usr/bin/env python3
"""Train-label dependency adjustment over frozen ViPragSent score outputs.

The only learned quantities are smoothed empirical co-occurrence rates from the
training labels.  They are combined with already-frozen prediction scores and
selected on development data.  No neural module or label classifier is fitted.
"""

from __future__ import annotations

import argparse
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

SMOOTHING = [0.0, 0.5, 1.0, 2.0, 4.0]
ALPHAS = np.arange(1, 11, dtype=np.float64) / 10
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100


def source(row: dict) -> str:
    return str((row.get("source") or {}).get("dataset") or "unknown")


def load(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def score(row: dict, label: str, suffix: str) -> float:
    key = label if suffix == "raw" else f"{label}{suffix}"
    return float(row["probabilities"][key])


def probabilities(rows: dict[str, dict], ids: list[str], suffix: str) -> np.ndarray:
    return np.asarray([[score(rows[record_id], label, suffix) for label in PRAGMATIC_LABELS] for record_id in ids], dtype=np.float64)


def table(train_rows: list[dict], *, smoothing: float, source_specific: bool) -> dict[str, np.ndarray]:
    groups: dict[str, list[dict]] = {}
    for row in train_rows:
        groups.setdefault(source(row) if source_specific else "global", []).append(row)
    result = {}
    for group, rows in groups.items():
        labels = np.asarray([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in rows], dtype=np.float64)
        # Matrix [target, parent, parent value] stores P(target=1 | parent=value).
        probabilities_for_group = np.zeros((len(PRAGMATIC_LABELS), len(PRAGMATIC_LABELS), 2), dtype=np.float64)
        prior = labels.mean(axis=0)
        for target_index in range(len(PRAGMATIC_LABELS)):
            for parent_index in range(len(PRAGMATIC_LABELS)):
                for parent_value in (0, 1):
                    mask = labels[:, parent_index] == parent_value
                    numerator = labels[mask, target_index].sum() + smoothing * prior[target_index]
                    denominator = mask.sum() + smoothing
                    probabilities_for_group[target_index, parent_index, parent_value] = numerator / denominator if denominator else prior[target_index]
        result[group] = probabilities_for_group
    return result


def dependency_values(probability_matrix: np.ndarray, groups: list[str], tables: dict[str, np.ndarray], target: int, parent: int, source_specific: bool) -> np.ndarray:
    values = np.zeros(len(probability_matrix), dtype=np.float64)
    for index, probabilities_for_row in enumerate(probability_matrix):
        mapping = tables[groups[index] if source_specific else "global"]
        values[index] = probabilities_for_row[parent] * mapping[target, parent, 1] + (1 - probabilities_for_row[parent]) * mapping[target, parent, 0]
    return values


def fit(args: argparse.Namespace) -> None:
    train_rows = list(read_jsonl(args.train))
    gold = {str(row["id"]): row for row in read_jsonl(args.dev_gold)}
    frozen = load(args.predictions)
    if set(gold) != set(frozen):
        raise SystemExit("development prediction IDs must exactly match gold IDs")
    ids = sorted(gold)
    groups = [source(gold[record_id]) for record_id in ids]
    base = probabilities(frozen, ids, args.probability_suffix)
    selected = {}
    for target_index, target in enumerate(PRAGMATIC_LABELS):
        truth = np.asarray([int(gold[record_id]["labels"][target]) for record_id in ids])
        best = (-1.0, None, None, None, None, None)
        for source_specific in (False, True):
            for smoothing in SMOOTHING:
                tables = table(train_rows, smoothing=smoothing, source_specific=source_specific)
                for parent_index, parent in enumerate(PRAGMATIC_LABELS):
                    if parent_index == target_index:
                        continue
                    dependency = dependency_values(base, groups, tables, target_index, parent_index, source_specific)
                    for alpha in ALPHAS:
                        outcome = metric_grid(truth, alpha * dependency + (1 - alpha) * base[:, target_index])
                        position = int(outcome.argmax())
                        candidate = (float(outcome[position]), parent, source_specific, smoothing, float(alpha), float(THRESHOLDS[position]))
                        if candidate[0] > best[0] + 1e-12:
                            best = candidate
        selected[target] = {
            "parent_label": best[1], "source_specific": best[2], "smoothing": best[3],
            "alpha_dependency": best[4], "threshold": best[5], "development_binary_macro_f1": round(best[0], 4),
        }
    payload = {
        "method": "development_selected_train_label_dependency_adjustment_of_frozen_scores",
        "selection_split": "development", "train_path": str(args.train), "dev_gold_path": str(args.dev_gold),
        "probability_suffix": args.probability_suffix, "labels": selected,
        "search": {"smoothing": SMOOTHING, "alpha_dependency": ALPHAS.tolist(), "source_specific": [False, True]},
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: rule["development_binary_macro_f1"] for label, rule in selected.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_rows = list(read_jsonl(args.train)); data = {str(row["id"]): row for row in read_jsonl(args.data)}
    frozen = load(args.predictions)
    if set(data) != set(frozen):
        raise SystemExit("prediction IDs must exactly match requested split")
    ids = sorted(data); groups = [source(data[record_id]) for record_id in ids]
    base = probabilities(frozen, ids, config["probability_suffix"])
    position = {record_id: index for index, record_id in enumerate(ids)}
    output = []
    cache: dict[tuple[bool, float], dict[str, np.ndarray]] = {}
    for target_index, target in enumerate(PRAGMATIC_LABELS):
        rule = config["labels"][target]
        key = (rule["source_specific"], rule["smoothing"])
        cache.setdefault(key, table(train_rows, smoothing=rule["smoothing"], source_specific=rule["source_specific"]))
        parent_index = PRAGMATIC_LABELS.index(rule["parent_label"])
        values = dependency_values(base, groups, cache[key], target_index, parent_index, rule["source_specific"])
        for record_id, row in frozen.items():
            index = position[record_id]
            value = rule["alpha_dependency"] * values[index] + (1 - rule["alpha_dependency"]) * base[index, target_index]
            row.setdefault("_dependency_values", {})[target] = float(value)
    for record_id in ids:
        row = json.loads(json.dumps(frozen[record_id]))
        for target in PRAGMATIC_LABELS:
            value = row["_dependency_values"][target]
            rule = config["labels"][target]
            row["probabilities"][f"{target}_dependency_adjusted"] = value
            row["predictions"][target] = int(value >= rule["threshold"])
        row.pop("_dependency_values")
        row["system"] = "vipragsent_frozen_label_dependency"
        output.append(row)
    write_jsonl(args.output, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--train", type=Path, required=True); fit_parser.add_argument("--dev-gold", type=Path, required=True)
    fit_parser.add_argument("--predictions", type=Path, required=True); fit_parser.add_argument("--probability-suffix", default="raw")
    fit_parser.add_argument("--output", type=Path, required=True); fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--train", type=Path, required=True); apply_parser.add_argument("--data", type=Path, required=True)
    apply_parser.add_argument("--predictions", type=Path, required=True); apply_parser.add_argument("--config", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
