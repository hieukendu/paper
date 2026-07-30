"""Five-fold development-only selector for final_best_tuned continuation runs.

The script intentionally accepts prediction files rather than model objects.  This
keeps every training run independent of selection and makes the OOF threshold,
ensemble, and calibration audit reproducible from saved probabilities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl


def stable_multilabel_folds(rows: list[dict], folds: int = 5) -> list[int]:
    """Deterministic greedy iterative stratification over all label marginals.

    Rows with rare label signatures are allocated first.  At every step the fold
    with the greatest deficit for the row's positive labels is selected, with a
    size deficit and stable SHA-256 ID tie-breaker.  This is a small dependency-
    free implementation suitable for the fixed six-label development set.
    """
    y = np.asarray([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in rows])
    signature = [tuple(value) for value in y]
    freq = Counter(signature)
    order = sorted(
        range(len(rows)),
        key=lambda i: (freq[signature[i]], int(y[i].sum()), hashlib.sha256(str(rows[i]["id"]).encode()).hexdigest()),
    )
    target = y.sum(axis=0) / folds
    assigned = [-1] * len(rows)
    fold_counts = np.zeros((folds, len(PRAGMATIC_LABELS)), dtype=float)
    fold_sizes = np.zeros(folds, dtype=float)
    target_size = len(rows) / folds
    for index in order:
        positive = y[index]
        scores = []
        for fold in range(folds):
            label_deficit = ((target - fold_counts[fold]).clip(min=0) * positive).sum()
            size_deficit = max(target_size - fold_sizes[fold], 0) / target_size
            scores.append((label_deficit + 0.05 * size_deficit, -fold_sizes[fold], -fold))
        chosen = max(range(folds), key=lambda fold: scores[fold])
        assigned[index] = chosen
        fold_counts[chosen] += positive
        fold_sizes[chosen] += 1
    return assigned


def load_probabilities(path: Path, expected_ids: set[str]) -> dict[str, dict[str, float]]:
    values = {}
    for row in read_jsonl(path):
        record_id = str(row["id"])
        if record_id in values:
            raise ValueError(f"duplicate ID {record_id} in {path}")
        values[record_id] = {
            label: float(row["probabilities"][label])
            for label in PRAGMATIC_LABELS
            if label in row["probabilities"]
        }
    if set(values) != expected_ids:
        raise ValueError(f"ID mismatch in {path}: expected={len(expected_ids)}, got={len(values)}")
    return values


def threshold_for(y: np.ndarray, p: np.ndarray, step: float) -> float:
    grid = np.arange(step, 1.0, step)
    score, threshold = max(
        (f1_score(y, p >= candidate, average="macro", zero_division=0), candidate)
        for candidate in grid
    )
    return float(threshold)


def evaluate(name: str, probabilities: dict[str, dict[str, float]], rows: list[dict], fold_ids: list[int], step: float):
    ids = [str(row["id"]) for row in rows]
    y = np.asarray([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in rows])
    p = np.asarray([[probabilities[record_id][label] for label in PRAGMATIC_LABELS] for record_id in ids])
    oof_pred = np.zeros_like(y)
    oof_thresholds = np.zeros_like(p, dtype=float)
    fold_metrics = []
    for fold in sorted(set(fold_ids)):
        calibration = np.asarray([index for index, value in enumerate(fold_ids) if value != fold])
        holdout = np.asarray([index for index, value in enumerate(fold_ids) if value == fold])
        thresholds = np.asarray([threshold_for(y[calibration, j], p[calibration, j], step) for j in range(len(PRAGMATIC_LABELS))])
        predicted = (p[holdout] >= thresholds).astype(int)
        oof_pred[holdout] = predicted
        oof_thresholds[holdout] = thresholds
        per_label = [f1_score(y[holdout, j], predicted[:, j], average="macro", zero_division=0) * 100 for j in range(len(PRAGMATIC_LABELS))]
        fold_metrics.append({"fold": int(fold), "macro_pragmatic_f1": float(np.mean(per_label)), **dict(zip(PRAGMATIC_LABELS, per_label))})
    all_label_f1 = np.asarray([f1_score(y[:, j], oof_pred[:, j], average="macro", zero_division=0) * 100 for j in range(len(PRAGMATIC_LABELS))])
    targets = [PRAGMATIC_LABELS.index(label) for label in ("irony", "idiom_figurative", "code_switching")]
    target_fold = np.asarray([[fold[label] for label in ("irony", "idiom_figurative", "code_switching")] for fold in fold_metrics])
    confusion = {}
    for j, label in enumerate(PRAGMATIC_LABELS):
        gold, predicted = y[:, j], oof_pred[:, j]
        confusion[label] = {
            "tp": int(((gold == 1) & (predicted == 1)).sum()),
            "tn": int(((gold == 0) & (predicted == 0)).sum()),
            "fp": int(((gold == 0) & (predicted == 1)).sum()),
            "fn": int(((gold == 1) & (predicted == 0)).sum()),
            "positive_precision": float(precision_score(gold, predicted, zero_division=0)),
            "positive_recall": float(recall_score(gold, predicted, zero_division=0)),
            "binary_macro_f1": float(all_label_f1[j]),
        }
    full_thresholds = {label: threshold_for(y[:, j], p[:, j], step) for j, label in enumerate(PRAGMATIC_LABELS)}
    payload = {
        "candidate": name,
        "selection_score": float(target_fold.mean() - 0.25 * target_fold.mean(axis=1).std(ddof=0)),
        "mean_target_label_f1": float(target_fold.mean()),
        "std_target_label_f1": float(target_fold.mean(axis=1).std(ddof=0)),
        "overall_pragmatic_macro_f1": float(all_label_f1.mean()),
        "per_label_oof_f1": dict(zip(PRAGMATIC_LABELS, map(float, all_label_f1))),
        "fold_metrics": fold_metrics,
        "confusion": confusion,
        "full_dev_thresholds": full_thresholds,
    }
    predictions = [
        {"id": record_id, "candidate": name, "fold": int(fold_ids[i]), "probabilities": dict(zip(PRAGMATIC_LABELS, map(float, p[i]))), "thresholds": dict(zip(PRAGMATIC_LABELS, map(float, oof_thresholds[i]))), "predictions": dict(zip(PRAGMATIC_LABELS, map(int, oof_pred[i])))}
        for i, record_id in enumerate(ids)
    ]
    return payload, predictions


def blend(a: dict, b: dict, alpha: float, rank: bool) -> dict:
    ids = sorted(a)
    out = {record_id: {} for record_id in ids}
    for label in PRAGMATIC_LABELS:
        av = np.asarray([a[record_id][label] for record_id in ids]); bv = np.asarray([b[record_id][label] for record_id in ids])
        if rank:
            av = av.argsort().argsort() / max(len(av) - 1, 1); bv = bv.argsort().argsort() / max(len(bv) - 1, 1)
        for i, record_id in enumerate(ids): out[record_id][label] = float(alpha * av[i] + (1 - alpha) * bv[i])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--cross-backbone", action="append", default=[], metavar="NAME=VISO,PHO")
    parser.add_argument("--overlay", action="append", default=[], metavar="NAME=BASE,EXPERT,LABEL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()
    rows = list(read_jsonl(args.dev)); ids = {str(row["id"]) for row in rows}
    folds = stable_multilabel_folds(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "development_folds.json").write_text(json.dumps({"folds": 5, "method": "deterministic greedy multilabel stratification", "assignments": dict(zip([str(row["id"]) for row in rows], folds))}, indent=2) + "\n")
    candidates = {}
    for spec in args.candidate:
        name, path = spec.split("=", 1); candidates[name] = load_probabilities(Path(path), ids)
    for spec in args.cross_backbone:
        name, source = spec.split("=", 1); first, second = source.split(",", 1)
        for alpha in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            candidates[f"{name}_alpha_{alpha:.1f}"] = blend(candidates[first], candidates[second], alpha, rank=False)
            candidates[f"{name}_rank_alpha_{alpha:.1f}"] = blend(candidates[first], candidates[second], alpha, rank=True)
    for spec in args.overlay:
        name, source = spec.split("=", 1); base, expert, label = source.split(",", 2)
        if label not in PRAGMATIC_LABELS:
            raise ValueError(f"unknown overlay label: {label}")
        candidates[name] = {
            record_id: {**candidates[base][record_id], label: candidates[expert][record_id][label]}
            for record_id in ids
        }
    summaries = []
    for name, values in candidates.items():
        # Specialist overlays intentionally provide only one label and are not
        # valid standalone pragmatic candidates.
        if any(set(values[record_id]) != set(PRAGMATIC_LABELS) for record_id in ids):
            continue
        result, predictions = evaluate(name, values, rows, folds, args.threshold_step)
        summaries.append(result)
        with (args.output_dir / f"{name}.oof.jsonl").open("w") as handle:
            for row in predictions: handle.write(json.dumps(row) + "\n")
        (args.output_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    summaries.sort(key=lambda x: x["selection_score"], reverse=True)
    fields = ["candidate", "selection_score", "mean_target_label_f1", "std_target_label_f1", "overall_pragmatic_macro_f1", *PRAGMATIC_LABELS]
    with (args.output_dir / "candidate_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in summaries:
            writer.writerow({**{key: row[key] for key in fields[:5]}, **row["per_label_oof_f1"]})
    (args.output_dir / "candidate_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"candidates": len(summaries), "best": summaries[0] if summaries else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
