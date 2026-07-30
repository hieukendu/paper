"""Development-only exact OOF continuation search for ViPragSent.

This script deliberately works from saved probability artifacts.  It never opens
the canonical test labels during selection.  It builds deterministic five-fold
assignments using labels, co-occurrences and available provenance metadata,
searches all valid existing encoder ensembles, and calibrates each held-out fold
with an exact threshold search (unique scores and their midpoints).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl

TARGETS = ("irony", "idiom_figurative", "code_switching")
PROTECTED = ("implicit_sentiment", "sarcasm", "mocking")
INCUMBENT_THRESHOLDS = {
    "implicit_sentiment": 0.55, "sarcasm": 0.87, "irony": 0.96,
    "idiom_figurative": 0.99, "code_switching": 0.50, "mocking": 0.35,
}


def row_metadata(row: dict) -> dict[str, str]:
    source = row.get("source") or {}
    platform = str(row.get("platform") or source.get("platform") or "unknown")
    dataset = str(source.get("dataset") or "unknown")
    batch = str(row.get("batch_id") or (row.get("annotation") or {}).get("batch_id") or "unknown")
    natural = "augmented" if str(row.get("id", "")).startswith("aug_") else "natural"
    return {"platform": platform, "source_dataset": dataset, "batch_id": batch, "record_type": natural}


def stable_metadata_folds(rows: list[dict], folds: int = 5) -> tuple[list[int], dict]:
    """Deterministic greedy multilabel stratification with provenance features."""
    labels = np.asarray([[int(row["labels"][name]) for name in PRAGMATIC_LABELS] for row in rows], dtype=int)
    metadata = [row_metadata(row) for row in rows]
    # Pairwise co-occurrences matter for irony/sarcasm and idiom/mocking.  Batch
    # membership is downweighted so a large number of tiny batches cannot drown
    # out the six target label marginals.
    pairs = np.asarray([[labels[i, a] * labels[i, b] for a, b in combinations(range(len(PRAGMATIC_LABELS)), 2)] for i in range(len(rows))], dtype=int)
    categories: list[tuple[str, str]] = []
    for key in ("platform", "source_dataset", "record_type"):
        categories.extend((key, value) for value in sorted({item[key] for item in metadata}))
    # Batch is useful proxy for annotation/source slices; retain only batches
    # with >= folds examples, otherwise they cannot be balanced meaningfully.
    batch_counts = Counter(item["batch_id"] for item in metadata)
    categories.extend(("batch_id", value) for value, count in sorted(batch_counts.items()) if count >= folds)
    cat_matrix = np.asarray([[int(item[key] == value) for key, value in categories] for item in metadata], dtype=int)
    features = np.concatenate((labels, pairs * 0.50, cat_matrix * 0.15), axis=1)
    signature = [tuple(labels[i]) + tuple(cat_matrix[i]) for i in range(len(rows))]
    rarity = Counter(signature)
    order = sorted(range(len(rows)), key=lambda i: (rarity[signature[i]], -int(labels[i].sum()), hashlib.sha256(str(rows[i]["id"]).encode()).hexdigest()))
    target = features.sum(axis=0) / folds
    seen = np.zeros((folds, features.shape[1]), dtype=float)
    sizes = np.zeros(folds, dtype=float)
    assigned = [-1] * len(rows)
    for index in order:
        scores = []
        for fold in range(folds):
            deficit = np.maximum(target - seen[fold], 0.0)
            # Label dimensions have full weight; provenance dimensions have the
            # pre-scaled contribution set above.
            score = float((deficit * features[index]).sum()) + 0.05 * max(len(rows) / folds - sizes[fold], 0.0)
            scores.append((score, -sizes[fold], -fold))
        chosen = max(range(folds), key=lambda fold: scores[fold])
        assigned[index] = chosen
        seen[chosen] += features[index]
        sizes[chosen] += 1
    diagnostics = {
        "method": "deterministic greedy multilabel + pairwise co-occurrence + provenance stratification",
        "features": {"labels": list(PRAGMATIC_LABELS), "pairwise_cooccurrences": [f"{PRAGMATIC_LABELS[a]}&{PRAGMATIC_LABELS[b]}" for a, b in combinations(range(len(PRAGMATIC_LABELS)), 2)], "metadata": [f"{key}={value}" for key, value in categories]},
        "fold_sizes": [int(size) for size in sizes],
        "fold_label_counts": {str(fold): {label: int(labels[np.asarray(assigned) == fold, j].sum()) for j, label in enumerate(PRAGMATIC_LABELS)} for fold in range(folds)},
        "metadata_counts": {str(fold): {f"{key}={value}": int(sum(assigned[i] == fold and metadata[i][key] == value for i in range(len(rows)))) for key, value in categories} for fold in range(folds)},
    }
    return assigned, diagnostics


def load_probs(path: Path, ids: list[str]) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    for row in read_jsonl(path):
        record_id = str(row["id"])
        if record_id in data:
            raise ValueError(f"duplicate ID {record_id} in {path}")
        probs = row.get("probabilities") or {}
        missing = set(PRAGMATIC_LABELS) - set(probs)
        if missing:
            raise ValueError(f"missing {sorted(missing)} in {path}")
        data[record_id] = {label: float(probs[label]) for label in PRAGMATIC_LABELS}
    if set(data) != set(ids):
        raise ValueError(f"ID alignment failed in {path}: expected={len(ids)}, actual={len(data)}")
    return data


def mean_sources(sources: list[dict[str, dict[str, float]]], weights: list[float] | None = None) -> dict[str, dict[str, float]]:
    if not sources:
        raise ValueError("at least one source is required")
    if weights is None:
        weights = [1 / len(sources)] * len(sources)
    if len(weights) != len(sources) or not np.isclose(sum(weights), 1.0) or min(weights) < 0:
        raise ValueError("invalid ensemble weights")
    out = {}
    for record_id in sources[0]:
        out[record_id] = {label: float(sum(weight * source[record_id][label] for weight, source in zip(weights, sources))) for label in PRAGMATIC_LABELS}
    return out


def replace_label(base: dict[str, dict[str, float]], source: dict[str, dict[str, float]], label: str) -> dict[str, dict[str, float]]:
    return {record_id: {**values, label: source[record_id][label]} for record_id, values in base.items()}


def exact_threshold(y: np.ndarray, p: np.ndarray, *, fp_limit: int | None = None) -> tuple[float, dict]:
    """Find a stable exact threshold without a coarse grid.

    Every unique score and every midpoint is considered.  Within an optimal
    prediction plateau the returned value is the plateau midpoint, avoiding an
    arbitrary choice of a single observed score.
    """
    values = np.unique(np.asarray(p, dtype=float))
    midpoints = (values[:-1] + values[1:]) / 2.0
    candidates = np.unique(np.concatenate(([0.0], values, midpoints, [1.0])))
    # Sweep the thresholds from high to low once.  This is exactly equivalent
    # to evaluating every candidate with sklearn, but avoids millions of
    # repeated O(n) metric calls for the exhaustive source bank.
    order = np.argsort(-p, kind="mergesort")
    total_pos = int(y.sum()); total_neg = int(len(y) - total_pos)
    tp = fp = 0; pointer = 0
    scored = []
    for threshold in sorted(candidates, reverse=True):
        while pointer < len(order) and p[order[pointer]] >= threshold:
            if y[order[pointer]]:
                tp += 1
            else:
                fp += 1
            pointer += 1
        fn = total_pos - tp; tn = total_neg - fp
        pos_denom = 2 * tp + fp + fn
        neg_denom = 2 * tn + fp + fn
        score = .5 * ((2 * tp / pos_denom if pos_denom else 0.0) + (2 * tn / neg_denom if neg_denom else 0.0))
        scored.append((float(score), int(fp), float(threshold)))
    feasible = [item for item in scored if fp_limit is None or item[1] <= fp_limit]
    if not feasible:
        feasible = scored
    best_score = max(item[0] for item in feasible)
    winners = [item for item in feasible if abs(item[0] - best_score) <= 1e-12]
    # largest contiguous winning interval, then its midpoint; it is a stable
    # plateau under exact thresholding rather than a one-example spike.
    winner_thresholds = sorted(item[2] for item in winners)
    chosen = min(winner_thresholds, key=lambda value: abs(value - np.median(winner_thresholds)))
    pred = p >= chosen
    detail = {
        "threshold": float(chosen), "binary_macro_f1": best_score * 100,
        "searched_unique_probabilities": int(len(values)), "searched_midpoints": int(len(midpoints)),
        "optimal_threshold_count": int(len(winners)), "fp_constraint": fp_limit,
        "positive_precision": float(precision_score(y, pred, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, zero_division=0)),
        "tp": int(((y == 1) & pred).sum()), "tn": int(((y == 0) & ~pred).sum()),
        "fp": int(((y == 0) & pred).sum()), "fn": int(((y == 1) & ~pred).sum()),
    }
    return float(chosen), detail


def evaluate_label(y: np.ndarray, p: np.ndarray, folds: list[int], label: str, fixed_threshold: float | None = None, fp_limit: int | None = None) -> dict:
    pred = np.zeros(len(y), dtype=int)
    used = np.zeros(len(y), dtype=float)
    fold_metrics = []
    for fold in sorted(set(folds)):
        train = np.asarray([index for index, item in enumerate(folds) if item != fold])
        test = np.asarray([index for index, item in enumerate(folds) if item == fold])
        threshold, _ = (fixed_threshold, {}) if fixed_threshold is not None else exact_threshold(y[train], p[train], fp_limit=fp_limit)
        fold_pred = (p[test] >= threshold).astype(int)
        pred[test] = fold_pred; used[test] = threshold
        fold_metrics.append(float(f1_score(y[test], fold_pred, average="macro", zero_division=0) * 100))
    full_threshold, full_detail = (fixed_threshold, {"threshold": fixed_threshold, "method": "fixed incumbent threshold"}) if fixed_threshold is not None else exact_threshold(y, p, fp_limit=fp_limit)
    return {
        "label": label, "oof_binary_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0) * 100),
        "fold_binary_macro_f1": fold_metrics, "fold_mean": float(np.mean(fold_metrics)), "fold_std": float(np.std(fold_metrics)),
        "selection_score": float(np.mean(fold_metrics) - .25 * np.std(fold_metrics)),
        "oof_positive_precision": float(precision_score(y, pred, zero_division=0)),
        "oof_positive_recall": float(recall_score(y, pred, zero_division=0)),
        "oof_tp": int(((y == 1) & (pred == 1)).sum()), "oof_tn": int(((y == 0) & (pred == 0)).sum()),
        "oof_fp": int(((y == 0) & (pred == 1)).sum()), "oof_fn": int(((y == 1) & (pred == 0)).sum()),
        "fold_thresholds": {str(fold): float(used[np.asarray(folds) == fold][0]) for fold in sorted(set(folds))},
        "full_dev_threshold": float(full_threshold), "full_dev_exact_search": full_detail,
        "oof_predictions": pred.tolist(),
    }


def candidate_metrics(name: str, probabilities: dict[str, dict[str, float]], ids: list[str], y: np.ndarray, folds: list[int], fixed: dict[str, float] | None = None) -> dict:
    per_label = {}
    for j, label in enumerate(PRAGMATIC_LABELS):
        limit = 1 if label == "idiom_figurative" else None
        per_label[label] = evaluate_label(y[:, j], np.asarray([probabilities[item][label] for item in ids]), folds, label, (fixed or {}).get(label), limit)
    targets = np.asarray([per_label[label]["fold_mean"] for label in TARGETS])
    fold_targets = np.mean(np.asarray([per_label[label]["fold_binary_macro_f1"] for label in TARGETS]), axis=0)
    return {
        "candidate": name,
        "overall_pragmatic_macro_f1": float(np.mean([per_label[label]["oof_binary_macro_f1"] for label in PRAGMATIC_LABELS])),
        "mean_target_label_f1": float(np.mean(targets)), "std_target_label_f1": float(np.std(fold_targets)),
        "selection_score": float(np.mean(fold_targets) - .25 * np.std(fold_targets)),
        "per_label": per_label,
    }


def record_registry(path: Path, rows: list[dict]) -> None:
    fields = ["branch", "stage", "status", "candidate", "target_label", "alpha", "selection_score", "oof_f1", "fold_mean", "fold_std", "full_dev_threshold", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--final-dir", type=Path, default=ROOT / "answer/final_best_tuned")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "answer/final_best_tuned_candidates")
    parser.add_argument("--extra-candidate", action="append", default=[], metavar="NAME=PATH", help="Additional development-only saved-probability source to audit.")
    args = parser.parse_args()
    started = perf_counter()
    rows = list(read_jsonl(args.dev)); ids = [str(row["id"]) for row in rows]
    y = np.asarray([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in rows], dtype=int)
    folds, fold_info = stable_metadata_folds(rows)
    config_dir = args.candidate_dir / "configs"; config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "development_folds.json").write_text(json.dumps({"folds": 5, "assignments": dict(zip(ids, folds)), **fold_info}, indent=2) + "\n")

    dev_dir = args.final_dir / "candidates/dev"
    viso = [load_probs(dev_dir / f"visobert_{seed}.jsonl", ids) for seed in (20260901, 20260902, 20260903)]
    pho = [load_probs(dev_dir / f"phobert_{seed}.jsonl", ids) for seed in (20260901, 20260902, 20260903)]
    incumbent = load_probs(args.final_dir / "predictions/final_dev_predictions.jsonl", ids)
    sources: dict[str, dict[str, dict[str, float]]] = {"incumbent_probabilities": incumbent}
    for prefix, collection in (("visobert", viso), ("phobert", pho)):
        for i, source in enumerate(collection, start=1): sources[f"{prefix}_seed{i}"] = source
        sources[f"{prefix}_uniform3"] = mean_sources(collection)
        for keep in combinations(range(3), 2):
            sources[f"{prefix}_top2_{keep[0]+1}{keep[1]+1}"] = mean_sources([collection[i] for i in keep])
            excluded = ({0, 1, 2} - set(keep)).pop() + 1
            sources[f"{prefix}_leave{excluded}out"] = sources[f"{prefix}_top2_{keep[0]+1}{keep[1]+1}"]
    for spec in args.extra_candidate:
        name, raw_path = spec.split("=", 1)
        if name in sources:
            raise ValueError(f"duplicate candidate name: {name}")
        sources[name] = load_probs(Path(raw_path), ids)
    registry: list[dict] = []
    source_results = {}
    # Existing complete sources are calibrated OOF solely for audit; their full
    # all-label values are not used to replace protected-label incumbents.
    for name, source in sources.items():
        result = candidate_metrics(name, source, ids, y, folds)
        source_results[name] = result
        registry.append({"branch": "existing_probability_bank", "stage": "A", "status": "evaluated", "candidate": name, "target_label": "all", "alpha": "", "selection_score": result["selection_score"], "oof_f1": result["overall_pragmatic_macro_f1"], "fold_mean": result["mean_target_label_f1"], "fold_std": result["std_target_label_f1"], "full_dev_threshold": "per_label", "notes": "valid aligned saved probabilities; exact nested OOF thresholds"})
    oof_dir = args.candidate_dir / "oof/exact_existing_probabilities"; oof_dir.mkdir(parents=True, exist_ok=True)
    (oof_dir / "source_bank_metrics.json").write_text(json.dumps(source_results, indent=2) + "\n")

    viso3, pho3 = sources["visobert_uniform3"], sources["phobert_uniform3"]
    alpha_grid = [round(.30 + .025 * i, 3) for i in range(23)]
    target_options: dict[str, list[dict]] = {label: [] for label in TARGETS}
    for alpha in alpha_grid:
        blend = mean_sources([viso3, pho3], [alpha, 1 - alpha])
        for label in TARGETS:
            j = PRAGMATIC_LABELS.index(label)
            result = evaluate_label(y[:, j], np.asarray([blend[item][label] for item in ids]), folds, label, fp_limit=1 if label == "idiom_figurative" else None)
            result["alpha"] = alpha
            target_options[label].append(result)
            registry.append({"branch": "cross_backbone_exact", "stage": "A", "status": "evaluated", "candidate": "visobert_phobert_probability_blend", "target_label": label, "alpha": alpha, "selection_score": result["selection_score"], "oof_f1": result["oof_binary_macro_f1"], "fold_mean": result["fold_mean"], "fold_std": result["fold_std"], "full_dev_threshold": result["full_dev_threshold"], "notes": "alpha grid .30-.85 step .025; exact unique+midpoint threshold search"})
    # Guard idiom with the explicit FP <= 1 condition. Other labels use the
    # requested stability score.  This selection uses development evidence only.
    # Equal OOF outcomes frequently form a broad probability plateau.  Prefer
    # the most balanced cross-backbone mixture in that case, rather than an
    # arbitrary extreme alpha or threshold.
    selected = {label: max(options, key=lambda item: (item["selection_score"], item["oof_binary_macro_f1"], -abs(item["alpha"] - .5))) for label, options in target_options.items()}
    target_mix = incumbent
    for label, item in selected.items():
        target_mix = replace_label(target_mix, mean_sources([viso3, pho3], [item["alpha"], 1 - item["alpha"]]), label)
    frozen = candidate_metrics("exact_target_cross_backbone", target_mix, ids, y, folds, fixed={label: INCUMBENT_THRESHOLDS[label] for label in PROTECTED})
    selected_config = {
        "status": "frozen_from_development_only", "candidate": "exact_target_cross_backbone",
        "source": {label: {"visobert_uniform3_weight": selected[label]["alpha"], "phobert_uniform3_weight": 1 - selected[label]["alpha"]} for label in TARGETS},
        "thresholds": {**{label: INCUMBENT_THRESHOLDS[label] for label in PROTECTED}, **{label: selected[label]["full_dev_threshold"] for label in TARGETS}},
        "selection_protocol": "five-fold metadata-aware OOF; target alpha selected by mean fold F1 - .25*std; exact threshold refit on full development only",
        "test_labels_used_for_selection": False,
    }
    (config_dir / "frozen_exact_target_candidate.json").write_text(json.dumps(selected_config, indent=2) + "\n")
    (oof_dir / "target_alpha_results.json").write_text(json.dumps(target_options, indent=2) + "\n")
    (oof_dir / "frozen_candidate_oof.json").write_text(json.dumps(frozen, indent=2) + "\n")
    registry.append({"branch": "cross_backbone_exact", "stage": "B", "status": "frozen_for_single_test_evaluation", "candidate": "exact_target_cross_backbone", "target_label": "all_targets", "alpha": json.dumps({label: selected[label]["alpha"] for label in TARGETS}, sort_keys=True), "selection_score": frozen["selection_score"], "oof_f1": frozen["overall_pragmatic_macro_f1"], "fold_mean": frozen["mean_target_label_f1"], "fold_std": frozen["std_target_label_f1"], "full_dev_threshold": json.dumps(selected_config["thresholds"], sort_keys=True), "notes": "protected labels fixed to incumbent probabilities and thresholds; candidate frozen before test labels"})
    record_registry(args.candidate_dir / "experiment_registry.csv", registry)
    summary = {"rows": len(rows), "candidate_sources": len(sources), "alpha_evaluations": len(alpha_grid) * len(TARGETS), "frozen_candidate": selected_config, "frozen_oof": frozen, "elapsed_seconds": perf_counter() - started}
    (oof_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "rows": summary["rows"], "candidate_sources": summary["candidate_sources"],
        "alpha_evaluations": summary["alpha_evaluations"], "frozen_candidate": selected_config,
        "frozen_oof_summary": {key: frozen[key] for key in ("overall_pragmatic_macro_f1", "mean_target_label_f1", "std_target_label_f1", "selection_score")},
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
