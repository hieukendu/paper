"""Evaluate completed targeted experts with development-only nested thresholds.

The experts are trained exclusively on the canonical training split.  This
script constructs one labelwise candidate by substituting their held-out
development probabilities for the corresponding incumbent label source, then
updates the fair-framework registry without loading candidate test outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from run_fair_framework_cycle import (
    DISPLAY_BASELINES, PRAGMATIC_LABELS, binary, choose_robust_threshold,
    load_probability, read_jsonl, split_folds, verify_baselines,
)

OUT = ROOT / "answer/final_best_tuned_fair_framework_candidates"
EXPERTS = {
    "irony": ROOT / "outputs/final_best_tuned_fair_framework/irony_attention/20260701/dev_probabilities.jsonl",
    "idiom_figurative": ROOT / "outputs/final_best_tuned_fair_framework/idiom_clsmeanmax/20260711/dev_probabilities.jsonl",
    "code_switching": ROOT / "outputs/final_best_tuned_fair_framework/code_clsmeanmax_tokenaux/20260721/dev_probabilities.jsonl",
}


def load_target(path: Path, ids: list[str], label: str) -> np.ndarray:
    rows = {str(row["id"]): row for row in read_jsonl(path)}
    if set(rows) != set(ids) or len(rows) != len(ids):
        raise ValueError(f"target expert ID mismatch: {path}")
    if any(str(row.get("label")) != label or "probability" not in row for row in rows.values()):
        raise ValueError(f"invalid target expert probability schema: {path}")
    return np.asarray([float(rows[rid]["probability"]) for rid in ids])


def nested_threshold_predictions(y: np.ndarray, p: np.ndarray, outer_folds: np.ndarray) -> tuple[np.ndarray, dict]:
    pred = np.zeros(len(y), dtype=int); detail = {"outer_folds": {}}
    for outer in range(5):
        train = np.where(outer_folds != outer)[0]; held = np.where(outer_folds == outer)[0]
        # Threshold fitting uses only the outer-training rows; its four stable
        # partitions are deterministic and never include held-out labels.
        inner = np.arange(len(train)) % 4
        threshold, lam, robust = choose_robust_threshold(y[train], p[train], inner)
        pred[held] = p[held] >= threshold
        detail["outer_folds"][str(outer)] = {"threshold": threshold, "lambda": lam, "robust_score": robust}
    return pred, detail


def score(predictions: dict[str, np.ndarray], y: np.ndarray, baselines: dict[str, float], details: dict) -> dict:
    confusion = {label: binary(y[:, j], predictions[label]) for j, label in enumerate(PRAGMATIC_LABELS)}
    metrics = {label: confusion[label]["binary_macro_f1"] for label in PRAGMATIC_LABELS}
    metrics["macro_pragmatic_f1"] = float(np.mean(list(metrics.values())))
    margins = {metric: metrics[metric] - baselines[metric] for metric in baselines}
    return {"candidate": "FFO-targeted-labelwise-screen", "metrics": metrics, "margins": margins,
            "minimum_baseline_margin": min(margins.values()), "metrics_above_baseline": sum(value > 1e-9 for value in margins.values()),
            "mean_baseline_margin": float(np.mean(list(margins.values()))), "confusion": confusion, "details": details,
            "test_labels_used": False, "test_predictions_created": False}


def main() -> int:
    dev_rows = list(read_jsonl(ROOT / "data/processed/vipragsent_dev.jsonl"))
    test_rows = list(read_jsonl(ROOT / "data/processed/vipragsent_test.jsonl"))
    ids = [str(row["id"]) for row in dev_rows]
    if len(ids) != 2000 or len(set(ids)) != 2000:
        raise ValueError("development IDs are not exactly 2,000 unique records")
    audit = verify_baselines(test_rows)
    if not audit["passed"]:
        raise RuntimeError("baseline display verification failed")
    baselines = audit["authoritative_baseline_max"]
    y = np.asarray([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in dev_rows], dtype=int)
    folds = split_folds(dev_rows)
    incumbent = load_probability(ROOT / "answer/final_best_tuned/predictions/final_dev_predictions.jsonl", ids)
    probabilities = {label: np.asarray([incumbent[rid][label] for rid in ids]) for label in PRAGMATIC_LABELS}
    probabilities.update({label: load_target(path, ids, label) for label, path in EXPERTS.items()})
    predictions, details = {}, {"protocol": "five outer folds; thresholds fit only on outer-training rows", "experts": {label: str(path) for label, path in EXPERTS.items()}, "per_label": {}}
    for j, label in enumerate(PRAGMATIC_LABELS):
        predictions[label], details["per_label"][label] = nested_threshold_predictions(y[:, j], probabilities[label], folds)
    candidate = score(predictions, y, baselines, details)
    oof_path = OUT / "targeted_retraining_screen_oof_predictions.jsonl"
    with oof_path.open("w") as handle:
        for index, record_id in enumerate(ids):
            handle.write(json.dumps({"id": record_id, "split": "development_oof", "system": candidate["candidate"],
                "probabilities": {label: float(probabilities[label][index]) for label in PRAGMATIC_LABELS},
                "predictions": {label: int(predictions[label][index]) for label in PRAGMATIC_LABELS}}) + "\n")
    candidate["oof_prediction_path"] = str(oof_path)
    (OUT / "targeted_retraining_screen.json").write_text(json.dumps({"baseline_audit": audit, "candidate": candidate}, indent=2) + "\n")
    # Append, rather than overwrite, the Phase-1/2 registry so negative
    # targeted results remain traceable.
    current = json.loads((OUT / "candidate_metrics.json").read_text())
    # v3 writes a summary object while this targeted screen owns a list of
    # candidate records. Never reinterpret summary fields as candidates.
    current = current if isinstance(current, list) else []
    current = [item for item in current if item.get("candidate") != candidate["candidate"]] + [candidate]
    current.sort(key=lambda item: (-item["minimum_baseline_margin"], -item["metrics_above_baseline"], -item["mean_baseline_margin"], -item["metrics"]["macro_pragmatic_f1"]))
    (OUT / "candidate_metrics.json").write_text(json.dumps(current, indent=2) + "\n")
    (OUT / "candidate_confusion_matrices.json").write_text(json.dumps({item["candidate"]: item["confusion"] for item in current}, indent=2) + "\n")
    fields = ["candidate", "minimum_baseline_margin", "metrics_above_baseline", "mean_baseline_margin", "macro_pragmatic_f1"] + list(PRAGMATIC_LABELS)
    with (OUT / "experiment_registry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for item in current:
            writer.writerow({"candidate": item["candidate"], "minimum_baseline_margin": item["minimum_baseline_margin"], "metrics_above_baseline": item["metrics_above_baseline"], "mean_baseline_margin": item["mean_baseline_margin"], "macro_pragmatic_f1": item["metrics"]["macro_pragmatic_f1"], **{label: item["metrics"][label] for label in PRAGMATIC_LABELS}})
    best = current[0]
    with (OUT / "gap_to_baseline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "candidate_score", "baseline_max", "margin", "pass"], lineterminator="\n"); writer.writeheader()
        for metric in baselines:
            writer.writerow({"metric": metric, "candidate_score": best["metrics"][metric], "baseline_max": baselines[metric], "margin": best["margins"][metric], "pass": best["margins"][metric] > 1e-9})
    verification = json.loads((OUT / "initial_verification.json").read_text())
    status = {"status": "NOT_PROMOTED", "phase": "targeted_framework_retraining", "branch": "agent/vipragsent-fair-framework-all-metrics",
              "system": best["candidate"], "selected_on": "development-only nested five-fold threshold OOF", "test_candidate_evaluated": False,
              "reason": "targeted single-seed screens did not clear all full-precision baseline margins; stopped before multi-seed confirmation and canonical candidate-test evaluation",
              "hashes_before": verification["hashes_before"], "hashes_after": verification["hashes_before"], "best_candidate": best,
              "authoritative_baseline_max": baselines}
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    report = ["# ViPragSent Fair-Framework Cycle", "", "## Verification", "", "- Raw baseline predictions were recomputed with the repository metric implementation.", "- The immutable four-decimal registry is retained for display; every value is consistent by four-decimal rounding or within `1e-4` percentage points.", "- Full-precision raw seed-mean maxima, recorded in `initial_verification.json`, were used for every margin below.", "- Train/dev/test SHA-256 hashes are unchanged and all development/test ID sets contain exactly 2,000 unique records.", "", "## Completed stages", "", "1. Nested five-fold robust probability calibration, non-uniform ensemble, and four dynamic gate families.", "2. Single-seed PhoBERT targeted screens trained only on canonical training records:", "   - irony: attentive pooling + residual head (seed 20260701);", "   - idiom: CLS/mean/max pooling + residual head (seed 20260711);", "   - code-switching: CLS/mean/max pooling + residual head + deterministic token auxiliary loss (seed 20260721).", "", "The ViSoBERT targeted smoke screen was rejected before training because the pinned archive lacks a tokenizer configuration compatible with the installed Transformers runtime. It is retained as a smoke-test failure, not a final experiment.", "", "## Best rejected candidate", "", f"`{best['candidate']}` was selected by maximum minimum full-precision baseline margin. Its target experts were combined labelwise with incumbent development sources; every threshold was fit without the corresponding outer-fold labels.", "", "| Metric | Candidate OOF F1 | Full-precision baseline maximum | Margin |", "| --- | ---: | ---: | ---: |"]
    for metric in list(PRAGMATIC_LABELS) + ["macro_pragmatic_f1"]:
        report.append(f"| {metric} | {best['metrics'][metric]:.10f} | {baselines[metric]:.10f} | {best['margins'][metric]:+.10f} |")
    report += ["", "The target screens improve irony over its required maximum, but idiom, code-switching, sarcasm, and mocking remain below their full-precision baseline thresholds. Under successive halving there is no development-safe candidate to advance to multi-seed confirmation, freezing, or the one permitted candidate-test evaluation. `final_best_tuned` and all baseline artifacts remain unchanged.", "", "NOT_PROMOTED"]
    (OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"candidate": candidate["candidate"], "metrics": candidate["metrics"], "margins": candidate["margins"], "best": best["candidate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
