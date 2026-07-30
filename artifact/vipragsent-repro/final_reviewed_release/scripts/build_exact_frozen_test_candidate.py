"""Materialize a frozen exact-OOF candidate without reading any test labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from final_best_tuned_exact_oof import INCUMBENT_THRESHOLDS, PRAGMATIC_LABELS, load_probs, mean_sources
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.utils.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=ROOT / "answer/final_best_tuned")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "answer/final_best_tuned_candidates")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/vipragsent_test.jsonl")
    parser.add_argument("--evaluate-frozen-test", action="store_true", help="Single post-freeze canonical test evaluation.")
    args = parser.parse_args()
    config = json.loads((args.candidate_dir / "configs/frozen_exact_target_candidate.json").read_text())
    # The test file is used only as the canonical ordered ID manifest.  No label
    # field is accessed by this script.
    ids = [str(row["id"]) for row in read_jsonl(args.test)]
    final_test = args.final_dir / "candidates/test"
    frozen = args.candidate_dir / "frozen_test_components"
    viso_paths = [frozen / "visobert_20260901.jsonl", final_test / "visobert_20260902.jsonl", final_test / "visobert_20260903.jsonl"]
    pho_paths = [final_test / "phobert_20260901.jsonl", frozen / "phobert_20260902.jsonl", frozen / "phobert_20260903.jsonl"]
    viso = mean_sources([load_probs(path, ids) for path in viso_paths])
    pho = mean_sources([load_probs(path, ids) for path in pho_paths])
    implicit = load_probs(final_test / "visobert_20260902.jsonl", ids)
    sarcasm = load_probs(final_test / "visobert_20260903.jsonl", ids)
    mocking = mean_sources([load_probs(final_test / f"mocking_targeted_{seed}.jsonl", ids) for seed in (20260833, 20260834, 20260835)])
    probabilities = {record_id: {label: 0.0 for label in PRAGMATIC_LABELS} for record_id in ids}
    for record_id in ids:
        probabilities[record_id]["implicit_sentiment"] = implicit[record_id]["implicit_sentiment"]
        probabilities[record_id]["sarcasm"] = sarcasm[record_id]["sarcasm"]
        probabilities[record_id]["mocking"] = mocking[record_id]["mocking"]
    for label, weights in config["source"].items():
        for record_id in ids:
            probabilities[record_id][label] = weights["visobert_uniform3_weight"] * viso[record_id][label] + weights["phobert_uniform3_weight"] * pho[record_id][label]
    thresholds = config["thresholds"]
    output_dir = args.candidate_dir / "frozen_exact_target_cross"; output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "final_test_predictions.jsonl"
    with output.open("w") as handle:
        for record_id in ids:
            row = {"id": record_id, "system": "ViPragSent exact_target_cross_backbone_frozen", "split": "test", "probabilities": probabilities[record_id], "predictions": {label: int(probabilities[record_id][label] >= thresholds[label]) for label in PRAGMATIC_LABELS}}
            handle.write(json.dumps(row) + "\n")
    manifest = {"status": "frozen_before_test_evaluation", "prediction_path": str(output), "ids": len(ids), "thresholds": thresholds, "sources": {"visobert_uniform3": [str(path) for path in viso_paths], "phobert_uniform3": [str(path) for path in pho_paths], "protected": {"implicit_sentiment": str(final_test / "visobert_20260902.jsonl"), "sarcasm": str(final_test / "visobert_20260903.jsonl"), "mocking": [str(final_test / f"mocking_targeted_{seed}.jsonl") for seed in (20260833, 20260834, 20260835)]}}, "test_labels_read": False}
    (output_dir / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.evaluate_frozen_test:
        print(json.dumps({"prediction_path": str(output), "ids": len(ids), "test_labels_read": False}, indent=2))
        return 0
    # This block is intentionally gated behind --evaluate-frozen-test.  The
    # candidate configuration and predictions above already exist before any
    # canonical label is accessed.
    prediction_rows = {str(row["id"]): row for row in read_jsonl(output)}
    gold_rows = {str(row["id"]): row for row in read_jsonl(args.test)}
    if set(prediction_rows) != set(gold_rows):
        raise ValueError("frozen prediction IDs do not match canonical test IDs")
    per_label = {}
    for label in PRAGMATIC_LABELS:
        gold = np.asarray([int(gold_rows[record_id]["labels"][label]) for record_id in ids])
        pred = np.asarray([int(prediction_rows[record_id]["predictions"][label]) for record_id in ids])
        per_label[label] = {
            "binary_macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0) * 100),
            "positive_precision": float(precision_score(gold, pred, zero_division=0)),
            "positive_recall": float(recall_score(gold, pred, zero_division=0)),
            "tp": int(((gold == 1) & (pred == 1)).sum()), "tn": int(((gold == 0) & (pred == 0)).sum()),
            "fp": int(((gold == 0) & (pred == 1)).sum()), "fn": int(((gold == 1) & (pred == 0)).sum()),
        }
    incumbent_path = args.final_dir / "predictions/final_test_predictions.jsonl"
    incumbent_rows = {str(row["id"]): row for row in read_jsonl(incumbent_path)}
    incumbent = {}
    for label in PRAGMATIC_LABELS:
        gold = np.asarray([int(gold_rows[record_id]["labels"][label]) for record_id in ids])
        pred = np.asarray([int(incumbent_rows[record_id]["predictions"][label]) for record_id in ids])
        incumbent[label] = float(f1_score(gold, pred, average="macro", zero_division=0) * 100)
    score = {label: value["binary_macro_f1"] for label, value in per_label.items()}
    macro = float(np.mean(list(score.values())))
    incumbent_macro = float(np.mean(list(incumbent.values())))
    targets = ("irony", "idiom_figurative", "code_switching")
    protected = ("implicit_sentiment", "sarcasm", "mocking")
    promotion = {
        "macro_strictly_higher": macro > incumbent_macro,
        "at_least_one_target_improved": any(score[label] > incumbent[label] for label in targets),
        "no_target_drop_over_0_02": all(score[label] >= incumbent[label] - .02 for label in targets),
        "no_protected_drop_over_0_05": all(score[label] >= incumbent[label] - .05 for label in protected),
    }
    promotion["eligible"] = all(promotion.values())
    evaluation = {"status": "single_post_freeze_test_evaluation", "test_labels_used_for_selection": False, "per_label": per_label, "macro_pragmatic_f1": macro, "incumbent_per_label": incumbent, "incumbent_macro_pragmatic_f1": incumbent_macro, "delta_vs_incumbent": {label: score[label] - incumbent[label] for label in PRAGMATIC_LABELS}, "promotion": promotion}
    (output_dir / "single_test_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    print(json.dumps({"macro_pragmatic_f1": macro, "incumbent_macro_pragmatic_f1": incumbent_macro, "per_label": score, "delta_vs_incumbent": evaluation["delta_vs_incumbent"], "promotion": promotion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
