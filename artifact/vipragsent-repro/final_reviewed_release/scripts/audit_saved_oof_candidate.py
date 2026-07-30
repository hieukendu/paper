"""Audit one already-trained development prediction source against frozen OOF folds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from final_best_tuned_exact_oof import PRAGMATIC_LABELS, candidate_metrics, load_probs
from vipragsent.utils.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "answer/final_best_tuned_candidates")
    args = parser.parse_args()
    rows = list(read_jsonl(args.dev)); ids = [str(row["id"]) for row in rows]
    folds_doc = json.loads((args.candidate_dir / "configs/development_folds.json").read_text())
    folds = [int(folds_doc["assignments"][record_id]) for record_id in ids]
    y = np.asarray([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in rows], dtype=int)
    result = candidate_metrics(args.name, load_probs(args.predictions, ids), ids, y, folds)
    output = args.candidate_dir / "encoder_screen" / f"{args.name}.exact_oof.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    registry = args.candidate_dir / "experiment_registry.csv"
    fields = ["branch", "stage", "status", "candidate", "target_label", "alpha", "selection_score", "oof_f1", "fold_mean", "fold_std", "full_dev_threshold", "notes"]
    row = {"branch": "compact_encoder_screen", "stage": "A", "status": "rejected" if result["selection_score"] < 91.71518923051637 else "survives_screen", "candidate": args.name, "target_label": "all", "alpha": "", "selection_score": result["selection_score"], "oof_f1": result["overall_pragmatic_macro_f1"], "fold_mean": result["mean_target_label_f1"], "fold_std": result["std_target_label_f1"], "full_dev_threshold": "exact_per_label", "notes": "one-seed compact architecture; development-only; test not evaluated"}
    existing = list(csv.DictReader(registry.open())) if registry.exists() else []
    existing.append({key: str(row[key]) for key in fields})
    with registry.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(existing)
    json_registry = registry.with_suffix(".json")
    json_rows = json.loads(json_registry.read_text()) if json_registry.exists() else []
    json_rows.append(row); json_registry.write_text(json.dumps(json_rows, indent=2) + "\n")
    print(json.dumps({"report": str(output), "candidate": args.name, "overall_oof": result["overall_pragmatic_macro_f1"], "selection_score": result["selection_score"], "target_oof": {label: result["per_label"][label]["oof_binary_macro_f1"] for label in ("irony", "idiom_figurative", "code_switching")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
