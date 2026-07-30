"""Assemble strict held-out-fold probabilities into a true canonical-train OOF source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.utils.io import read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    args = parser.parse_args()
    assignments = json.loads(args.folds.read_text(encoding="utf-8"))["assignments"]
    train = {str(row["id"]): row for row in read_jsonl(args.train)}
    if set(train) != set(assignments):
        raise ValueError("fold assignment must cover exactly the canonical train IDs")
    merged, provenance = {}, {}
    for fold in range(5):
        directory = args.fold_root / f"fold_{fold}"
        prediction_path, manifest_path = directory / "heldout_probabilities.jsonl", directory / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {record_id for record_id, value in assignments.items() if int(value) == fold}
        current = {str(row["id"]): float(row["probability"]) for row in read_jsonl(prediction_path)}
        if set(current) != expected or set(merged).intersection(current):
            raise ValueError(f"fold {fold} does not contain its exact held-out IDs")
        threshold = float(manifest["best_checkpoint_selection"]["threshold"])
        merged.update(current)
        provenance[str(fold)] = {"checkpoint": manifest["checkpoint"], "checkpoint_sha256": sha256(Path(manifest["checkpoint"])), "threshold": threshold, "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path), "training_records": manifest["train_records"]}
    if set(merged) != set(train):
        raise ValueError("OOF coverage is not exact once")
    output = []
    for record_id in sorted(train):
        fold = int(assignments[record_id]); probability = merged[record_id]; threshold = provenance[str(fold)]["threshold"]
        output.append({"id": record_id, "fold": fold, "label": args.label, "probability": probability, "prediction": int(probability >= threshold), "threshold": threshold, "source": args.fold_root.name})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    y = np.asarray([int(train[row["id"]]["labels"][args.label]) for row in output]); pred = np.asarray([row["prediction"] for row in output])
    summary = {"status": "ok", "label": args.label, "records": len(output), "coverage_exact_once": True, "folds": 5, "test_access": False, "binary_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0) * 100), "provenance": provenance, "output": str(args.output)}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
