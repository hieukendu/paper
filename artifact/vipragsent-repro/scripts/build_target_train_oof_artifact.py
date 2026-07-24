"""Assemble one auditable true-OOF target-expert artifact from fold outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    args = parser.parse_args()

    assignments = json.loads(args.folds.read_text())["assignments"]
    rows = {str(row["id"]): row for row in read_jsonl(args.train)}
    if set(rows) != set(assignments):
        raise ValueError("fold assignments do not cover precisely the train IDs")

    probabilities: dict[str, float] = {}
    thresholds: dict[int, float] = {}
    manifests = {}
    for fold in range(5):
        directory = args.fold_output_root / f"headonly_fold_{fold}"
        manifest = json.loads((directory / "run_manifest.json").read_text())
        if manifest.get("status") != "ok" or manifest.get("test_labels_read") or manifest.get("test_predictions_created"):
            raise ValueError(f"invalid fold manifest: {directory}")
        thresholds[fold] = float(manifest["best_checkpoint_selection"]["threshold"])
        manifests[str(fold)] = {"path": str(directory / "run_manifest.json"), "checkpoint": manifest["checkpoint"], "threshold": thresholds[fold]}
        fold_probs = {str(item["id"]): float(item["probability"]) for item in read_jsonl(directory / "dev_probabilities.jsonl")}
        expected = {record_id for record_id, value in assignments.items() if int(value) == fold}
        if set(fold_probs) != expected or set(probabilities).intersection(fold_probs):
            raise ValueError(f"fold {fold} is not an exact held-out prediction set")
        probabilities.update(fold_probs)
    if set(probabilities) != set(rows):
        raise ValueError("OOF probabilities do not cover precisely the train IDs")

    artifact = []
    for record_id, row in rows.items():
        fold = int(assignments[record_id])
        truth = int(row["labels"][args.label])
        probability = probabilities[record_id]
        predicted = int(probability >= thresholds[fold])
        error_type = ("tp" if truth else "fp") if predicted else ("fn" if truth else "tn")
        source = row.get("source") or {}
        artifact.append({
            "id": record_id, "fold": fold, "label": args.label, "gold": truth,
            "probability": probability, "threshold": thresholds[fold], "prediction": predicted,
            "error_type": error_type, "platform": row.get("platform") or source.get("platform") or "unknown",
            "source_dataset": source.get("dataset") or "unknown", "batch_id": row.get("batch_id") or "unknown",
        })
    artifact.sort(key=lambda item: item["id"])
    y = np.asarray([item["gold"] for item in artifact])
    p = np.asarray([item["probability"] for item in artifact])
    pred = np.asarray([item["prediction"] for item in artifact])
    counts = {name: sum(item["error_type"] == name for item in artifact) for name in ("tp", "tn", "fp", "fn")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.label}_train_oof.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for item in artifact:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok", "label": args.label, "records": len(artifact), "folds": 5,
        "coverage_exact_once": True, "test_labels_read": False, "test_predictions_created": False,
        "metrics": {"binary_macro_f1": float(f1_score(y, pred, average="macro")), "pr_auc": float(average_precision_score(y, p)), **counts},
        "fold_manifests": manifests, "output": str(output),
    }
    (args.output_dir / f"{args.label}_train_oof_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
