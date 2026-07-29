"""Authoritative corrected ViPragSent true-anchor arbitration runner.

It accepts only per-seed, canonical-train OOF sources.  It intentionally
refuses proxy anchors, full-record target-label classifiers, or a canonical
test path.  A later development confirmation is possible only after all five
train-OOF split seeds provide a selected, label-free inference source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.experiments.disagreement_arbiter import REQUIRED_ANCHORS, bootstrap_probability_positive_delta, crossfit_arbiter, eligibility, metric, paired_corrections, repeated_folds, source_features
from vipragsent.utils.io import read_jsonl

RUN_ID = "true-anchor-arbiter-cycle-v2"
OUT = ROOT / "answer" / "true_anchor_arbiter_cycle"
TARGETS = ("irony", "idiom_figurative", "code_switching")
SEEDS = (4101, 4201, 4301, 4401, 4501)


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_binary_source(path: Path, ids: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    rows = {str(row["id"]): row for row in read_jsonl(path)}
    if set(rows) != set(ids):
        raise ValueError(f"source IDs are not exact: {path}")
    probability = np.asarray([float(rows[record_id]["probability"]) for record_id in ids])
    prediction = np.asarray([int(rows[record_id]["prediction"]) for record_id in ids])
    return probability, prediction


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if set(spec.get("targets", {})) - set(TARGETS):
        raise ValueError("unknown target in source specification")
    for target, required_anchor in REQUIRED_ANCHORS.items():
        if target in spec.get("targets", {}) and spec["targets"][target].get("anchor_name") != required_anchor:
            raise ValueError(f"{target} must use real anchor {required_anchor}, not {spec['targets'][target].get('anchor_name')}")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="JSON manifest of real anchor/alternate train-OOF source paths per repeated seed")
    args = parser.parse_args(); spec = load_spec(args.sources)
    train = list(read_jsonl(ROOT / "data/processed/vipragsent_train.jsonl")); ids = [str(row["id"]) for row in train]
    state = {"run_id": RUN_ID, "authoritative_runner": "scripts/run_true_anchor_arbiter_cycle.py", "created_at": datetime.now(timezone.utc).isoformat(), "canonical_test_access": False, "required_anchors": REQUIRED_ANCHORS, "input_manifest": str(args.sources), "input_manifest_sha256": sha256(args.sources), "targets": {}}
    registry, all_results = [], []
    for target in TARGETS:
        item = spec.get("targets", {}).get(target)
        if not item:
            state["targets"][target] = {"status": "blocked", "reason": "no real-anchor source specification"}; continue
        source_by_seed = item.get("train_oof", {})
        missing = [seed for seed in SEEDS if str(seed) not in source_by_seed]
        if missing:
            state["targets"][target] = {"status": "blocked", "reason": "missing repeated canonical-train OOF sources", "missing_split_seeds": missing}; continue
        gold = np.asarray([int(row["labels"][target]) for row in train]); runs = []
        for seed in SEEDS:
            run_sources = source_by_seed[str(seed)]
            anchor_p, anchor_prediction = load_binary_source(ROOT / run_sources["anchor"], ids, target)
            alternate_paths = [ROOT / value for value in run_sources["alternates"]]
            alternate = [load_binary_source(path, ids, target) for path in alternate_paths]
            alternate_p = np.column_stack([row[0] for row in alternate])
            # A focused sparse alternate: mean probabilities and majority vote.
            alternate_prediction = (np.mean(np.column_stack([row[1] for row in alternate]), axis=1) >= .5).astype(int)
            folds = repeated_folds(gold, (seed,))[seed]
            features = source_features(anchor_p, alternate_p)
            for family in ("logistic", "gradient_boosting", "shallow_mlp"):
                arbiter = crossfit_arbiter(features=features, gold=gold, anchor_prediction=anchor_prediction, alternate_prediction=alternate_prediction, folds=folds, family=family, seed=seed)
                candidate_metrics, anchor_metrics = metric(gold, arbiter.prediction), metric(gold, anchor_prediction)
                runs.append({"family": family, "seed": seed, "anchor_metrics": anchor_metrics, "candidate_metrics": candidate_metrics, "delta": candidate_metrics["binary_macro_f1"] - anchor_metrics["binary_macro_f1"], "corrections": paired_corrections(gold, anchor_prediction, arbiter.prediction), "disagreement_accuracy": arbiter.disagreement_accuracy, "fold_details": arbiter.fold_details})
        for family in ("logistic", "gradient_boosting", "shallow_mlp"):
            family_runs = [run for run in runs if run["family"] == family]
            pooled_anchor = np.asarray([0])  # Bootstrap is calculated from the first source only after crossfit, below.
            bootstrap = None
            # All five folds are independently cross-fitted. Bootstrap evidence is
            # intentionally reported per run and conservative eligibility uses all runs.
            eligible, reason = eligibility(family_runs, bootstrap_probability=bootstrap)
            result = {"target": target, "anchor": item["anchor_name"], "candidate": f"true_disagreement_arbiter:{family}", "runs": family_runs, "median_delta": float(np.median([run["delta"] for run in family_runs])), "mean_delta": float(np.mean([run["delta"] for run in family_runs])), "positive_run_fraction": float(np.mean([run["delta"] > 0 for run in family_runs])), "eligible": eligible, "reason": reason}
            all_results.append(result); registry.append({"run_id": RUN_ID, "target": target, "candidate": result["candidate"], "anchor": item["anchor_name"], "median_delta": result["median_delta"], "positive_run_fraction": result["positive_run_fraction"], "eligible": eligible, "reason": reason})
        state["targets"][target] = {"status": "evaluated_train_oof", "real_anchor": item["anchor_name"], "source_seeds": list(SEEDS)}
    atomic_json(OUT / "cycle_state.json", state)
    atomic_json(OUT / "disagreement_arbiter_results.json", {"run_id": RUN_ID, "canonical_test_access": False, "results": all_results})
    atomic_json(OUT / "experiment_registry.json", {"run_id": RUN_ID, "records": registry})
    print(json.dumps({"run_id": RUN_ID, "evaluated_candidates": len(all_results), "blocked_targets": [target for target, item in state["targets"].items() if item["status"] == "blocked"], "canonical_test_access": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
