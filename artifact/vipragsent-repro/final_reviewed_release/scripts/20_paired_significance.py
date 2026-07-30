from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.evaluation.metrics import pragmatic_f1
from vipragsent.experiments.evaluator import load_gold_records
from vipragsent.experiments.predictions import PredictionError, join_gold_predictions, load_prediction_records


def score(records: list[dict]) -> float:
    return pragmatic_f1([row["gold"] for row in records], [row["predictions"] for row in records])["macro_pragmatic_f1"]


def paired_test(reference: list[dict], challenger: list[dict], *, resamples: int, seed: int) -> dict:
    challenger_by_id = {row["id"]: row for row in challenger}
    paired = [(row, challenger_by_id[row["id"]]) for row in reference if row["id"] in challenger_by_id]
    if not paired:
        raise ValueError("No paired records")
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        indices = [rng.randrange(len(paired)) for _ in paired]
        left = [paired[i][0] for i in indices]
        right = [paired[i][1] for i in indices]
        differences.append(score(right) - score(left))
    differences.sort()
    observed = score([b for _, b in paired]) - score([a for a, _ in paired])
    p_lower = sum(value <= 0 for value in differences) / len(differences)
    p_upper = sum(value >= 0 for value in differences) / len(differences)
    return {
        "n": len(paired),
        "delta_macro_pragmatic_f1": observed,
        "paired_bootstrap_ci95": [differences[int(.025 * (len(differences) - 1))], differences[int(.975 * (len(differences) - 1))]],
        "two_sided_p": min(1.0, 2 * min(p_lower, p_upper)),
        "resamples": resamples,
    }


def main() -> int:
    gold = load_gold_records(ROOT / "data/processed/vipragsent_test.jsonl", task="pragmatic", require_adjudicated=True)
    root = ROOT / "results/predictions/main_pragmatic"
    reference_system = "vipragsent_full"
    reference_files = {path.stem: path for path in (root / reference_system).glob("*.jsonl")}
    comparisons = {}
    skipped = {}
    for system_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != reference_system):
        per_seed = {}
        for path in sorted(system_dir.glob("*.jsonl")):
            if path.stem not in reference_files:
                continue
            reference = join_gold_predictions(gold, load_prediction_records(reference_files[path.stem]))
            try:
                challenger = join_gold_predictions(gold, load_prediction_records(path))
            except PredictionError as exc:
                skipped.setdefault(system_dir.name, {})[path.stem] = str(exc)
                continue
            per_seed[path.stem] = paired_test(reference, challenger, resamples=1000, seed=int(path.stem))
        if per_seed:
            comparisons[system_dir.name] = per_seed
    report = {
        "status": "complete",
        "reference_system": reference_system,
        "test": "paired non-parametric bootstrap over test records",
        "multiple_comparison_note": "Raw p-values are reported; apply Holm correction when selecting a final hypothesis family.",
        "comparisons": comparisons,
        "skipped_incomplete_predictions": skipped,
    }
    output = ROOT / "results/significance.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "systems": len(comparisons)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
