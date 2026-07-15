from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.experiments.evaluator import evaluate_calibration_records, evaluate_pragmatic_records, load_gold_records
from vipragsent.experiments.predictions import join_gold_predictions, load_prediction_records

SYSTEM_TO_ROW = {
    "vipragsent_full": "full",
    "vipragsent_no_emotion": "no_emotion_auxiliary",
    "vipragsent_no_polarity": "no_ordinary_sentiment_auxiliary",
    "vipragsent_no_rationale": "no_explanation_cot_auxiliary",
    "phobert_finetune": "no_multitask",
    "vipragsent_no_uncertainty": "no_task_uncertainty_weighting",
}

NOT_IMPLEMENTED_ROWS = {
    "explanation_augmented_inference": "No inference-time rationale-generation variant is implemented in this protocol.",
    "hard_label_distillation": "No teacher-logit or hard-label distillation training run is implemented in this protocol.",
}


def main() -> int:
    gold = load_gold_records(ROOT / "data/processed/vipragsent_test.jsonl", task="pragmatic", require_adjudicated=True)
    prediction_root = ROOT / "results/predictions/main_pragmatic"
    rows = {}
    for system, row_name in SYSTEM_TO_ROW.items():
        files = sorted((prediction_root / system).glob("*.jsonl"))
        if not files:
            rows[row_name] = {"status": "blocked", "reason": "missing predictions"}
            continue
        joined = join_gold_predictions(gold, load_prediction_records(files[0]))
        metrics = evaluate_pragmatic_records(joined, bootstrap_resamples=1000)
        calibration = evaluate_calibration_records(joined, target="pragmatic_polarity", bins=10)
        rows[row_name] = {
            "status": "complete",
            "macro_pragmatic_f1": metrics["metrics"]["macro_pragmatic_f1"]["value"],
            "ordinary_f1": None,
            "ece": calibration["ece"],
            "prediction_file": str(files[0]),
        }
    for row_name, reason in NOT_IMPLEMENTED_ROWS.items():
        rows[row_name] = {"status": "not_run", "reason": reason}
    output = ROOT / "results/predictions/multitask_ablation/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
