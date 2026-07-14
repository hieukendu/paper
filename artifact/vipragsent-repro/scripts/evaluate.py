from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.experiments.evaluator import evaluate_calibration_records, evaluate_confusion_records, evaluate_pragmatic_records, load_gold_records
from vipragsent.experiments.predictions import join_gold_predictions, load_prediction_records
from vipragsent.evaluation.metrics import pragmatic_f1
from vipragsent.utils.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate existing predictions only; does not train.")
    parser.add_argument("--predictions", required=False, help="JSONL with labels and predictions objects.")
    parser.add_argument("--gold", required=False, help="Gold JSONL. If omitted, predictions must contain labels.")
    parser.add_argument("--task", choices=["pragmatic", "calibration", "confusion"], default="pragmatic")
    parser.add_argument("--allow-silver-gold", action="store_true", help="Smoke-test only.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Optional gold-record limit for smoke tests.")
    args = parser.parse_args()

    if not args.predictions or not Path(args.predictions).exists():
        print(json.dumps({"status": "pending_missing_predictions", "message": "No evaluation run was executed."}, indent=2))
        return 0
    if args.gold:
        gold_records = load_gold_records(
            args.gold,
            task="pragmatic",
            require_adjudicated=True,
            allow_silver=args.allow_silver_gold,
            limit=args.limit,
        )
        joined = join_gold_predictions(gold_records, load_prediction_records(args.predictions))
        if args.task == "calibration":
            payload = evaluate_calibration_records(joined)
        elif args.task == "confusion":
            payload = evaluate_confusion_records(joined)
        else:
            payload = evaluate_pragmatic_records(joined, bootstrap_resamples=args.bootstrap_resamples)
        print(json.dumps({"status": "ok", **payload}, ensure_ascii=False, indent=2))
        return 0
    records = list(read_jsonl(args.predictions))
    golds = [record["labels"] for record in records]
    preds = [record["predictions"] for record in records]
    print(json.dumps({"status": "ok", "metrics": pragmatic_f1(golds, preds)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
