from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.experiments.baselines import heuristic_prediction_records
from vipragsent.utils.io import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Write no-train heuristic predictions for smoke evaluation.")
    parser.add_argument("--input", required=True, help="Gold/unlabeled JSONL with id and text.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--system", default="heuristic_local")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    prediction_records = heuristic_prediction_records(records, system=args.system, seed=args.seed)
    count = write_jsonl(args.output, prediction_records)
    print(json.dumps({"status": "ok", "output": args.output, "records": count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
