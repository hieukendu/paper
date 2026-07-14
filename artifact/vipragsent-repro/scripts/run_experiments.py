from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.experiments.runner import run_no_finetune_suite
from vipragsent.utils.env import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ViPragSent no-fine-tuning experiments from gold JSONL and imported predictions."
    )
    parser.add_argument("--predictions-dir", default=str(ROOT / "results" / "predictions"))
    parser.add_argument("--vipragsent-test", default=None, help="Gold ViPragSent test JSONL.")
    parser.add_argument("--public-data", default=str(ROOT / "data" / "processed" / "all_unified.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--include-heuristic", action="store_true", help="Add a no-train smoke baseline.")
    parser.add_argument(
        "--allow-silver-gold",
        action="store_true",
        help="Allow silver/reviewed labels as gold for smoke tests only. Do not use for final claims.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests.")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    summary = run_no_finetune_suite(
        root=ROOT,
        predictions_dir=args.predictions_dir,
        vipragsent_test=args.vipragsent_test,
        public_data=args.public_data,
        output_dir=args.output_dir,
        include_heuristic=args.include_heuristic,
        allow_silver_gold=args.allow_silver_gold,
        bootstrap_resamples=args.bootstrap_resamples,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
