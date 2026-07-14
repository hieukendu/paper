from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.annotation.llm_labeler import run_prelabelling
from vipragsent.utils.env import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Create local-reasoning silver agent labels; never gold labels.")
    parser.add_argument("--input", default=str(ROOT / "data" / "annotation" / "batches" / "batch_001_input.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "annotation" / "agent_silver" / "batch_001_agent_a.jsonl"))
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI for silver labels; otherwise use local heuristic.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rate-limit-seconds", type=float, default=0.0)
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    if not Path(args.input).exists():
        print(json.dumps({"status": "pending_missing_input", "input": args.input}, indent=2))
        return 0
    report = run_prelabelling(args.input, args.output, use_azure=args.azure, limit=args.limit, rate_limit_seconds=args.rate_limit_seconds)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
