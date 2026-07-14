from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.annotation.batch_builder import build_batches
from vipragsent.data.schema import make_record
from vipragsent.utils.io import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Create annotation batches for agent/human review.")
    parser.add_argument("--input", default=str(ROOT / "data" / "processed" / "all_unified.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "annotation" / "batches"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.toy:
        input_path = ROOT / "data" / "interim" / "toy_unlabeled.jsonl"
        write_jsonl(
            input_path,
            [
                make_record("Hay qua =))", dataset="toy", platform="tiktok", pii_cleaned=True),
                make_record("Vibe ok, GG", dataset="toy", platform="facebook", pii_cleaned=True),
            ],
        )
    if not input_path.exists():
        print(json.dumps({"status": "pending_missing_input", "input": str(input_path)}, indent=2))
        return 0

    paths = build_batches(input_path, args.output_dir, batch_size=args.batch_size, limit=args.limit)
    print(json.dumps({"status": "ok", "batches": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
