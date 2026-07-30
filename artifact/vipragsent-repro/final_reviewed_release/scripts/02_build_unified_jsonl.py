from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.data.ingest_visobert import build_unified_records
from vipragsent.data.schema import make_record
from vipragsent.utils.io import read_jsonl
from vipragsent.utils.env import load_env_file
from vipragsent.utils.io import write_jsonl


def toy_records() -> list[dict]:
    return [
        make_record("Hay lam =)) dung la thien tai", dataset="toy", platform="tiktok", pii_cleaned=True),
        make_record("Vibe cuc manh, GG anh em", dataset="toy", platform="facebook", pii_cleaned=True),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified JSONL from local sources.")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "all_unified.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-public", action="store_true", help="Append data/interim/public_datasets_unified.jsonl if present.")
    parser.add_argument("--use-agent-prelabeled-social", action="store_true", help="Use data/interim/agent_prelabeled.jsonl for social records if present.")
    parser.add_argument("--toy", action="store_true", help="Write tiny toy data for local smoke checks.")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    if args.toy:
        count = write_jsonl(args.output, toy_records())
        print(json.dumps({"status": "ok", "mode": "toy", "records_written": count, "output": args.output}, indent=2))
        return 0

    source_root = Path(args.source_root or os.getenv("LOCAL_VISOBERT_EXPORT", r"D:\hf_cache\exports\visobert_12000_api"))
    if not source_root.exists():
        print(json.dumps({"status": "DATA_INCOMPLETE", "message": f"Missing source root: {source_root}"}, indent=2))
        return 0
    silver_social_path = ROOT / "data" / "interim" / "agent_prelabeled.jsonl"
    if args.use_agent_prelabeled_social and silver_social_path.exists():
        records = list(read_jsonl(silver_social_path))
    else:
        records = list(build_unified_records(source_root, limit=args.limit))
    public_path = ROOT / "data" / "interim" / "public_datasets_unified.jsonl"
    if args.include_public and public_path.exists():
        records.extend(read_jsonl(public_path))
    count = write_jsonl(args.output, records)
    print(json.dumps({"status": "ok", "records_written": count, "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
