from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.annotation.disagreement import disagreement_record
from vipragsent.utils.io import read_jsonl, write_jsonl


def by_id(path: str | Path) -> dict[str, dict]:
    return {str(record.get("id")): record for record in read_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge human reviewer files and create disagreements.")
    parser.add_argument("--reviewer-1", default=str(ROOT / "data" / "annotation" / "reviewer_01" / "batch_001.jsonl"))
    parser.add_argument("--reviewer-2", default=str(ROOT / "data" / "annotation" / "reviewer_02" / "batch_001.jsonl"))
    parser.add_argument("--disagreements-output", default=str(ROOT / "data" / "annotation" / "disagreements" / "batch_001_disagreements.jsonl"))
    args = parser.parse_args()

    if not Path(args.reviewer_1).exists() or not Path(args.reviewer_2).exists():
        print(json.dumps({"status": "pending_human_review", "reviewer_1": args.reviewer_1, "reviewer_2": args.reviewer_2}, indent=2))
        return 0

    left = by_id(args.reviewer_1)
    right = by_id(args.reviewer_2)
    shared_ids = sorted(set(left) & set(right))
    disagreements = [disagreement_record(left[item_id], right[item_id]) for item_id in shared_ids]
    disagreements = [record for record in disagreements if record["needs_adjudication"]]
    count = write_jsonl(args.disagreements_output, disagreements)
    print(json.dumps({"status": "ok", "shared_records": len(shared_ids), "disagreements": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
