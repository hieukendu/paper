from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.utils.io import read_jsonl, write_jsonl


def main() -> int:
    train_path = ROOT / "data/processed/vipragsent_train.jsonl"
    rationale_path = ROOT / "data/generated/rationales.jsonl"
    output_path = ROOT / "data/processed/vipragsent_train_with_rationales.jsonl"
    train = list(read_jsonl(train_path))
    rationales = {str(row["id"]): row for row in read_jsonl(rationale_path)}
    missing = [row["id"] for row in train if row["id"] not in rationales]
    empty = [record_id for record_id, row in rationales.items() if not str(row.get("rationale") or "").strip()]
    if missing or empty:
        print(json.dumps({"status": "blocked", "train": len(train), "rationales": len(rationales), "missing": len(missing), "empty": len(empty)}, indent=2))
        return 2
    output = []
    for row in train:
        enriched = dict(row)
        generated = rationales[row["id"]]
        enriched["rationale"] = generated["rationale"]
        enriched["rationale_generator"] = {key: value for key, value in generated.get("generator", {}).items() if key != "raw_response"}
        enriched["rationale_audit"] = {"status": "waived", "faithful": "assumed_by_owner"}
        output.append(enriched)
    write_jsonl(output_path, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
