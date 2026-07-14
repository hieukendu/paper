from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.data.schema import make_record
from vipragsent.data.validate_jsonl import validate_jsonl, validate_records, write_enrichment_queue


def toy_records() -> list[dict]:
    labels = {
        "implicit_sentiment": 1,
        "sarcasm": 1,
        "irony": 1,
        "idiom_figurative": 0,
        "code_switching": 0,
        "mocking": 1,
        "polarity": "negative",
        "emotion": "disgust",
    }
    return [
        make_record(
            "Hay lam =)) dung la thien tai",
            dataset="toy",
            platform="tiktok",
            labels=labels,
            label_status="adjudicated",
            split="train",
            pii_cleaned=True,
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate processed JSONL files and toy schema.")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--write-enrichment", action="store_true")
    args = parser.parse_args()

    if args.toy:
        report = validate_records(toy_records())
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1

    paths = [Path(path) for path in args.input] or [
        ROOT / "data" / "processed" / "vipragsent_train.jsonl",
        ROOT / "data" / "processed" / "vipragsent_dev.jsonl",
        ROOT / "data" / "processed" / "vipragsent_test.jsonl",
    ]
    reports = {}
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        reports[str(path)] = validate_jsonl(path)
        if args.write_enrichment:
            write_enrichment_queue(path, ROOT / "data" / "annotation" / "enrichment_queue.jsonl")
    status = "ok" if reports and all(report["status"] == "ok" for report in reports.values()) else "pending_or_failed"
    print(json.dumps({"status": status, "missing": missing, "reports": reports}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status in {"ok", "pending_or_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
