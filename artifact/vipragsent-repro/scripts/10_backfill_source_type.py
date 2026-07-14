from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.data.schema import source_type_for
from vipragsent.utils.io import dump_json, read_jsonl, write_jsonl


BACKFILL_FILES = [
    ROOT / "data" / "processed" / "all_unified.jsonl",
    ROOT / "data" / "interim" / "public_datasets_unified.jsonl",
    ROOT / "data" / "interim" / "agent_prelabeled.jsonl",
    ROOT / "data" / "interim" / "cleaned_social.jsonl",
    ROOT / "data" / "annotation" / "enrichment_queue.jsonl",
]

BACKFILL_GLOBS = [
    ROOT / "data" / "annotation" / "batches",
    ROOT / "data" / "annotation" / "agent_silver",
]

AIVIVN_DEV_NOTE = "derived_from_train_csv_seed_20260520_ratio_0.10"


def build_master_index(master_path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in read_jsonl(master_path)}


def normalized_record(record: dict[str, Any], master: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    output = dict(record)
    original = master.get(str(output.get("id")))
    changed = False

    if original:
        for field in ("source", "split"):
            if field not in output or output.get(field) in (None, "", {}):
                output[field] = original.get(field)
                changed = True

    source = output.get("source") or {}
    split = output.get("split")
    record_type = source_type_for(source.get("dataset"), split)
    if "type" not in output or output.get("type") != record_type:
        output["type"] = record_type
        changed = True

    platform = output.get("platform") or source.get("platform")
    if platform and output.get("platform") != platform:
        output["platform"] = platform
        changed = True

    if source.get("dataset") == "aivivn_2019" and record_type == "dev":
        provenance = dict(output.get("provenance") or {})
        if provenance.get("source_type_note") != AIVIVN_DEV_NOTE:
            provenance["source_type_note"] = AIVIVN_DEV_NOTE
            output["provenance"] = provenance
            changed = True

    return output, changed


def backfill_path(path: Path, master: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "missing"}
    records = []
    changed_count = 0
    type_counts: Counter[str] = Counter()
    dataset_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_index = 0
    for record in read_jsonl(path):
        if str(record.get("id")) not in master:
            missing_index += 1
        normalized, changed = normalized_record(record, master)
        records.append(normalized)
        changed_count += int(changed)
        type_key = normalized.get("type") if normalized.get("type") is not None else "null"
        dataset = (normalized.get("source") or {}).get("dataset", "unknown")
        type_counts[type_key] += 1
        dataset_type_counts[dataset][type_key] += 1
    write_jsonl(path, records)
    return {
        "path": str(path.relative_to(ROOT)),
        "status": "ok",
        "records": len(records),
        "changed_records": changed_count,
        "missing_master_index": missing_index,
        "type_counts": dict(sorted(type_counts.items())),
        "dataset_type_counts": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(dataset_type_counts.items())
        },
    }


def target_paths() -> list[Path]:
    paths = [path for path in BACKFILL_FILES if path.exists()]
    for folder in BACKFILL_GLOBS:
        if folder.exists():
            paths.extend(sorted(folder.glob("*.jsonl")))
    return paths


def main() -> int:
    master_path = ROOT / "data" / "processed" / "all_unified.jsonl"
    if not master_path.exists():
        print(json.dumps({"status": "missing_master", "path": str(master_path)}, indent=2))
        return 1

    master = build_master_index(master_path)
    reports = [backfill_path(path, master) for path in target_paths()]
    summary = {
        "status": "ok",
        "policy": {
            "type_values": ["train", "dev", "test", None],
            "public_dataset_rule": "type mirrors source split for UIT-VSFC, UIT-VSMEC and AIVIVN.",
            "visobert_rule": "type is null for visobert_local; split must be assigned later after human review.",
            "aivivn_dev_note": AIVIVN_DEV_NOTE,
        },
        "files": reports,
    }
    dump_json(ROOT / "data" / "manifest" / "source_type_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
