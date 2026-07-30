from __future__ import annotations

import csv
from pathlib import Path

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import ensure_dir, read_jsonl


def export_review_csv(input_jsonl: str | Path, output_csv: str | Path) -> int:
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    fields = ["id", "batch_id", "platform", "text", *PRAGMATIC_LABELS, "polarity", "emotion", "rationale_free_text"]
    count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in read_jsonl(input_jsonl):
            writer.writerow(
                {
                    "id": record.get("id"),
                    "batch_id": record.get("batch_id") or record.get("annotation", {}).get("batch_id"),
                    "platform": record.get("platform") or record.get("source", {}).get("platform"),
                    "text": record.get("text"),
                }
            )
            count += 1
    return count
