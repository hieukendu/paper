from __future__ import annotations

import csv
from pathlib import Path

from vipragsent.utils.io import ensure_dir


LEDGER_FIELDS = ["anchor", "claim", "source_file", "json_path", "script_hash", "status"]


def write_claim_ledger(path: str | Path, rows: list[dict]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})
    return len(rows)


def pending_claim_rows() -> list[dict]:
    return [
        {
            "anchor": "Table2.main_pragmatic",
            "claim": "Pending Q1 pragmatic metrics",
            "source_file": "results/pending/main_pragmatic.pending.json",
            "json_path": "$",
            "script_hash": "pending",
            "status": "pending",
        },
        {
            "anchor": "Figure7.calibration",
            "claim": "Pending Q4 calibration metrics",
            "source_file": "results/pending/calibration.pending.json",
            "json_path": "$",
            "script_hash": "pending",
            "status": "pending",
        },
    ]
