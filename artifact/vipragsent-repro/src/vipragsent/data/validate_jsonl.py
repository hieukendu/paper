from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Any

from vipragsent.data.clean_pii import detect_pii
from vipragsent.data.schema import PRAGMATIC_LABELS, canonicalize_labels, validate_record
from vipragsent.utils.io import read_jsonl, write_jsonl

RARE_MINIMUMS = {
    "sarcasm": 1200,
    "mocking": 600,
    "irony": 900,
    "idiom_figurative": 900,
    "code_switching": 1500,
    "implicit_sentiment": 1500,
}

CUE_TERMS = ["=))", ":))", "hay qua", "dinh", "vibe", "gg", "nasa", "thien tai"]


def validate_records(records: Iterable[dict[str, Any]], *, check_no_pii: bool = True) -> dict[str, Any]:
    seen: set[str] = set()
    errors: list[str] = []
    count = 0
    label_counts: Counter[str] = Counter()
    for line_no, record in enumerate(records, start=1):
        count += 1
        record_id = str(record.get("id", f"line_{line_no}"))
        if record_id in seen:
            errors.append(f"{record_id}: duplicate id")
        seen.add(record_id)
        for error in validate_record(record, allow_unlabeled=True):
            errors.append(f"{record_id}: {error}")
        if check_no_pii:
            pii = detect_pii(record.get("text", ""))
            if pii:
                errors.append(f"{record_id}: possible PII remains: {','.join(pii)}")
        labels = canonicalize_labels(record.get("labels"))
        for label in PRAGMATIC_LABELS:
            if labels.get(label) in (1, True):
                label_counts[label] += 1
    return {
        "status": "ok" if not errors else "failed",
        "records": count,
        "unique_ids": len(seen),
        "errors": errors,
        "label_counts": dict(label_counts),
        "rare_label_report": rare_label_report(label_counts),
    }


def validate_jsonl(path: str | Path, *, check_no_pii: bool = True) -> dict[str, Any]:
    return validate_records(read_jsonl(path), check_no_pii=check_no_pii)


def rare_label_report(label_counts: Counter[str] | dict[str, int]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for label, minimum in RARE_MINIMUMS.items():
        count = int(label_counts.get(label, 0))
        report[label] = {"count": count, "minimum_for_20k": minimum, "status": "ok" if count >= minimum else "low"}
    return report


def enrichment_candidates(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        text = f"{record.get('text', '')} {record.get('text_normalized', '')}".lower()
        if any(term in text for term in CUE_TERMS):
            candidate = dict(record)
            candidate["enrichment_reason"] = "cue_term"
            selected.append(candidate)
    return selected


def write_enrichment_queue(input_path: str | Path, output_path: str | Path) -> int:
    return write_jsonl(output_path, enrichment_candidates(read_jsonl(input_path)))
