from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from vipragsent.data.clean_pii import clean_pii
from vipragsent.data.normalize_text import normalize_text
from vipragsent.data.schema import LABEL_ALIASES, canonicalize_labels, make_record
from vipragsent.utils.hashing import sha256_file

TEXT_FIELDS = [
    "text",
    "comment",
    "content",
    "sentence",
    "review",
    "document",
    "input",
    "body",
    "raw_text",
]
SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".txt"}
SKIP_NAME_PARTS = {"summary", "readme", "license"}


def discover_source_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part in path.name.lower() for part in SKIP_NAME_PARTS)
    )


def infer_platform(path: Path, row: dict[str, Any] | None = None) -> str:
    row = row or {}
    for key in ("platform", "source_platform", "channel"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value.lower()
    lowered = path.name.lower()
    for platform in ("facebook", "tiktok", "youtube"):
        if platform in lowered:
            return platform
    return "unknown"


def extract_text(row: Any) -> str | None:
    if isinstance(row, str):
        return row.strip() or None
    if not isinstance(row, dict):
        return None
    for key in TEXT_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) >= 3:
            return value.strip()
    return None


def _json_objects(payload: Any) -> Iterator[Any]:
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("data", "records", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
        yield payload


def iter_source_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    obj = json.loads(line)
                    yield obj if isinstance(obj, dict) else {"text": str(obj)}
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for obj in _json_objects(payload):
            yield obj if isinstance(obj, dict) else {"text": str(obj)}
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
    elif suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield {"text": text}


def build_unified_records(
    root: str | Path,
    *,
    limit: int | None = None,
    created_by: str = "src/vipragsent/data/ingest_visobert.py",
) -> Iterator[dict[str, Any]]:
    seen_texts: set[str] = set()
    emitted = 0
    for path in discover_source_files(root):
        checksum = sha256_file(path)
        for row in iter_source_rows(path):
            text = extract_text(row)
            if not text:
                continue
            cleaned = clean_pii(text)
            normalized = normalize_text(cleaned)
            dedupe_key = normalized.lower()
            if dedupe_key in seen_texts:
                continue
            seen_texts.add(dedupe_key)
            yield make_record(
                cleaned,
                dataset="visobert_local",
                platform=infer_platform(path, row),
                source_path=str(path),
                text_normalized=normalized,
                license_name=str(row.get("license") or "unknown_or_dataset_card"),
                record_id=str(row.get("id")) if row.get("id") else None,
                labels=_labels_from_row(row),
                label_status=_label_status_from_row(row),
                created_by=created_by,
                pii_cleaned=True,
                checksum=checksum,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def scan_local_export(root: str | Path, *, limit_files: int | None = None) -> dict[str, Any]:
    files = discover_source_files(root)
    if limit_files is not None:
        files = files[:limit_files]
    return {
        "root": str(root),
        "exists": Path(root).exists(),
        "supported_file_count": len(files),
        "files": [{"path": str(path), "suffix": path.suffix, "bytes": path.stat().st_size} for path in files],
    }


def _labels_from_row(row: dict[str, Any]) -> dict[str, Any]:
    labels = {}
    for key in [
        "implicit_sentiment",
        "implicit",
        "sarcasm",
        "irony",
        "idiom_figurative",
        "idiom",
        "code_switching",
        "code_switch",
        "mocking",
        "polarity",
        "emotion",
    ]:
        if key in row:
            value = row[key]
            if value in ("", "None", "none", "null"):
                value = None
            labels[LABEL_ALIASES.get(key, key)] = value
    return canonicalize_labels(labels)


def _label_status_from_row(row: dict[str, Any]) -> str:
    explicit = row.get("label_status")
    if explicit in {"unlabeled", "silver_agent", "reviewed", "adjudicated"}:
        return str(explicit)
    labels = _labels_from_row(row)
    if any(value is not None for value in labels.values()):
        return "silver_agent"
    return "unlabeled"
