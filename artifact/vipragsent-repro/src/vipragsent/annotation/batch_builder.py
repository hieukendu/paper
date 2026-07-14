from __future__ import annotations

from pathlib import Path

from vipragsent.data.schema import source_type_for
from vipragsent.utils.io import ensure_dir, read_jsonl, write_jsonl


def annotation_input(record: dict, batch_id: str) -> dict:
    source = record.get("source") or {}
    split = record.get("split")
    platform = record.get("platform") or source.get("platform", "unknown")
    return {
        "id": record["id"],
        "batch_id": batch_id,
        "source": source,
        "split": split,
        "type": record.get("type", source_type_for(source.get("dataset"), split)),
        "platform": platform,
        "text": record["text"],
        "text_normalized": record.get("text_normalized"),
        "label_status": record.get("label_status", "unlabeled"),
    }


def build_batches(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 100,
    limit: int | None = None,
    prefix: str = "batch",
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    records = list(read_jsonl(input_path))
    if limit is not None:
        records = records[:limit]
    paths: list[Path] = []
    for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
        batch_id = f"{prefix}_{batch_index:03d}"
        batch = [annotation_input(record, batch_id) for record in records[start : start + batch_size]]
        path = output_dir / f"{batch_id}_input.jsonl"
        write_jsonl(path, batch)
        paths.append(path)
    return paths
