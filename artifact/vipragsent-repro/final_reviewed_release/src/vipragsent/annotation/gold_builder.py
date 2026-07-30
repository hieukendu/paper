from __future__ import annotations

from pathlib import Path
from typing import Any

from vipragsent.annotation.adjudication import make_adjudicated_record
from vipragsent.annotation.disagreement import disagreement_record
from vipragsent.data.schema import PRAGMATIC_LABELS, assert_valid_record, canonicalize_labels
from vipragsent.data.split_stratified import assign_multilabel_stratified_splits, find_split_overlaps, split_counts
from vipragsent.utils.io import ensure_dir, read_jsonl, write_jsonl


LABEL_FIELDS = [*PRAGMATIC_LABELS, "polarity", "emotion"]


def build_gold_after_annotation(
    *,
    batches_dir: str | Path,
    reviewer_1_dir: str | Path,
    reviewer_2_dir: str | Path,
    adjudicated_dir: str | Path,
    output_dir: str | Path,
    disagreements_output: str | Path,
    report_output: str | Path,
    seed: int = 20260520,
) -> dict[str, Any]:
    base_records = _load_by_id(Path(batches_dir))
    reviewer_1 = _load_by_id(Path(reviewer_1_dir))
    reviewer_2 = _load_by_id(Path(reviewer_2_dir))
    adjudicated = _load_by_id(Path(adjudicated_dir), required=False)

    final_records = []
    disagreements = []
    missing_review = []
    invalid = []

    for record_id, base in sorted(base_records.items()):
        adjudicated_record = adjudicated.get(record_id)
        if adjudicated_record is not None:
            final = make_adjudicated_record(
                base,
                _labels_from(adjudicated_record),
                adjudicator_id=str(
                    adjudicated_record.get("adjudicator_id")
                    or (adjudicated_record.get("annotation") or {}).get("adjudicator")
                    or "adjudicator"
                ),
            )
            _copy_annotation_ids(final, reviewer_1.get(record_id), reviewer_2.get(record_id), adjudicated_record)
        else:
            left = reviewer_1.get(record_id)
            right = reviewer_2.get(record_id)
            if left is None or right is None:
                missing_review.append(record_id)
                continue
            if _labels_from(left) != _labels_from(right):
                disagreements.append(disagreement_record(left, right))
                continue
            final = make_adjudicated_record(base, _labels_from(left), adjudicator_id="auto_agreement")
            _copy_annotation_ids(final, left, right, None)
        try:
            assert_valid_record(final, require_adjudicated_gold=True)
        except ValueError as exc:
            invalid.append({"id": record_id, "error": str(exc)})
            continue
        final_records.append(final)

    final_records = assign_multilabel_stratified_splits(final_records, seed=seed) if final_records else []
    overlaps = find_split_overlaps(final_records)
    output_dir = ensure_dir(output_dir)
    paths = {
        "all": output_dir / "vipragsent_all_adjudicated.jsonl",
        "train": output_dir / "vipragsent_train.jsonl",
        "dev": output_dir / "vipragsent_dev.jsonl",
        "test": output_dir / "vipragsent_test.jsonl",
    }
    wrote_outputs = False
    if final_records:
        write_jsonl(paths["all"], final_records)
        for split in ("train", "dev", "test"):
            write_jsonl(paths[split], [record for record in final_records if record.get("split") == split])
            nested_dir = ensure_dir(output_dir / "vipragsent")
            write_jsonl(nested_dir / f"{split}.jsonl", [record for record in final_records if record.get("split") == split])
        wrote_outputs = True
    write_jsonl(disagreements_output, disagreements)

    report = {
        "status": _status_for_report(final_records, overlaps, invalid, missing_review, disagreements),
        "base_records": len(base_records),
        "final_gold_records": len(final_records),
        "missing_review_records": len(missing_review),
        "disagreements": len(disagreements),
        "invalid_records": invalid,
        "split_counts": split_counts(final_records),
        "split_overlaps": overlaps,
        "outputs": {key: str(value) for key, value in paths.items()} if wrote_outputs else {},
        "wrote_gold_outputs": wrote_outputs,
        "disagreements_output": str(disagreements_output),
    }
    ensure_dir(Path(report_output).parent)
    import json

    Path(report_output).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_by_id(directory: Path, *, required: bool = True) -> dict[str, dict]:
    if not directory.exists():
        if required:
            return {}
        return {}
    out = {}
    for path in sorted(directory.glob("*.jsonl")):
        for record in read_jsonl(path):
            record_id = str(record.get("id"))
            if record_id and record_id != "None":
                out[record_id] = record
    return out


def _labels_from(record: dict[str, Any]) -> dict[str, Any]:
    labels = record.get("labels") if isinstance(record.get("labels"), dict) else record
    return {field: canonicalize_labels(labels).get(field) for field in LABEL_FIELDS}


def _copy_annotation_ids(final: dict, left: dict | None, right: dict | None, adjudicated: dict | None) -> None:
    final.setdefault("annotation", {})
    if left is not None:
        final["annotation"]["reviewer_1"] = left.get("annotator_id") or (left.get("annotation") or {}).get("reviewer_1")
    if right is not None:
        final["annotation"]["reviewer_2"] = right.get("annotator_id") or (right.get("annotation") or {}).get("reviewer_2")
    if adjudicated is not None:
        final["annotation"]["adjudicator"] = (
            adjudicated.get("adjudicator_id")
            or (adjudicated.get("annotation") or {}).get("adjudicator")
            or final["annotation"].get("adjudicator")
        )


def _status_for_report(
    final_records: list[dict],
    overlaps: dict[str, list[str]],
    invalid: list[dict],
    missing_review: list[str],
    disagreements: list[dict],
) -> str:
    if not final_records:
        return "pending_missing_human_annotations"
    if overlaps or invalid:
        return "needs_attention"
    if missing_review or disagreements:
        return "partial_needs_adjudication"
    return "ok"
