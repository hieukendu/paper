from __future__ import annotations

from vipragsent.data.schema import PRAGMATIC_LABELS, canonicalize_labels

LABEL_FIELDS = PRAGMATIC_LABELS + ["polarity", "emotion"]


def disagreement_fields(left: dict, right: dict) -> list[str]:
    left_labels = canonicalize_labels(left.get("labels"))
    right_labels = canonicalize_labels(right.get("labels"))
    return [field for field in LABEL_FIELDS if left_labels.get(field) != right_labels.get(field)]


def disagreement_record(left: dict, right: dict) -> dict:
    fields = disagreement_fields(left, right)
    return {
        "id": left.get("id") or right.get("id"),
        "reviewer_1": left,
        "reviewer_2": right,
        "disagreement_fields": fields,
        "needs_adjudication": bool(fields),
    }
