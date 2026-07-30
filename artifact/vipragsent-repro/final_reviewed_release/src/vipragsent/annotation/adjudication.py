from __future__ import annotations

from vipragsent.data.schema import available_labels_from, canonicalize_labels


def make_adjudicated_record(base_record: dict, final_labels: dict, *, adjudicator_id: str) -> dict:
    output = dict(base_record)
    labels = canonicalize_labels(final_labels)
    output["labels"] = labels
    output["available_labels"] = available_labels_from(labels)
    output["label_status"] = "adjudicated"
    output.setdefault("annotation", {})
    output["annotation"]["adjudicator"] = adjudicator_id
    output["annotation"]["needs_human_review"] = False
    return output


def merge_if_reviewers_agree(base_record: dict, reviewer_1: dict, reviewer_2: dict) -> dict | None:
    labels_1 = canonicalize_labels(reviewer_1.get("labels"))
    labels_2 = canonicalize_labels(reviewer_2.get("labels"))
    if labels_1 != labels_2:
        return None
    output = dict(base_record)
    output["labels"] = labels_1
    output["available_labels"] = available_labels_from(labels_1)
    output["label_status"] = "reviewed"
    output.setdefault("annotation", {})
    output["annotation"]["reviewer_1"] = reviewer_1.get("annotator_id")
    output["annotation"]["reviewer_2"] = reviewer_2.get("annotator_id")
    output["annotation"]["needs_human_review"] = True
    return output
