from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from vipragsent.utils.hashing import sha256_text, short_hash

SCHEMA_VERSION = "1.0"

PRAGMATIC_LABELS = [
    "implicit_sentiment",
    "sarcasm",
    "irony",
    "idiom_figurative",
    "code_switching",
    "mocking",
]

POLARITY_LABELS = {"positive", "neutral", "negative"}
EMOTION_LABELS = {"enjoyment", "sadness", "anger", "disgust", "fear", "surprise", "other"}
LABEL_STATUSES = {"unlabeled", "silver_agent", "reviewed", "adjudicated"}
SPLITS = {"train", "dev", "test", "unassigned"}
SOURCE_SPLIT_TYPES = {"train", "dev", "test"}
DATASETS_WITH_SOURCE_SPLITS = {"uit_vsfc", "uit_vsmec", "aivivn_2019"}

LABEL_ALIASES = {
    "implicit": "implicit_sentiment",
    "idiom": "idiom_figurative",
    "figurative": "idiom_figurative",
    "code_switch": "code_switching",
    "codeswitch": "code_switching",
    "emotion_vsmec": "emotion",
    "polarity_signed_3way": "polarity",
}


class SchemaError(ValueError):
    """Raised when a unified JSONL record violates the setup schema."""


def canonical_label_name(name: str) -> str:
    return LABEL_ALIASES.get(name, name)


def canonicalize_labels(labels: dict[str, Any] | None) -> dict[str, Any]:
    labels = labels or {}
    out = empty_labels()
    for key, value in labels.items():
        out[canonical_label_name(key)] = value
    return out


def empty_labels() -> dict[str, Any]:
    return {**{label: None for label in PRAGMATIC_LABELS}, "polarity": None, "emotion": None}


def available_labels_from(labels: dict[str, Any]) -> dict[str, bool]:
    labels = canonicalize_labels(labels)
    pragmatic = all(labels.get(label) in (0, 1, False, True) for label in PRAGMATIC_LABELS)
    polarity = labels.get("polarity") in POLARITY_LABELS
    emotion = labels.get("emotion") in EMOTION_LABELS
    return {"pragmatic": pragmatic, "polarity": polarity, "emotion": emotion}


def stable_record_id(text: str, dataset: str = "unknown") -> str:
    return "vp_" + short_hash(f"{dataset}\n{text}", size=12)


def source_type_for(dataset: str | None, split: str | None) -> str | None:
    if dataset in DATASETS_WITH_SOURCE_SPLITS and split in SOURCE_SPLIT_TYPES:
        return split
    return None


def make_record(
    text: str,
    *,
    dataset: str,
    platform: str = "unknown",
    source_path: str | None = None,
    url: str | None = None,
    license_name: str = "unknown_or_dataset_card",
    split: str = "unassigned",
    record_id: str | None = None,
    text_normalized: str | None = None,
    text_segmented: str | None = None,
    labels: dict[str, Any] | None = None,
    label_status: str = "unlabeled",
    created_by: str = "vipragsent.data.schema.make_record",
    pii_cleaned: bool = False,
    checksum: str | None = None,
    split_type: str | None = None,
) -> dict[str, Any]:
    labels = canonicalize_labels(labels)
    record_type = split_type if split_type in SOURCE_SPLIT_TYPES else source_type_for(dataset, split)
    return {
        "id": record_id or stable_record_id(text, dataset),
        "source": {
            "dataset": dataset,
            "platform": platform,
            "license": license_name,
            "source_path": source_path,
            "url": url,
            "checksum": checksum or sha256_text(text),
        },
        "split": split,
        "type": record_type,
        "platform": platform,
        "text": text,
        "text_normalized": text_normalized,
        "text_segmented": text_segmented,
        "labels": labels,
        "available_labels": available_labels_from(labels),
        "label_status": label_status,
        "annotation": {
            "batch_id": None,
            "reviewer_1": None,
            "reviewer_2": None,
            "adjudicator": None,
            "needs_human_review": label_status != "adjudicated",
        },
        "rationale": None,
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "created_by": created_by,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pii_cleaned": pii_cleaned,
        },
    }


def validate_record(
    record: dict[str, Any],
    *,
    allow_unlabeled: bool = True,
    require_adjudicated_gold: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(record.get("id"), str) or not record.get("id"):
        errors.append("id must be a non-empty string")
    if not isinstance(record.get("text"), str) or not record.get("text").strip():
        errors.append("text must be a non-empty string")
    if record.get("split") not in SPLITS:
        errors.append(f"split must be one of {sorted(SPLITS)}")
    if record.get("type") not in SOURCE_SPLIT_TYPES and record.get("type") is not None:
        errors.append(f"type must be one of {sorted(SOURCE_SPLIT_TYPES)} or null")
    if not isinstance(record.get("source"), dict):
        errors.append("source must be an object")
    if record.get("label_status") not in LABEL_STATUSES:
        errors.append(f"label_status must be one of {sorted(LABEL_STATUSES)}")

    labels = canonicalize_labels(record.get("labels"))
    for label in PRAGMATIC_LABELS:
        value = labels.get(label)
        if value is None and allow_unlabeled:
            continue
        if value not in (0, 1, False, True):
            errors.append(f"labels.{label} must be 0/1 or null")

    polarity = labels.get("polarity")
    if not (polarity is None and allow_unlabeled) and polarity not in POLARITY_LABELS:
        errors.append(f"labels.polarity must be one of {sorted(POLARITY_LABELS)} or null")

    emotion = labels.get("emotion")
    if not (emotion is None and allow_unlabeled) and emotion not in EMOTION_LABELS:
        errors.append(f"labels.emotion must be one of {sorted(EMOTION_LABELS)} or null")

    if require_adjudicated_gold or record.get("label_status") == "adjudicated":
        if record.get("label_status") != "adjudicated":
            errors.append("gold records must have label_status=adjudicated")
        available = available_labels_from(labels)
        if not all(available.values()):
            errors.append("adjudicated records must have all pragmatic, polarity and emotion labels")

    return errors


def assert_valid_record(record: dict[str, Any], **kwargs: Any) -> None:
    errors = validate_record(record, **kwargs)
    if errors:
        identifier = record.get("id", "<missing-id>")
        raise SchemaError(f"{identifier}: " + "; ".join(errors))
