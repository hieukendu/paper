import pytest

from vipragsent.data.schema import assert_valid_record, make_record, validate_record


def test_adjudicated_record_schema_is_valid():
    labels = {
        "implicit_sentiment": 1,
        "sarcasm": 1,
        "irony": 1,
        "idiom_figurative": 0,
        "code_switching": 0,
        "mocking": 1,
        "polarity": "negative",
        "emotion": "disgust",
    }
    record = make_record(
        "Hay lam =)) dung la thien tai",
        dataset="toy",
        split="train",
        labels=labels,
        label_status="adjudicated",
        pii_cleaned=True,
    )
    assert validate_record(record) == []
    assert_valid_record(record, require_adjudicated_gold=True)


def test_invalid_binary_label_is_rejected():
    record = make_record("sample", dataset="toy", labels={"sarcasm": 2})
    errors = validate_record(record)
    assert any("sarcasm" in error for error in errors)


def test_adjudicated_requires_full_labels():
    record = make_record("sample", dataset="toy", label_status="adjudicated")
    with pytest.raises(ValueError):
        assert_valid_record(record, require_adjudicated_gold=True)
