from vipragsent.annotation.batch_builder import annotation_input
from vipragsent.data.schema import make_record, source_type_for


def test_source_type_public_dataset_follows_split():
    assert source_type_for("uit_vsfc", "train") == "train"
    assert source_type_for("uit_vsmec", "dev") == "dev"
    assert source_type_for("aivivn_2019", "test") == "test"


def test_source_type_visobert_is_null_even_when_unassigned():
    assert source_type_for("visobert_local", "unassigned") is None


def test_annotation_input_preserves_source_split_and_type():
    record = make_record(
        "good",
        dataset="uit_vsfc",
        platform="student_feedback",
        split="dev",
        pii_cleaned=True,
    )
    payload = annotation_input(record, "batch_001")
    assert payload["source"]["dataset"] == "uit_vsfc"
    assert payload["split"] == "dev"
    assert payload["type"] == "dev"
    assert payload["platform"] == "student_feedback"
