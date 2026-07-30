from vipragsent.annotation.gold_builder import build_gold_after_annotation
from vipragsent.data.schema import make_record
from vipragsent.utils.io import read_jsonl, write_jsonl


def _labels(value: int) -> dict:
    return {
        "implicit_sentiment": value,
        "sarcasm": value,
        "irony": value,
        "idiom_figurative": 0,
        "code_switching": 0,
        "mocking": value,
        "polarity": "negative" if value else "neutral",
        "emotion": "disgust" if value else "other",
    }


def test_build_gold_after_annotation_auto_agreement(tmp_path):
    batches = tmp_path / "batches"
    r1 = tmp_path / "reviewer_01"
    r2 = tmp_path / "reviewer_02"
    adj = tmp_path / "adjudicated"
    for path in (batches, r1, r2, adj):
        path.mkdir(parents=True)
    records = [
        make_record("Hay lam =))", dataset="toy", labels=_labels(1), split="unassigned"),
        make_record("Binh thuong", dataset="toy", labels=_labels(0), split="unassigned"),
    ]
    write_jsonl(batches / "batch_001_input.jsonl", records)
    reviewer_records = [
        {"id": record["id"], "annotator_id": "ann", "labels": record["labels"]}
        for record in records
    ]
    write_jsonl(r1 / "batch_001.jsonl", reviewer_records)
    write_jsonl(r2 / "batch_001.jsonl", reviewer_records)

    report = build_gold_after_annotation(
        batches_dir=batches,
        reviewer_1_dir=r1,
        reviewer_2_dir=r2,
        adjudicated_dir=adj,
        output_dir=tmp_path / "processed",
        disagreements_output=tmp_path / "disagreements.jsonl",
        report_output=tmp_path / "report.json",
    )

    assert report["final_gold_records"] == 2
    assert report["disagreements"] == 0
    gold = list(read_jsonl(tmp_path / "processed" / "vipragsent_all_adjudicated.jsonl"))
    assert {record["label_status"] for record in gold} == {"adjudicated"}
