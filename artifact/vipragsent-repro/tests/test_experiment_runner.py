from vipragsent.data.schema import make_record
from vipragsent.experiments.evaluator import evaluate_pragmatic_records
from vipragsent.experiments.predictions import join_gold_predictions
from vipragsent.experiments.runner import run_no_finetune_suite
from vipragsent.utils.io import write_jsonl


def _labels(value: int, polarity: str = "negative") -> dict:
    return {
        "implicit_sentiment": value,
        "sarcasm": value,
        "irony": value,
        "idiom_figurative": value,
        "code_switching": value,
        "mocking": value,
        "polarity": polarity,
        "emotion": "disgust" if polarity == "negative" else "other",
    }


def test_evaluate_pragmatic_records_from_joined_predictions():
    gold = [
        make_record("Hay lam =))", dataset="toy", split="test", labels=_labels(1), label_status="adjudicated"),
        make_record("Binh thuong", dataset="toy", split="test", labels=_labels(0, "neutral"), label_status="adjudicated"),
    ]
    predictions = [{"id": record["id"], "predictions": record["labels"]} for record in gold]
    joined = join_gold_predictions(gold, predictions)
    report = evaluate_pragmatic_records(joined, bootstrap_resamples=10)
    assert report["metrics"]["macro_pragmatic_f1"]["value"] == 100.0


def test_no_finetune_suite_writes_results_with_heuristic(tmp_path):
    gold = [
        make_record("Hay lam =))", dataset="toy", split="test", labels=_labels(1), label_status="adjudicated"),
        make_record("Binh thuong", dataset="toy", split="test", labels=_labels(0, "neutral"), label_status="adjudicated"),
    ]
    gold_path = tmp_path / "vipragsent_test.jsonl"
    write_jsonl(gold_path, gold)
    summary = run_no_finetune_suite(
        root=tmp_path,
        predictions_dir=tmp_path / "predictions",
        vipragsent_test=gold_path,
        external_evaluation_data=gold_path,
        output_dir=tmp_path / "results",
        include_heuristic=True,
        bootstrap_resamples=10,
    )
    assert summary["status"] == "ok"
    assert (tmp_path / "results" / "main_pragmatic.json").exists()
