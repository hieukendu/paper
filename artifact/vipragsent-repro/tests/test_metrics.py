from vipragsent.evaluation.calibration import expected_calibration_error
from vipragsent.evaluation.metrics import binary_macro_f1, pragmatic_f1


def test_binary_macro_f1():
    assert binary_macro_f1([0, 0, 1, 1], [0, 1, 1, 0]) == 0.5


def test_pragmatic_f1_returns_macro_key():
    labels = {
        "implicit_sentiment": 1,
        "sarcasm": 1,
        "irony": 0,
        "idiom_figurative": 0,
        "code_switching": 1,
        "mocking": 0,
    }
    metrics = pragmatic_f1([labels], [labels])
    assert metrics["macro_pragmatic_f1"] == 50.0


def test_ece_bins():
    report = expected_calibration_error([0.9, 0.2], [1, 0], bins=10)
    assert report["ece"] >= 0
    assert len(report["bins"]) == 10
