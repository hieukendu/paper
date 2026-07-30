from __future__ import annotations

from vipragsent.data.schema import PRAGMATIC_LABELS


def f1_for_class(y_true: list[int], y_pred: list[int], cls: int) -> float:
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
    denom = (2 * tp) + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def binary_macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    return (f1_for_class(y_true, y_pred, 0) + f1_for_class(y_true, y_pred, 1)) / 2


def multiclass_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    if not labels:
        raise ValueError("labels must not be empty")
    scores = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(scores) / len(scores)


def pragmatic_f1(golds: list[dict], preds: list[dict]) -> dict[str, float]:
    if len(golds) != len(preds):
        raise ValueError("golds and preds must have equal lengths")
    out: dict[str, float] = {}
    for field in PRAGMATIC_LABELS:
        y_true = [int(record[field]) for record in golds]
        y_pred = [int(record[field]) for record in preds]
        out[field] = binary_macro_f1(y_true, y_pred) * 100
    out["macro_pragmatic_f1"] = sum(out[field] for field in PRAGMATIC_LABELS) / len(PRAGMATIC_LABELS)
    return out
