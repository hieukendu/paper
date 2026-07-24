#!/usr/bin/env python3
"""Train-set logistic calibration over frozen ViPragSent output logits.

This is a post-hoc calibration layer, not a neural classifier: frozen model
weights are never opened for writing and no gradient-based neural training is
performed.  Model/regularization choices and decision thresholds are selected
only from the development split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl

CS = [0.01, 0.1, 1.0, 10.0, 100.0]
CLASS_WEIGHTS = [None, "balanced"]
THRESHOLDS = [round(value / 100, 2) for value in range(1, 100)]


def _members(path: Path) -> dict[str, list[dict]]:
    files = sorted(path.glob("reproduction_*.jsonl"))
    if not files:
        raise ValueError(f"no reproduction predictions in {path}")
    grouped: dict[str, list[dict]] = {}
    for file in files:
        for row in read_jsonl(file):
            grouped.setdefault(str(row["id"]), []).append(row)
    if any(len(rows) != len(files) for rows in grouped.values()):
        raise ValueError("incomplete ensemble predictions")
    return grouped


def _features(path: Path, ids: list[str]) -> np.ndarray:
    grouped = _members(path)
    if set(ids) != set(grouped):
        raise ValueError("gold/prediction IDs differ")
    features = []
    for record_id in ids:
        rows = grouped[record_id]
        feature = []
        for group, expected in (("pragmatic", 6), ("polarity", 3), ("emotion", 7)):
            values = np.array([row["logits"][group] for row in rows], dtype=np.float64)
            if values.shape != (len(rows), expected):
                raise ValueError(f"unexpected {group} logit shape for {record_id}")
            feature.extend(values.mean(axis=0).tolist())
        features.append(feature)
    return np.asarray(features, dtype=np.float64)


def _gold(path: Path) -> tuple[list[str], dict[str, dict]]:
    rows = list(read_jsonl(path)); ids = [str(row["id"]) for row in rows]
    return ids, {str(row["id"]): row["labels"] for row in rows}


def _probability(features: np.ndarray, state: dict) -> np.ndarray:
    mean = np.asarray(state["feature_mean"]); scale = np.asarray(state["feature_scale"])
    coefficient = np.asarray(state["coefficient"]); intercept = float(state["intercept"])
    logits = ((features - mean) / scale) @ coefficient + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def fit(args: argparse.Namespace) -> None:
    train_ids, train_gold = _gold(args.train_gold); dev_ids, dev_gold = _gold(args.dev_gold)
    train_features = _features(args.train_predictions, train_ids); dev_features = _features(args.dev_predictions, dev_ids)
    mean = train_features.mean(axis=0); scale = train_features.std(axis=0); scale[scale == 0] = 1.0
    train_x = (train_features - mean) / scale; dev_x = (dev_features - mean) / scale
    labels = {}
    for label in PRAGMATIC_LABELS:
        train_y = np.asarray([int(train_gold[record_id][label]) for record_id in train_ids])
        dev_y = [int(dev_gold[record_id][label]) for record_id in dev_ids]
        best = (-1.0, None)
        for c in CS:
            for class_weight in CLASS_WEIGHTS:
                estimator = LogisticRegression(C=c, class_weight=class_weight, solver="liblinear", max_iter=2000, random_state=0)
                estimator.fit(train_x, train_y)
                probabilities = estimator.predict_proba(dev_x)[:, 1]
                for threshold in THRESHOLDS:
                    score = binary_macro_f1(dev_y, [int(value >= threshold) for value in probabilities]) * 100
                    candidate = (score, -abs(threshold - 0.5), -abs(np.log10(c)), class_weight == None, c, threshold, estimator)
                    if best[1] is None or candidate[:-1] > best[0]:
                        best = (candidate[:-1], estimator)
        score, _, _, _, c, threshold = best[0]
        estimator = best[1]
        labels[label] = {
            "C": c, "class_weight": estimator.class_weight, "threshold": threshold, "development_binary_macro_f1": round(score, 4),
            "coefficient": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
            "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        }
    payload = {"method": "train_logistic_calibration_over_frozen_logits", "selection_split": "development", "train_gold_path": str(args.train_gold), "dev_gold_path": str(args.dev_gold), "feature_layout": ["pragmatic_logits_6", "polarity_logits_3", "emotion_logits_7"], "labels": labels, "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: value["development_binary_macro_f1"] for label, value in labels.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_ids, _ = _gold(args.data); features = _features(args.predictions, data_ids)
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    by_id = {row["id"]: index for index, row in enumerate(rows)}
    for label, state in config["labels"].items():
        probabilities = _probability(features, state)
        for record_id, probability in zip(data_ids, probabilities):
            row = rows[by_id[record_id]]
            row["probabilities"][f"{label}_logit_calibrated"] = float(probability)
            row["predictions"][label] = int(probability >= state["threshold"])
    for row in rows: row["system"] = "vipragsent_frozen_logit_calibration"
    write_jsonl(args.output, rows); print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--train-gold", type=Path, required=True); fit_parser.add_argument("--train-predictions", type=Path, required=True)
    fit_parser.add_argument("--dev-gold", type=Path, required=True); fit_parser.add_argument("--dev-predictions", type=Path, required=True); fit_parser.add_argument("--output", type=Path, required=True); fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--config", type=Path, required=True); apply_parser.add_argument("--data", type=Path, required=True); apply_parser.add_argument("--predictions", type=Path, required=True); apply_parser.add_argument("--base-config", type=Path, required=True); apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
