#!/usr/bin/env python3
"""Train-only lexical-neighbour retrieval rule for frozen code-switch labels.

The text representation is an unsupervised character/word preprocessing
transform fitted on train text only.  Labels participate only in the final
non-parametric neighbour average; no classifier, gradient, or optimizer is
created.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_code_switch_rules import ALPHAS, metric_grid
from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.utils.io import read_jsonl, write_jsonl

REPRESENTATIONS = [
    {"name": "char_2_4", "analyzer": "char_wb", "ngram_range": [2, 4]},
    {"name": "char_3_5", "analyzer": "char_wb", "ngram_range": [3, 5]},
    {"name": "char_3_6", "analyzer": "char_wb", "ngram_range": [3, 6]},
    {"name": "word_1_2", "analyzer": "word", "ngram_range": [1, 2]},
]
K_VALUES = [1, 3, 5, 11, 25, 50, 100]
POWERS = [0.0, 1.0, 2.0, 4.0]


def source(row: dict) -> str:
    return str((row.get("source") or {}).get("dataset") or "unknown")


def vectorizer(rep: dict) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=rep["analyzer"], ngram_range=tuple(rep["ngram_range"]), lowercase=True, sublinear_tf=True, norm="l2", min_df=1)


def scores(train_rows: list[dict], query_rows: list[dict], rep: dict, k_values: list[int], powers: list[float]) -> dict[tuple[int, float], np.ndarray]:
    result = {(k, power): np.zeros(len(query_rows), dtype=np.float64) for k in k_values for power in powers}
    by_source_train, by_source_query = {}, {}
    for index, row in enumerate(train_rows): by_source_train.setdefault(source(row), []).append(index)
    for index, row in enumerate(query_rows): by_source_query.setdefault(source(row), []).append(index)
    for key, query_indices in by_source_query.items():
        train_indices = by_source_train.get(key, list(range(len(train_rows))))
        train_text = [str(train_rows[index]["text"]) for index in train_indices]
        query_text = [str(query_rows[index]["text"]) for index in query_indices]
        transform = vectorizer(rep)
        train_matrix = transform.fit_transform(train_text)
        query_matrix = transform.transform(query_text)
        similarity = (query_matrix @ train_matrix.T).toarray()
        target = np.asarray([int(train_rows[index]["labels"]["code_switching"]) for index in train_indices], dtype=np.float64)
        max_k = min(max(k_values), len(train_indices))
        top_indices = np.argpartition(-similarity, kth=max_k - 1, axis=1)[:, :max_k]
        top_values = np.take_along_axis(similarity, top_indices, axis=1)
        order = np.argsort(-top_values, axis=1)
        top_indices = np.take_along_axis(top_indices, order, axis=1)
        top_values = np.take_along_axis(top_values, order, axis=1)
        labels = target[top_indices]
        for k in k_values:
            effective_k = min(k, len(train_indices))
            for power in powers:
                if power == 0:
                    weighted = labels[:, :effective_k].mean(axis=1)
                else:
                    weights = np.maximum(top_values[:, :effective_k], 1e-8) ** power
                    weighted = (weights * labels[:, :effective_k]).sum(axis=1) / weights.sum(axis=1)
                result[(k, power)][query_indices] = weighted
    return result


def fit(args: argparse.Namespace) -> None:
    train_rows = list(read_jsonl(args.train))
    dev_rows = list(read_jsonl(args.dev_gold))
    ensemble = {row["id"]: row for row in _ensemble_rows(sorted(args.dev_predictions.glob("reproduction_*.jsonl")))}
    if {str(row["id"]) for row in dev_rows} != set(ensemble):
        raise SystemExit("development IDs and frozen prediction IDs differ")
    truth = np.asarray([int(row["labels"]["code_switching"]) for row in dev_rows])
    base = np.asarray([float(ensemble[str(row["id"])]["probabilities"]["code_switching"]) for row in dev_rows])
    best = (-1.0, None, None, None, None, None)
    for rep in REPRESENTATIONS:
        retrieval = scores(train_rows, dev_rows, rep, K_VALUES, POWERS)
        for (k, power), values in retrieval.items():
            for alpha in ALPHAS:
                outcome = metric_grid(truth, alpha * values + (1 - alpha) * base)
                position = int(outcome.argmax())
                candidate = (float(outcome[position]), rep, k, power, float(alpha), float(np.arange(1, 100)[position] / 100))
                if candidate[0] > best[0] + 1e-12:
                    best = candidate
    payload = {
        "method": "development_selected_train_only_lexical_neighbour_retrieval_blended_with_frozen_probability",
        "label": "code_switching", "selection_split": "development", "train_path": str(args.train),
        "representation": best[1], "k": best[2], "similarity_power": best[3], "alpha_retrieval": best[4], "threshold": best[5],
        "development_binary_macro_f1": round(best[0], 4),
        "search": {"representations": REPRESENTATIONS, "k_values": K_VALUES, "powers": POWERS},
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    train_rows = list(read_jsonl(args.train)); data_rows = list(read_jsonl(args.data))
    retrieval = scores(train_rows, data_rows, config["representation"], [config["k"]], [config["similarity_power"]])[(config["k"], config["similarity_power"])]
    index = {str(row["id"]): position for position, row in enumerate(data_rows)}
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    if set(index) != {row["id"] for row in rows}:
        raise SystemExit("input IDs and frozen prediction IDs differ")
    for row in rows:
        value = config["alpha_retrieval"] * retrieval[index[row["id"]]] + (1 - config["alpha_retrieval"]) * float(row["probabilities"]["code_switching"])
        row["probabilities"]["code_switching_text_retrieval_blend"] = float(value)
        row["predictions"]["code_switching"] = int(value >= config["threshold"])
        row["system"] = "vipragsent_frozen_code_switch_text_retrieval"
    write_jsonl(args.output, rows)
    print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--train", type=Path, required=True); fit_parser.add_argument("--dev-gold", type=Path, required=True)
    fit_parser.add_argument("--dev-predictions", type=Path, required=True); fit_parser.add_argument("--output", type=Path, required=True); fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--train", type=Path, required=True); apply_parser.add_argument("--data", type=Path, required=True)
    apply_parser.add_argument("--predictions", type=Path, required=True); apply_parser.add_argument("--base-config", type=Path, required=True)
    apply_parser.add_argument("--config", type=Path, required=True); apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
