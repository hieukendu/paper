#!/usr/bin/env python3
"""Development-selected lexical retrieval for frozen ViPragSent code switching.

The lexicon is a non-parametric train-set statistic.  No neural model is
loaded, trained, or modified; the held-out test labels are never read.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl


TOKEN_RE = re.compile(r"(?<![\w])([A-Za-z]{2,})(?![\w])")
GRID = [round(value / 100, 2) for value in range(1, 100)]


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _lexicon(train: Path) -> tuple[Counter, Counter]:
    positive, total = Counter(), Counter()
    for row in read_jsonl(train):
        for token in _token_set(str(row["text"])):
            total[token] += 1
            positive[token] += int(row["labels"]["code_switching"])
    return positive, total


def _score(text: str, positive: Counter, total: Counter) -> float:
    values = [
        (positive[token] + 1) / (total[token] + 2)
        for token in _token_set(text)
        if total[token] >= 2
    ]
    return max(values, default=0.0)


def _fit(train: Path, dev_gold: Path, dev_predictions: Path) -> dict:
    positive, total = _lexicon(train)
    rows = _ensemble_rows(sorted(dev_predictions.glob("reproduction_*.jsonl")))
    texts = {str(row["id"]): str(row["text"]) for row in read_jsonl(dev_gold)}
    labels = {str(row["id"]): int(row["labels"]["code_switching"]) for row in read_jsonl(dev_gold)}
    if set(texts) != {row["id"] for row in rows}:
        raise ValueError("development IDs and prediction IDs differ")
    base = [float(row["probabilities"]["code_switching"]) for row in rows]
    retrieval = [_score(texts[row["id"]], positive, total) for row in rows]
    truth = [labels[row["id"]] for row in rows]
    best = (-1.0, None, None)
    for alpha_int in range(11):
        alpha = alpha_int / 10
        blended = [alpha * lexical + (1.0 - alpha) * model for lexical, model in zip(retrieval, base)]
        for threshold in GRID:
            value = binary_macro_f1(truth, [int(score >= threshold) for score in blended]) * 100
            if value > best[0]:
                best = (value, alpha, threshold)
    return {
        "method": "train_lexicon_max_score_blended_with_frozen_probability",
        "label": "code_switching",
        "selection_split": "development",
        "train_path": str(train),
        "dev_gold_path": str(dev_gold),
        "alpha_lexical": best[1],
        "threshold": best[2],
        "development_binary_macro_f1": round(best[0], 4),
        "token_pattern": TOKEN_RE.pattern,
        "smoothing": "(positive_count + 1) / (token_count + 2); token_count >= 2",
        "frozen_weight_compliance": {
            "neural_weight_updates": False,
            "selection_uses_test_labels": False,
            "optimizer_or_backward_called": False,
        },
    }


def fit(args: argparse.Namespace) -> None:
    payload = _fit(args.train, args.dev_gold, args.dev_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    positive, total = _lexicon(args.train)
    texts = {str(row["id"]): str(row["text"]) for row in read_jsonl(args.data)}
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    for row in rows:
        lexical = _score(texts[row["id"]], positive, total)
        model = float(row["probabilities"]["code_switching"])
        score = rule["alpha_lexical"] * lexical + (1.0 - rule["alpha_lexical"]) * model
        row["probabilities"]["code_switching_lexical_blend"] = score
        row["predictions"]["code_switching"] = int(score >= rule["threshold"])
    write_jsonl(args.output, rows)
    print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--train", type=Path, required=True)
    fit_parser.add_argument("--dev-gold", type=Path, required=True)
    fit_parser.add_argument("--dev-predictions", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--train", type=Path, required=True)
    apply_parser.add_argument("--data", type=Path, required=True)
    apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--base-config", type=Path, required=True)
    apply_parser.add_argument("--rule", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
