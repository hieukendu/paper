#!/usr/bin/env python3
"""Development-selected train-lexicon rules for the frozen code-switch label.

Each candidate is a deterministic statistic of English-looking tokens observed
in the labelled training split.  Development labels choose only a rule,
blending coefficient, and threshold; all test labels remain unread.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.utils.io import read_jsonl, write_jsonl

TOKEN_RE = re.compile(r"(?<![\w])([A-Za-z]{2,})(?![\w])")
MIN_COUNTS = [1, 2, 3, 5, 10]
SMOOTHING = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
MODES = ["max", "mean", "top2_mean", "noisy_or", "log_odds_sum"]
ALPHAS = np.arange(0, 11, dtype=np.float64) / 10
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def make_lexicon(path: Path) -> tuple[Counter, Counter, float]:
    positive, total, records = Counter(), Counter(), 0
    for row in read_jsonl(path):
        records += 1
        target = int(row["labels"]["code_switching"])
        for token in tokens(str(row["text"])):
            total[token] += 1
            positive[token] += target
    prior = sum(positive.values()) / sum(total.values()) if total else 0.0
    return positive, total, prior


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-50.0, min(50.0, value))))


def lexical_score(text: str, positive: Counter, total: Counter, prior: float, *, mode: str, min_count: int, smoothing: float) -> float:
    probabilities = []
    for token in tokens(text):
        count = total[token]
        if count >= min_count:
            denominator = count + smoothing
            probability = (positive[token] + smoothing * prior) / denominator if denominator else 0.0
            probabilities.append(min(1.0, max(0.0, probability)))
    if not probabilities:
        return 0.0
    probabilities.sort(reverse=True)
    if mode == "max":
        return probabilities[0]
    if mode == "mean":
        return float(np.mean(probabilities))
    if mode == "top2_mean":
        return float(np.mean(probabilities[:2]))
    if mode == "noisy_or":
        return float(1.0 - np.prod([1 - value for value in probabilities]))
    if mode == "log_odds_sum":
        base_log_odds = math.log(max(prior, 1e-6) / max(1 - prior, 1e-6))
        evidence = sum(math.log(max(value, 1e-6) / max(1 - value, 1e-6)) - base_log_odds for value in probabilities)
        return sigmoid(base_log_odds + evidence)
    raise ValueError(f"unknown mode: {mode}")


def metric_grid(truth: np.ndarray, scores: np.ndarray) -> np.ndarray:
    prediction = scores[:, None] >= THRESHOLDS[None, :]
    true = truth[:, None].astype(bool)
    tp = np.sum(prediction & true, axis=0); fp = np.sum(prediction & ~true, axis=0)
    fn = np.sum(~prediction & true, axis=0); tn = np.sum(~prediction & ~true, axis=0)
    pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    return (pos + neg) * 50


def rule_candidates(texts: list[str], positive: Counter, total: Counter, prior: float):
    for mode in MODES:
        for min_count in MIN_COUNTS:
            for smoothing in SMOOTHING:
                values = np.asarray([lexical_score(text, positive, total, prior, mode=mode, min_count=min_count, smoothing=smoothing) for text in texts])
                yield {"mode": mode, "min_count": min_count, "smoothing": smoothing}, values


def fit(args: argparse.Namespace) -> None:
    positive, total, prior = make_lexicon(args.train)
    ensemble = _ensemble_rows(sorted(args.dev_predictions.glob("reproduction_*.jsonl")))
    by_id = {str(row["id"]): row for row in read_jsonl(args.dev_gold)}
    if set(by_id) != {row["id"] for row in ensemble}:
        raise SystemExit("development IDs and frozen prediction IDs differ")
    texts = [str(by_id[row["id"]]["text"]) for row in ensemble]
    truth = np.asarray([int(by_id[row["id"]]["labels"]["code_switching"]) for row in ensemble])
    base = np.asarray([float(row["probabilities"]["code_switching"]) for row in ensemble])
    best = (-1.0, None, None, None)
    for rule, lexical in rule_candidates(texts, positive, total, prior):
        for alpha in ALPHAS:
            outcome = metric_grid(truth, alpha * lexical + (1 - alpha) * base)
            threshold_index = int(outcome.argmax())
            candidate = (float(outcome[threshold_index]), rule, float(alpha), float(THRESHOLDS[threshold_index]))
            if candidate[0] > best[0] + 1e-12:
                best = candidate
    payload = {
        "method": "development_selected_train_token_posterior_rule_blended_with_frozen_probability",
        "label": "code_switching", "selection_split": "development",
        "train_path": str(args.train), "dev_gold_path": str(args.dev_gold),
        "rule": best[1], "alpha_lexical": best[2], "threshold": best[3],
        "development_binary_macro_f1": round(best[0], 4),
        "search": {"modes": MODES, "min_counts": MIN_COUNTS, "smoothing": SMOOTHING},
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.rule.read_text(encoding="utf-8"))
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    positive, total, prior = make_lexicon(args.train)
    texts = {str(row["id"]): str(row["text"]) for row in read_jsonl(args.data)}
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    if set(texts) != {row["id"] for row in rows}:
        raise SystemExit("input IDs and frozen prediction IDs differ")
    for row in rows:
        lexical = lexical_score(texts[row["id"]], positive, total, prior, **config["rule"])
        score = config["alpha_lexical"] * lexical + (1 - config["alpha_lexical"]) * float(row["probabilities"]["code_switching"])
        row["probabilities"]["code_switching_lexical_rules_blend"] = score
        row["predictions"]["code_switching"] = int(score >= config["threshold"])
        row["system"] = "vipragsent_frozen_code_switch_rules"
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
    apply_parser.add_argument("--rule", type=Path, required=True); apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
