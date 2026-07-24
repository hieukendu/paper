#!/usr/bin/env python3
"""Blend a fixed train-lexicon code rule with frozen candidate scores on dev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_code_switch_rules import ALPHAS, lexical_score, make_lexicon, metric_grid
from vipragsent.utils.io import read_jsonl, write_jsonl


def parse_candidate(value: str) -> tuple[str, tuple[Path, str]]:
    name, separator, raw = value.partition("=")
    path, suffix_separator, suffix = raw.rpartition(":")
    if not separator or not name or not suffix_separator or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH:PROBABILITY_SUFFIX")
    return name, (Path(path), suffix)


def load(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def score(rows: dict[str, dict], suffix: str, record_ids: list[str]) -> np.ndarray:
    key = "code_switching" if suffix == "raw" else f"code_switching{suffix}"
    try:
        return np.asarray([float(rows[record_id]["probabilities"][key]) for record_id in record_ids])
    except KeyError as error:
        raise SystemExit(f"missing code-switch score field {key!r}") from error


def fit(args: argparse.Namespace) -> None:
    lexical_config = json.loads(args.lexical_rule.read_text(encoding="utf-8"))
    positive, total, prior = make_lexicon(args.train)
    gold = {str(row["id"]): row for row in read_jsonl(args.dev_gold)}
    record_ids = sorted(gold)
    inputs = dict(args.candidate)
    candidates = {name: (load(path), suffix) for name, (path, suffix) in inputs.items()}
    if any(set(rows) != set(gold) for rows, _ in candidates.values()):
        raise SystemExit("candidate prediction IDs must exactly match development gold IDs")
    lexical = np.asarray([lexical_score(str(gold[record_id]["text"]), positive, total, prior, **lexical_config["rule"]) for record_id in record_ids])
    truth = np.asarray([int(gold[record_id]["labels"]["code_switching"]) for record_id in record_ids])
    best = (-1.0, None, None, None)
    for name, (rows, suffix) in candidates.items():
        base = score(rows, suffix, record_ids)
        for alpha in ALPHAS:
            outcome = metric_grid(truth, alpha * lexical + (1 - alpha) * base)
            threshold_index = int(outcome.argmax())
            candidate = (float(outcome[threshold_index]), name, float(alpha), float(np.arange(1, 100, dtype=float)[threshold_index] / 100))
            if candidate[0] > best[0] + 1e-12:
                best = candidate
    payload = {
        "method": "development_only_lexicon_rule_blended_with_frozen_retrieval_score",
        "label": "code_switching", "selection_split": "development", "train_path": str(args.train),
        "lexical_rule": lexical_config, "candidates": {name: {"path": str(path), "probability_suffix": suffix} for name, (path, suffix) in inputs.items()},
        "base_candidate": best[1], "alpha_lexical": best[2], "threshold": best[3], "development_binary_macro_f1": round(best[0], 4),
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("base_candidate", "alpha_lexical", "threshold", "development_binary_macro_f1")}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    positive, total, prior = make_lexicon(args.train)
    data = {str(row["id"]): row for row in read_jsonl(args.data)}
    inputs = dict(args.candidate)
    if config["base_candidate"] not in inputs:
        raise SystemExit("the selected frozen candidate was not supplied")
    rows, suffix = inputs[config["base_candidate"]]
    predictions = load(rows)
    if set(predictions) != set(data):
        raise SystemExit("candidate prediction IDs differ from the requested split")
    output = []
    for record_id in sorted(data):
        row = json.loads(json.dumps(predictions[record_id]))
        lexical = lexical_score(str(data[record_id]["text"]), positive, total, prior, **config["lexical_rule"]["rule"])
        base = score(predictions, suffix, [record_id])[0]
        blended = config["alpha_lexical"] * lexical + (1 - config["alpha_lexical"]) * base
        row["probabilities"]["code_switching_lexical_retrieval_blend"] = float(blended)
        row["predictions"]["code_switching"] = int(blended >= config["threshold"])
        row["system"] = "vipragsent_frozen_code_switch_score_blend"
        output.append(row)
    write_jsonl(args.output, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--train", type=Path, required=True); fit_parser.add_argument("--dev-gold", type=Path, required=True)
    fit_parser.add_argument("--lexical-rule", type=Path, required=True); fit_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    fit_parser.add_argument("--output", type=Path, required=True); fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--train", type=Path, required=True); apply_parser.add_argument("--data", type=Path, required=True)
    apply_parser.add_argument("--config", type=Path, required=True); apply_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
