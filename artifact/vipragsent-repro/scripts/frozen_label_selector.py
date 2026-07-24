#!/usr/bin/env python3
"""Choose a frozen prediction source per label on development data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl


def parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    return name, Path(raw_path)


def load(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def fit(args: argparse.Namespace) -> None:
    gold = {str(row["id"]): row["labels"] for row in read_jsonl(args.gold)}
    candidates = dict(args.candidate)
    predictions = {name: load(path) for name, path in candidates.items()}
    if any(set(rows) != set(gold) for rows in predictions.values()):
        raise SystemExit("candidate prediction IDs must exactly match development gold IDs")
    selected, scores = {}, {}
    for label in PRAGMATIC_LABELS:
        truth = [int(gold[record_id][label]) for record_id in sorted(gold)]
        label_scores = {
            name: binary_macro_f1(truth, [int(rows[record_id]["predictions"][label]) for record_id in sorted(gold)]) * 100
            for name, rows in predictions.items()
        }
        # Candidate order is an explicit deterministic complexity tiebreak.
        chosen = max(candidates, key=lambda name: label_scores[name])
        selected[label] = chosen
        scores[label] = {name: round(score, 4) for name, score in label_scores.items()}
    payload = {
        "method": "development_only_per_label_candidate_selection",
        "selection_split": "development",
        "gold_path": str(args.gold),
        "candidates": {name: str(path) for name, path in candidates.items()},
        "selected_source": selected,
        "development_binary_macro_f1": scores,
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = dict(args.candidate)
    candidates = {name: load(path) for name, path in inputs.items()}
    wanted = set(config["selected_source"].values())
    if wanted - set(candidates):
        raise SystemExit(f"missing selected candidate(s): {sorted(wanted - set(candidates))}")
    reference = candidates[next(iter(wanted))]
    if any(set(rows) != set(reference) for rows in candidates.values()):
        raise SystemExit("candidate prediction IDs differ")
    output = []
    for record_id in sorted(reference):
        row = json.loads(json.dumps(reference[record_id]))
        for label, source in config["selected_source"].items():
            row["predictions"][label] = candidates[source][record_id]["predictions"][label]
            row.setdefault("selected_source", {})[label] = source
        row["system"] = "vipragsent_frozen_dev_selected_hybrid"
        output.append(row)
    write_jsonl(args.output, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--gold", type=Path, required=True)
    fit_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--config", type=Path, required=True)
    apply_parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
