#!/usr/bin/env python3
"""Development-only source-conditioned thresholds for frozen predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl

GRID = [round(value / 100, 2) for value in range(1, 100)]


def source_key(row: dict) -> str:
    return str((row.get("source") or {}).get("dataset") or "unknown")


def inputs(path: Path) -> list[dict]:
    files = sorted(path.glob("reproduction_*.jsonl"))
    if not files:
        raise ValueError(f"no reproduction predictions in {path}")
    return _ensemble_rows(files)


def fit(args: argparse.Namespace) -> None:
    gold = {str(row["id"]): row for row in read_jsonl(args.gold)}
    rows = inputs(args.predictions)
    if set(gold) != {row["id"] for row in rows}:
        raise SystemExit("gold/prediction IDs differ")
    grouped = {}
    for row in rows:
        grouped.setdefault(source_key(gold[row["id"]]), []).append(row)
    thresholds = {}
    for source, members in grouped.items():
        thresholds[source] = {}
        for label in PRAGMATIC_LABELS:
            truth = [int(gold[row["id"]]["labels"][label]) for row in members]
            scores = [float(row["probabilities"][label]) for row in members]
            value, threshold = max(
                (binary_macro_f1(truth, [int(score >= candidate) for score in scores]) * 100, candidate)
                for candidate in GRID
            )
            thresholds[source][label] = {"threshold": threshold, "development_binary_macro_f1": round(value, 4), "n": len(members)}
    payload = {"method": "development_only_source_conditioned_label_thresholds", "selection_split": "development", "gold_path": str(args.gold), "thresholds": thresholds, "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = {str(row["id"]): row for row in read_jsonl(args.data)}
    rows = inputs(args.predictions)
    if set(data) != {row["id"] for row in rows}:
        raise SystemExit("data/prediction IDs differ")
    for row in rows:
        group = source_key(data[row["id"]])
        for label in PRAGMATIC_LABELS:
            threshold = config["thresholds"][group][label]["threshold"]
            row["predictions"][label] = int(float(row["probabilities"][label]) >= threshold)
        row["system"] = "vipragsent_frozen_source_thresholds"
    write_jsonl(args.output, rows); print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit"); fit_parser.add_argument("--gold", type=Path, required=True); fit_parser.add_argument("--predictions", type=Path, required=True); fit_parser.add_argument("--output", type=Path, required=True); fit_parser.set_defaults(func=fit)
    apply_parser = commands.add_parser("apply"); apply_parser.add_argument("--config", type=Path, required=True); apply_parser.add_argument("--data", type=Path, required=True); apply_parser.add_argument("--predictions", type=Path, required=True); apply_parser.add_argument("--output", type=Path, required=True); apply_parser.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
