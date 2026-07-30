#!/usr/bin/env python3
"""Audit the three historical ViPragSent label gaps without tuning on test.

The test split is read only to characterize the already reported gap.  This
script neither writes model weights nor selects a replacement configuration.
All later model selection must use development-only results.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl


CURRENT = ROOT / "answers/optimized_vipragsent/flexible_test_predictions/encoder_lexical_hybrid_test.json"
GOLD = ROOT / "data/processed/vipragsent_test.jsonl"
TARGETS = {
    "irony": {
        "historical_best": "Sailor-7B SFT",
        "reported_score": 97.4132,
        "directory": ROOT / "results/predictions/main_pragmatic/sailor_7b_sft",
        "current_source": "ViSoBERT weighted encoder",
        "threshold": 0.59,
    },
    "idiom_figurative": {
        "historical_best": "PhoBERT fine-tune",
        "reported_score": 97.2958,
        "directory": ROOT / "results/predictions/main_pragmatic/phobert_finetune",
        "current_source": "PhoBERT weighted encoder",
        "threshold": 0.66,
    },
    "mocking": {
        "historical_best": "Vistral-7B SFT",
        "reported_score": 81.9802,
        "directory": ROOT / "results/predictions/main_pragmatic/vistral_7b_sft",
        "current_source": "ViSoBERT weighted encoder",
        "threshold": 0.65,
    },
}


def load_by_id(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def confusion(gold: list[int], prediction: list[int]) -> dict[str, int]:
    counts = Counter(zip(gold, prediction))
    return {
        "tn": counts[(0, 0)],
        "fp": counts[(0, 1)],
        "fn": counts[(1, 0)],
        "tp": counts[(1, 1)],
    }


def rate(value: int, denom: int) -> float | None:
    return None if not denom else round(value / denom, 4)


def score(gold: list[int], prediction: list[int]) -> dict[str, float | int | None]:
    matrix = confusion(gold, prediction)
    return {
        "binary_macro_f1": round(binary_macro_f1(gold, prediction) * 100, 4),
        **matrix,
        "positive_precision": rate(matrix["tp"], matrix["tp"] + matrix["fp"]),
        "positive_recall": rate(matrix["tp"], matrix["tp"] + matrix["fn"]),
        "negative_precision": rate(matrix["tn"], matrix["tn"] + matrix["fn"]),
        "negative_recall": rate(matrix["tn"], matrix["tn"] + matrix["fp"]),
    }


def co_occurrence(rows: list[dict], label: str, current: dict[str, dict]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for error_name, predicate in {
        "false_negative": lambda gold, pred: gold == 1 and pred == 0,
        "false_positive": lambda gold, pred: gold == 0 and pred == 1,
    }.items():
        selected = [row for row in rows if predicate(int(row["labels"][label]), int(current[row["id"]]["predictions"][label]))]
        output[error_name] = {
            "records": len(selected),
            **{
                companion: round(sum(int(row["labels"][companion]) for row in selected) / len(selected), 4)
                if selected else 0.0
                for companion in PRAGMATIC_LABELS
                if companion != label
            },
        }
    return output


def probability_slices(rows: list[dict], label: str, current: dict[str, dict]) -> dict[str, dict[str, float | int | None]]:
    groups = {"true_positive": [], "false_negative": [], "false_positive": [], "true_negative": []}
    for row in rows:
        value = current[row["id"]].get("probabilities", {}).get(label)
        if value is None:
            continue
        gold = int(row["labels"][label]); pred = int(current[row["id"]]["predictions"][label])
        group = {(1, 1): "true_positive", (1, 0): "false_negative", (0, 1): "false_positive", (0, 0): "true_negative"}[(gold, pred)]
        groups[group].append(float(value))
    return {
        group: {
            "records": len(values),
            "mean_probability": round(sum(values) / len(values), 4) if values else None,
            "min_probability": round(min(values), 4) if values else None,
            "max_probability": round(max(values), 4) if values else None,
        }
        for group, values in groups.items()
    }


def source_slices(rows: list[dict], label: str, current: dict[str, dict]) -> dict[str, dict[str, int]]:
    slices: dict[str, dict[str, int]] = {}
    for row in rows:
        source = row.get("source") or {}
        if isinstance(source, dict):
            name = str(source.get("dataset") or source.get("platform") or "unknown")
        else:
            name = str(source)
        bucket = slices.setdefault(name, {"records": 0, "fn": 0, "fp": 0})
        bucket["records"] += 1
        gold = int(row["labels"][label]); pred = int(current[row["id"]]["predictions"][label])
        if gold and not pred:
            bucket["fn"] += 1
        if not gold and pred:
            bucket["fp"] += 1
    return dict(sorted(slices.items()))


def main() -> int:
    gold_rows = list(read_jsonl(GOLD))
    current = load_by_id(CURRENT)
    if set(current) != {str(row["id"]) for row in gold_rows}:
        raise SystemExit("Current prediction IDs do not match gold test IDs")
    report: dict[str, object] = {
        "scope": "post-hoc diagnostic only; no test-driven model selection",
        "current_system": str(CURRENT),
        "test_gold": str(GOLD),
        "labels": {},
    }
    for label, details in TARGETS.items():
        gold = [int(row["labels"][label]) for row in gold_rows]
        current_pred = [int(current[str(row["id"])]["predictions"][label]) for row in gold_rows]
        baseline_runs = {}
        for file in sorted(details["directory"].glob("*.jsonl")):
            predictions = load_by_id(file)
            if set(predictions) != set(current):
                raise SystemExit(f"Baseline IDs do not match for {file}")
            values = [int(predictions[str(row["id"])]["predictions"][label]) for row in gold_rows]
            baseline_runs[file.name] = score(gold, values)
        report["labels"][label] = {
            "historical_best": details["historical_best"],
            "historical_reported_score": details["reported_score"],
            "current_source": details["current_source"],
            "current_dev_selected_threshold": details["threshold"],
            "prevalence": round(sum(gold) / len(gold), 4),
            "current": score(gold, current_pred),
            "historical_baseline_runs": baseline_runs,
            "current_probability_slices": probability_slices(gold_rows, label, current),
            "current_error_label_cooccurrence": co_occurrence(gold_rows, label, current),
            "current_error_source_slices": source_slices(gold_rows, label, current),
        }
    output = ROOT / "answer/label_gap_diagnosis/diagnosis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
