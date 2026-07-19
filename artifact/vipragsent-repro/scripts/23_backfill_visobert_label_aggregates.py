from __future__ import annotations

"""Backfill canonical three-seed ViSoBERT label aggregates from stored reports."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.experiments.evaluator import aggregate_seed_reports


METRICS = (
    "implicit_sentiment",
    "sarcasm",
    "irony",
    "idiom_figurative",
    "code_switching",
    "mocking",
    "macro_pragmatic_f1",
)


def update(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    system = payload["system"]
    system["aggregate"] = aggregate_seed_reports(system["per_seed"], list(METRICS))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    result = update(ROOT / "results" / "p0_visobert_baseline.json")
    summary_path = ROOT / "results" / "p0_p1_p2_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["p0_visobert_baseline"] = result
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"updated": ["results/p0_visobert_baseline.json", "results/p0_p1_p2_summary.json"], "metrics": list(result["system"]["aggregate"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
