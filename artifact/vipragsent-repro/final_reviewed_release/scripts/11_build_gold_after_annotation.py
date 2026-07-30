from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.annotation.gold_builder import build_gold_after_annotation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build final adjudicated ViPragSent train/dev/test JSONL after human annotation."
    )
    parser.add_argument("--batches-dir", default=str(ROOT / "data" / "annotation" / "batches"))
    parser.add_argument("--reviewer-1-dir", default=str(ROOT / "data" / "annotation" / "reviewer_01"))
    parser.add_argument("--reviewer-2-dir", default=str(ROOT / "data" / "annotation" / "reviewer_02"))
    parser.add_argument("--adjudicated-dir", default=str(ROOT / "data" / "annotation" / "adjudicated"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--disagreements-output", default=str(ROOT / "data" / "annotation" / "disagreements" / "all_disagreements.jsonl"))
    parser.add_argument("--report-output", default=str(ROOT / "data" / "manifest" / "gold_build_report.json"))
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    report = build_gold_after_annotation(
        batches_dir=args.batches_dir,
        reviewer_1_dir=args.reviewer_1_dir,
        reviewer_2_dir=args.reviewer_2_dir,
        adjudicated_dir=args.adjudicated_dir,
        output_dir=args.output_dir,
        disagreements_output=args.disagreements_output,
        report_output=args.report_output,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] in {"ok", "needs_attention", "partial_needs_adjudication", "pending_missing_human_annotations"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
