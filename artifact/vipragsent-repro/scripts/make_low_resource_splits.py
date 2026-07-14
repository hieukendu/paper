from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.data.schema import PRAGMATIC_LABELS, canonicalize_labels
from vipragsent.experiments.constants import LOW_RESOURCE_BUDGETS
from vipragsent.utils.io import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic low-resource sarcasm train subsets.")
    parser.add_argument("--train", required=True, help="Adjudicated train JSONL.")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed" / "low_resource"))
    parser.add_argument("--positive-label", default="sarcasm", choices=PRAGMATIC_LABELS)
    parser.add_argument("--budgets", nargs="*", type=int, default=LOW_RESOURCE_BUDGETS)
    parser.add_argument("--negative-ratio", type=float, default=3.0, help="Negatives per positive.")
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    records = list(read_jsonl(args.train))
    positives = [record for record in records if canonicalize_labels(record.get("labels")).get(args.positive_label) in (1, True)]
    negatives = [record for record in records if canonicalize_labels(record.get("labels")).get(args.positive_label) in (0, False)]
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "ok", "outputs": [], "positive_label": args.positive_label}
    for budget in args.budgets:
        if budget > len(positives):
            summary["outputs"].append({"budget": budget, "status": "blocked_not_enough_positives", "available": len(positives)})
            continue
        pos = rng.sample(positives, budget)
        neg_count = min(len(negatives), int(round(budget * args.negative_ratio)))
        neg = rng.sample(negatives, neg_count)
        subset = [*pos, *neg]
        rng.shuffle(subset)
        path = output_dir / f"{args.positive_label}_{budget}.jsonl"
        write_jsonl(path, subset)
        summary["outputs"].append({"budget": budget, "status": "ok", "output": str(path), "records": len(subset), "positives": budget, "negatives": neg_count})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
