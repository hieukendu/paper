from __future__ import annotations

"""Prepare immutable public-test JSONL files used by the real experiment runs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.experiments.constants import LOW_RESOURCE_BUDGETS
from vipragsent.utils.io import read_jsonl, write_jsonl


DATASETS = ("uit_vsfc", "uit_vsmec", "aivivn_2019")


def main() -> int:
    public_source = ROOT / "data" / "processed" / "all_unified.jsonl"
    output_dir = ROOT / "data" / "processed" / "public_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(public_source))
    summary: dict[str, object] = {"status": "ok", "public_source": str(public_source), "datasets": {}, "low_resource_budgets": LOW_RESOURCE_BUDGETS}
    for dataset in DATASETS:
        subset = [
            row
            for row in rows
            if (row.get("source") or {}).get("dataset") == dataset and (row.get("type") == "test" or row.get("split") == "test")
        ]
        path = output_dir / f"{dataset}_test.jsonl"
        write_jsonl(path, subset)
        summary["datasets"][dataset] = {"records": len(subset), "output": str(path)}
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
