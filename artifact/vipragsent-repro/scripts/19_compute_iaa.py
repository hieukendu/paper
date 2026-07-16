from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.annotation.agreement import cohen_kappa_nominal, krippendorff_alpha_nominal
from vipragsent.annotation.disagreement import LABEL_FIELDS
from vipragsent.data.schema import canonicalize_labels
from vipragsent.utils.io import read_jsonl


def load_folder(path: Path) -> dict[str, dict]:
    records = {}
    for source in sorted(path.glob("*.jsonl")):
        for row in read_jsonl(source):
            records[str(row["id"])] = canonicalize_labels(row.get("labels"))
    return records


def main() -> int:
    left = load_folder(ROOT / "data/annotation/reviewer_01")
    right = load_folder(ROOT / "data/annotation/reviewer_02")
    shared = sorted(set(left) & set(right))
    if not shared:
        raise SystemExit("No shared reviewer records; IAA cannot be computed.")
    fields = {}
    for field in LABEL_FIELDS:
        left_values = [left[item][field] for item in shared]
        right_values = [right[item][field] for item in shared]
        fields[field] = {
            "n": len(shared),
            "percent_agreement": sum(a == b for a, b in zip(left_values, right_values, strict=True)) / len(shared),
            "cohen_kappa": cohen_kappa_nominal(left_values, right_values),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal([[a, b] for a, b in zip(left_values, right_values, strict=True)]),
        }
    report = {
        "status": "complete",
        "reviewers": ["reviewer_01", "reviewer_02"],
        "shared_records": len(shared),
        "method": "Nominal Cohen's kappa and Krippendorff's alpha; no ordinal weighting.",
        "fields": fields,
        "macro": {
            "percent_agreement": sum(row["percent_agreement"] for row in fields.values()) / len(fields),
            "cohen_kappa": sum(row["cohen_kappa"] for row in fields.values()) / len(fields),
            "krippendorff_alpha_nominal": sum(row["krippendorff_alpha_nominal"] for row in fields.values()) / len(fields),
        },
    }
    output = ROOT / "results/annotation_agreement.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["# Inter-annotator agreement", "", "| Label | N | Agreement | Cohen's kappa | Krippendorff's alpha |", "| --- | ---: | ---: | ---: | ---: |"]
    for field, row in fields.items():
        table.append(f"| {field} | {row['n']} | {row['percent_agreement']:.3f} | {row['cohen_kappa']:.3f} | {row['krippendorff_alpha_nominal']:.3f} |")
    table.append(f"| **Macro** | {len(shared)} | {report['macro']['percent_agreement']:.3f} | {report['macro']['cohen_kappa']:.3f} | {report['macro']['krippendorff_alpha_nominal']:.3f} |")
    (ROOT / "tables/annotation_agreement.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "shared_records": len(shared)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
