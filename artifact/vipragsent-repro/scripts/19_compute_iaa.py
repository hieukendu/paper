from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.annotation.agreement import cohen_kappa_nominal, fleiss_kappa_nominal, krippendorff_alpha_nominal
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
    raters = {
        "reviewer_01": load_folder(ROOT / "data/annotation/reviewer_01"),
        "reviewer_02": load_folder(ROOT / "data/annotation/reviewer_02"),
        "adjudicator_quynh_nhu": load_folder(ROOT / "data/annotation/adjudicated"),
    }
    shared = sorted(set.intersection(*(set(records) for records in raters.values())))
    if not shared:
        raise SystemExit("No shared records across all raters; IAA cannot be computed.")
    rater_names = list(raters)
    pair_names = list(combinations(rater_names, 2))
    fields = {}
    for field in LABEL_FIELDS:
        values = {name: [records[item][field] for item in shared] for name, records in raters.items()}
        ratings = [[values[name][index] for name in rater_names] for index in range(len(shared))]
        pairwise_agreement = {
            f"{left}__{right}": sum(a == b for a, b in zip(values[left], values[right], strict=True)) / len(shared)
            for left, right in pair_names
        }
        pairwise_kappa = {
            f"{left}__{right}": cohen_kappa_nominal(values[left], values[right])
            for left, right in pair_names
        }
        fields[field] = {
            "n": len(shared),
            "mean_pairwise_percent_agreement": sum(pairwise_agreement.values()) / len(pairwise_agreement),
            "pairwise_percent_agreement": pairwise_agreement,
            "mean_pairwise_cohen_kappa": sum(pairwise_kappa.values()) / len(pairwise_kappa),
            "pairwise_cohen_kappa": pairwise_kappa,
            "fleiss_kappa_nominal": fleiss_kappa_nominal(ratings),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(ratings),
        }
    report = {
        "status": "complete",
        "reviewers": rater_names,
        "shared_records": len(shared),
        "method": "Nominal mean pairwise agreement, pairwise Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha across three raters; no ordinal weighting.",
        "protocol_note": "The third rater is the fixed adjudicator whose labels define the experiment gold split. Describe this as post-adjudication agreement unless independent blinded re-annotation is documented.",
        "fields": fields,
        "macro": {
            "mean_pairwise_percent_agreement": sum(row["mean_pairwise_percent_agreement"] for row in fields.values()) / len(fields),
            "mean_pairwise_cohen_kappa": sum(row["mean_pairwise_cohen_kappa"] for row in fields.values()) / len(fields),
            "fleiss_kappa_nominal": sum(row["fleiss_kappa_nominal"] for row in fields.values()) / len(fields),
            "krippendorff_alpha_nominal": sum(row["krippendorff_alpha_nominal"] for row in fields.values()) / len(fields),
        },
    }
    output = ROOT / "results/annotation_agreement.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Inter-annotator agreement",
        "",
        "Three-rater nominal agreement: reviewer 01, reviewer 02, and the fixed Quynh Nhu adjudicator.",
        "",
        "| Label | N | Mean pairwise agreement | Mean pairwise Cohen's kappa | Fleiss' kappa | Krippendorff's alpha |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, row in fields.items():
        table.append(f"| {field} | {row['n']} | {row['mean_pairwise_percent_agreement']:.3f} | {row['mean_pairwise_cohen_kappa']:.3f} | {row['fleiss_kappa_nominal']:.3f} | {row['krippendorff_alpha_nominal']:.3f} |")
    table.append(f"| **Macro** | {len(shared)} | {report['macro']['mean_pairwise_percent_agreement']:.3f} | {report['macro']['mean_pairwise_cohen_kappa']:.3f} | {report['macro']['fleiss_kappa_nominal']:.3f} | {report['macro']['krippendorff_alpha_nominal']:.3f} |")
    table.extend(["", report["protocol_note"]])
    (ROOT / "tables/annotation_agreement.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "shared_records": len(shared)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
