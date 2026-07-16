from __future__ import annotations

"""Create a compact hand-off folder containing only reproducible results."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


RESULTS = (
    "main_pragmatic.json",
    "ordinary_sentiment.json",
    "multitask_ablation.json",
    "low_resource_sarcasm.json",
    "calibration.json",
    "error_confusion.json",
    "learning_curves.json",
    "cost_breakdown.json",
    "artifact_index.json",
    "claim_ledger.csv",
    "annotation_agreement.json",
    "significance.json",
    "paper_readiness.json",
)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "answer"))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    copied: list[str] = []
    for source in (ROOT.parents[1] / "LICENSE", ROOT.parents[1] / "THIRD_PARTY_NOTICES.md"):
        if copy_if_exists(source, output / source.name):
            copied.append(source.name)
    for name in RESULTS:
        if copy_if_exists(ROOT / "results" / name, output / "results" / name):
            copied.append(f"results/{name}")
    for folder in ("tables", "figures"):
        for source in sorted((ROOT / folder).glob("*")):
            if source.is_file() and copy_if_exists(source, output / folder / source.name):
                copied.append(f"{folder}/{source.name}")
    for source in sorted((ROOT / "outputs").glob("**/run_manifest.json")):
        relative = source.relative_to(ROOT / "outputs")
        if copy_if_exists(source, output / "run_manifests" / relative):
            copied.append(f"run_manifests/{relative}")
    for source in [ROOT / "data/manifest/gold_build_report.json", ROOT / "data/manifest/annotation_import_report.json", ROOT / "data/manifest/rationale_audit_waiver.json", ROOT / "data/manifest/datasets.json", ROOT / "data/manifest/source_registry.json", ROOT / "data/manifest/checksums.json", ROOT / "data/processed/public_eval/manifest.json"]:
        if copy_if_exists(source, output / "data_provenance" / source.name):
            copied.append(f"data_provenance/{source.name}")
    status = "results_generated" if (ROOT / "results/main_pragmatic.json").exists() else "setup_complete_results_not_run"
    readme = [
        "# ViPragSent experiment hand-off",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Status: `{status}`",
        "",
        "All metrics in `results/`, tables, and figures are generated from prediction JSONL and trainer manifests in this run; no values are copied from `main.pdf`.",
        "",
        "Important protocol notes:",
        "",
        "- Gold split: 8,000 train / 2,000 dev / 2,000 test adjudicated records.",
        "- Rationale coverage: 8,000 train records; manual faithfulness audit was waived and is documented in data_provenance/.",
        "- Q3 budgets: 64, 128, 256, 512, and 1,024 sarcasm-positive examples. A 2,048 budget is infeasible with the actual train set.",
        "- UIT-VSMEC is reported as seven-way emotion macro-F1, not a fabricated polarity score.",
        "",
        "Copied files:",
        "",
        *[f"- `{item}`" for item in copied],
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "files": len(copied)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
