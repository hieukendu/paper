from __future__ import annotations

from pathlib import Path

from vipragsent.artifacts.claim_ledger import pending_claim_rows, write_claim_ledger
from vipragsent.artifacts.result_schema import make_pending_result
from vipragsent.utils.io import dump_json, ensure_dir

PENDING_EXPERIMENTS = [
    "main_pragmatic",
    "ordinary_sentiment",
    "multitask_ablation",
    "low_resource_sarcasm",
    "error_confusion",
    "calibration",
    "cost_breakdown",
    "learning_curves",
]


def make_artifacts_dryrun(root: str | Path = ".") -> dict[str, object]:
    root = Path(root)
    pending_dir = ensure_dir(root / "results" / "pending")
    figures_dir = ensure_dir(root / "figures")
    tables_dir = ensure_dir(root / "tables")

    result_paths = []
    for experiment in PENDING_EXPERIMENTS:
        path = pending_dir / f"{experiment}.pending.json"
        dump_json(path, make_pending_result(experiment))
        result_paths.append(str(path))

    ledger_path = root / "results" / "claim_ledger.csv"
    ledger_rows = write_claim_ledger(ledger_path, pending_claim_rows())

    (tables_dir / "pending_summary.md").write_text(
        "# Pending Artifact Summary\n\nNo real experiment metrics have been generated in setup-only mode.\n",
        encoding="utf-8",
    )
    (figures_dir / "README.md").write_text(
        "# Figures\n\nFigure generation is pending real result JSON files.\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "pending_results": result_paths,
        "claim_ledger": str(ledger_path),
        "claim_rows": ledger_rows,
    }
