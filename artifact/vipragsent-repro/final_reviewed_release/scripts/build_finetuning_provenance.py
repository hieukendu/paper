from __future__ import annotations

"""Collect paper-ready provenance for flexible ViPragSent optimization runs."""

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def gpu_info() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> int:
    output = ROOT / "answers/optimized_vipragsent/flexible_finetuning"
    reports = ROOT / "reports"
    output.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    script_paths = [
        ROOT / "scripts/train_multitask_encoder.py",
        ROOT / "scripts/train_qlora_multilabel.py",
        ROOT / "scripts/train_tfidf_multilabel.py",
        ROOT / "scripts/frozen_threshold_ensemble.py",
        ROOT / "scripts/frozen_label_selector.py",
    ]
    runs = []
    for history_path in sorted((ROOT / "outputs/flexible_optimization").glob("*/*/history.json")):
        run_dir = history_path.parent
        history = json.loads(history_path.read_text(encoding="utf-8"))
        manifest_path = run_dir / "run_manifest.json"
        selection_path = run_dir / "selection.json"
        runs.append(
            {
                "system": run_dir.parent.name,
                "seed": run_dir.name,
                "history_path": str(history_path.relative_to(ROOT)),
                "history_sha256": sha256(history_path),
                "epochs_logged": len(history),
                "best_logged_dev_macro": max(
                    (row.get("development_macro_pragmatic_f1", row.get("dev_macro_pragmatic_f1", -1.0)) for row in history),
                    default=None,
                ),
                "run_manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None,
                "selection": json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None,
            }
        )
    metrics = {}
    for path in sorted((ROOT / "answers/optimized_vipragsent/flexible_metrics").glob("*_dev.csv")):
        metrics[path.name] = read_metrics(path)
    test_metrics = {}
    for path in sorted((ROOT / "answers/optimized_vipragsent/flexible_metrics").glob("*_test.csv")):
        test_metrics[path.name] = {"sha256": sha256(path), "metrics": read_metrics(path)}
    test_predictions = {}
    for path in sorted((ROOT / "answers/optimized_vipragsent/flexible_test_predictions").glob("*.json")):
        test_predictions[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "records": sum(1 for _ in path.open(encoding="utf-8"))}
    thresholds = {}
    for path in sorted((ROOT / "answers/optimized_vipragsent/flexible_configs").glob("*_thresholds.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        thresholds[path.name] = {
            "sha256": sha256(path),
            "selection_split": payload.get("selection_split"),
            "thresholds": payload.get("thresholds"),
            "development_binary_macro_f1": payload.get("development_binary_macro_f1"),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "training_labels": "vipragsent_train.jsonl only",
            "model_selection": "vipragsent_dev.jsonl only",
            "test_labels_used_for_selection": False,
            "evaluation_status": "development-only until a single preregistered final test run",
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "gpu": gpu_info()},
        "scripts": {str(path.relative_to(ROOT)): sha256(path) for path in script_paths if path.exists()},
        "runs": runs,
        "development_metrics": metrics,
        "final_test_metrics": test_metrics,
        "final_test_predictions": test_predictions,
        "threshold_configs": thresholds,
    }
    registry = output / "run_registry.json"
    registry.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Fine-tuning experiment provenance", "", "This report is generated from the machine-readable registry and is suitable as a paper appendix/run log.", "", "## Protocol", "", "- Training labels: `vipragsent_train.jsonl` only.", "- Architecture/threshold/source selection: `vipragsent_dev.jsonl` only.", "- Test labels have not been used in this optimization phase.", "", "## Runs", "", "| System | Seed | Epochs logged | Best logged dev macro-F1 |", "| --- | ---: | ---: | ---: |"]
    for run in runs:
        value = run["best_logged_dev_macro"]
        lines.append(f"| {run['system']} | {run['seed']} | {run['epochs_logged']} | {value:.4f} |" if value is not None else f"| {run['system']} | {run['seed']} | {run['epochs_logged']} | N/A |")
    lines.extend(["", f"Machine-readable registry: `{registry.relative_to(ROOT)}`."])
    (reports / "FINE_TUNING_EXPERIMENT_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "registry": str(registry), "runs": len(runs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
