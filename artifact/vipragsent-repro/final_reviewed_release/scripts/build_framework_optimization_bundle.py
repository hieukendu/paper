#!/usr/bin/env python3
"""Package development-only ViPragSent framework ablations for audit.

This report intentionally does not create a new test-set result.  The test
split had already been inspected before this search, so all design and
threshold decisions in this bundle use the development split only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "implicit_sentiment",
    "sarcasm",
    "irony",
    "idiom_figurative",
    "code_switching",
    "mocking",
    "macro_pragmatic_f1",
]
CANDIDATES = [
    ("visobert_full_uncertainty", "ViSoBERT full multitask + uncertainty", 20260731),
    ("visobert_full_fixed_weights", "ViSoBERT full multitask, fixed loss weights", 20260732),
    ("visobert_full_mlp_focal", "ViSoBERT full multitask, MLP + focal", 20260733),
    ("phobert_full_fixed_weights", "PhoBERT full multitask, fixed loss weights", 20260734),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def path_info(path: Path, *, hash_file: bool = True) -> dict[str, object]:
    info: dict[str, object] = {"source": str(path), "exists": path.exists()}
    if path.exists():
        info["bytes"] = path.stat().st_size
        if hash_file:
            info["sha256"] = sha256(path)
    return info


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", *LABELS])
        writer.writeheader()
        writer.writerows(rows)


def add_readonly_alias(link: Path, target: Path) -> None:
    """Expose canonical artifacts without duplicating large checkpoints."""
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(target)


def main() -> int:
    answer = ROOT / "answers/framework_optimization"
    comparison = ROOT / "answer/comparison/framework_optimization_dev"
    matrix = ROOT / "configs/framework_optimization_dev_matrix.yaml"
    selected_metrics = answer / "selected/locked_plus_full_multitask_dev_metrics.csv"
    old_selection = ROOT / "answers/optimized_vipragsent/flexible_configs/encoder_lexical_hybrid_selection.json"
    new_selection = answer / "selection/locked_plus_full_multitask_dev_selection.json"

    rows: list[dict[str, object]] = []
    indexed: dict[str, object] = {"protocol": path_info(matrix)}
    for candidate, display, seed in CANDIDATES:
        metric_file = answer / f"dev_thresholded/{candidate}_metrics.csv"
        values = metrics(metric_file)
        rows.append({"system": display, **{label: f"{values[label]:.4f}" for label in LABELS}})
        run = ROOT / f"outputs/framework_optimization/{candidate}/{seed}"
        indexed[candidate] = {
            "run_manifest": path_info(run / "run_manifest.json"),
            "history": path_info(run / "history.json"),
            "checkpoint": path_info(run / "best.pt", hash_file=False),
            "development_raw_predictions": path_info(answer / f"dev_predictions/{candidate}/predictions.jsonl"),
            "threshold_config": path_info(answer / f"thresholds/{candidate}.json"),
            "development_thresholded_predictions": path_info(answer / f"dev_thresholded/{candidate}.jsonl"),
            "development_metrics": path_info(metric_file),
        }

    selected = metrics(selected_metrics)
    rows.append({"system": "Dev-selected hybrid: locked + full multitask mocking", **{label: f"{selected[label]:.4f}" for label in LABELS}})
    indexed["dev_selected_hybrid"] = {
        "previous_selection": path_info(old_selection),
        "new_selection": path_info(new_selection),
        "development_predictions": path_info(answer / "selected/locked_plus_full_multitask_dev.jsonl"),
        "development_metrics": path_info(selected_metrics),
    }

    status = {
        "scope": "development-only framework optimization",
        "test_set_policy": {
            "new_test_prediction_or_scoring": False,
            "test_labels_used_for_selection": False,
            "reason": "The historical test split was already evaluated before this search; it is not a fresh confirmatory split.",
        },
        "result": {
            "selected_system": "locked_plus_full_multitask_dev",
            "development_macro_pragmatic_f1": round(selected["macro_pragmatic_f1"], 4),
            "change_from_locked_dev_hybrid": "+0.0223 macro-F1; the full multitask candidate is selected only for mocking.",
        },
        "claims_not_established": [
            "Strict superiority over every historical best-tuned baseline on every metric.",
            "Strict per-label dominance over every deployment-default baseline.",
        ],
        "deployment_default_policy": "Unchanged. The existing fixed deployment-default protocol is retained rather than weakened or altered to obtain a desired ranking.",
        "manuscript_modified": False,
    }
    readme = """# ViPragSent framework-optimization addendum (development only)

This bundle records a predeclared four-run framework ablation over ViPragSent's rationale, polarity, emotion, loss-balancing, MLP-head, and focal-loss choices. It does **not** modify the manuscript.

## Outcome

The selected development-only hybrid reaches **{macro:.4f} Macro-F1**. It keeps the locked encoder hybrid for five labels and uses the full multitask ViSoBERT candidate only for `mocking` ({mocking:.4f} on dev, versus 80.5806 for the prior locked source).

This is not a new test claim: the historical test split was already observed before this search. Therefore neither all-metric superiority over historical best-tuned baselines nor per-label dominance over every deployment-default baseline is established here.

The deployment-default comparison is deliberately unchanged. Its profile remains a declared, fixed protocol; it was not weakened or adjusted to force a ranking.

## Calibrated development results

| System | Implicit | Sarcasm | Irony | Idiom | Code-switch | Mocking | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## Audit contents

- `dev_ablation_metrics.csv`: the calibrated results in the table.
- `status.json`: scope and claim-status guards.
- `artifact_index.json`: paths, sizes, and hashes for configuration, histories, threshold files, dev predictions, and metrics; checkpoint paths are indexed by size but not rehashed.
- The canonical generated predictions, threshold configurations, selection file, histories, manifests, and checkpoints remain under `answers/framework_optimization/` and `outputs/framework_optimization/`.

All hyperparameters were fixed in `configs/framework_optimization_dev_matrix.yaml` before these four runs. Training used the 8,000-record train split; threshold and source selection used the 2,000-record dev split only.
""".format(
        macro=selected["macro_pragmatic_f1"],
        mocking=selected["mocking"],
        table="\n".join(
            "| {system} | {implicit_sentiment} | {sarcasm} | {irony} | {idiom_figurative} | {code_switching} | {mocking} | {macro_pragmatic_f1} |".format(**row)
            for row in rows
        ),
    )

    for target in (answer, comparison):
        target.mkdir(parents=True, exist_ok=True)
        write_csv(target / "dev_ablation_metrics.csv", rows)
        (target / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "artifact_index.json").write_text(json.dumps(indexed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "README.md").write_text(readme, encoding="utf-8")
    add_readonly_alias(comparison / "canonical_runs", ROOT / "outputs/framework_optimization")
    add_readonly_alias(comparison / "canonical_dev_artifacts", answer)
    add_readonly_alias(comparison / "protocol_config", matrix)
    print(json.dumps({"status": "ok", "outputs": [str(answer), str(comparison)], "dev_macro": selected["macro_pragmatic_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
