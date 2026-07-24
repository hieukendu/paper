from __future__ import annotations

"""Assemble the non-manuscript deployment comparison bundle under answer/."""

import csv
import hashlib
import json
import os
import shutil
import sys
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


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def copy_file(source: Path, destination: Path, index: list[dict]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    index.append(
        {
            "bundle_path": str(destination.relative_to(BUNDLE)),
            "source_path": str(source.relative_to(ROOT)),
            "sha256": file_hash(destination),
            "bytes": destination.stat().st_size,
            "kind": "copied_file",
        }
    )


def link_artifact(source: Path, destination: Path, index: list[dict]) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The comparison bundle remains compact while preserving a directly usable
    # pointer to each canonical trained weight artifact.
    relative = os.path.relpath(source, destination.parent)
    os.symlink(relative, destination, target_is_directory=source.is_dir())
    index.append(
        {
            "bundle_path": str(destination.relative_to(BUNDLE)),
            "source_path": str(source.relative_to(ROOT)),
            "kind": "symbolic_link_to_canonical_artifact",
            "is_directory": source.is_dir(),
        }
    )


def write_svg(rows: list[tuple[str, dict[str, float]]], output: Path) -> None:
    width, left, right, top, row_height = 960, 240, 80, 58, 58
    height = top + row_height * len(rows) + 58
    plot_width = width - left - right
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:700}</style>',
        '<text x="40" y="32" class="title">Macro-F1: ViPragSent vs deployment-default baselines</text>',
        '<text x="40" y="50" class="small">Fixed profiles; detailed conditions are in protocol/deployment_default_baselines.yaml.</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100
        pieces.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - 34}" stroke="#dbe3ef"/>')
        pieces.append(f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" class="small">{tick}</text>')
    for index, (name, metrics) in enumerate(rows):
        y = top + index * row_height
        value = metrics["macro_pragmatic_f1"]
        bar_width = plot_width * value / 100
        color = "#11845b" if name == "ViPragSent current" else "#58708f"
        pieces.append(f'<text x="{left - 14}" y="{y + 23}" text-anchor="end" class="label">{name}</text>')
        pieces.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="30" rx="4" fill="{color}"/>')
        pieces.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 22}" class="label">{value:.2f}</text>')
    pieces.append("</svg>")
    output.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    global BUNDLE
    BUNDLE = ROOT / "answer/comparison"
    if BUNDLE.exists():
        raise SystemExit(f"refusing to overwrite existing comparison bundle: {BUNDLE}")
    BUNDLE.mkdir(parents=True)
    index: list[dict] = []
    systems = {
        "ViPragSent current": {
            "metric": ROOT / "answers/optimized_vipragsent/flexible_metrics/encoder_lexical_hybrid_test.csv",
            "prediction": ROOT / "answers/optimized_vipragsent/flexible_test_predictions/encoder_lexical_hybrid_test.json",
            "kind": "locked_finetuned_hybrid",
            "run_dir": None,
        },
        "PhoBERT default": {
            "metric": ROOT / "answers/optimized_vipragsent/deployment_default_metrics/deployment_default_phobert_test.csv",
            "prediction": ROOT / "answers/optimized_vipragsent/deployment_default_predictions/deployment_default_phobert/20260730.jsonl",
            "kind": "deployment_default",
            "run_dir": ROOT / "outputs/deployment_default/deployment_default_phobert/20260730",
        },
        "ViSoBERT default": {
            "metric": ROOT / "answers/optimized_vipragsent/deployment_default_metrics/deployment_default_visobert_test.csv",
            "prediction": ROOT / "answers/optimized_vipragsent/deployment_default_predictions/deployment_default_visobert/20260730.jsonl",
            "kind": "deployment_default",
            "run_dir": ROOT / "outputs/deployment_default/deployment_default_visobert/20260730",
        },
        "XLM-R large default": {
            "metric": ROOT / "answers/optimized_vipragsent/deployment_default_metrics/deployment_default_xlmr_large_test.csv",
            "prediction": ROOT / "answers/optimized_vipragsent/deployment_default_predictions/deployment_default_xlmr_large/20260730.jsonl",
            "kind": "deployment_default",
            "run_dir": ROOT / "outputs/deployment_default/deployment_default_xlmr_large/20260730",
        },
        "Vistral-7B default": {
            "metric": ROOT / "answers/optimized_vipragsent/deployment_default_metrics/deployment_default_vistral_7b_test.csv",
            "prediction": ROOT / "answers/optimized_vipragsent/deployment_default_predictions/deployment_default_vistral_7b/20260730.jsonl",
            "kind": "deployment_default",
            "run_dir": ROOT / "outputs/deployment_default/deployment_default_vistral_7b/20260730",
        },
        "Sailor-7B default": {
            "metric": ROOT / "answers/optimized_vipragsent/deployment_default_metrics/deployment_default_sailor_7b_test.csv",
            "prediction": ROOT / "answers/optimized_vipragsent/deployment_default_predictions/deployment_default_sailor_7b/20260730.jsonl",
            "kind": "deployment_default",
            "run_dir": ROOT / "outputs/deployment_default/deployment_default_sailor_7b/20260730",
        },
    }
    metrics = {name: load_metrics(value["metric"]) for name, value in systems.items()}
    copy_file(ROOT / "configs/deployment_default_baselines.yaml", BUNDLE / "protocol/deployment_default_baselines.yaml", index)
    copy_file(ROOT / "data/processed/deployment_default_train_512.jsonl.metadata.json", BUNDLE / "protocol/deployment_default_train_512.metadata.json", index)
    for source in [
        ROOT / "answers/optimized_vipragsent/flexible_configs/encoder_lexical_hybrid_selection.json",
        ROOT / "answers/optimized_vipragsent/flexible_configs/visobert_weighted_cls_s1_thresholds.json",
        ROOT / "answers/optimized_vipragsent/flexible_configs/phobert_weighted_cls_s2_thresholds.json",
        ROOT / "answers/optimized_vipragsent/flexible_finetuning/run_registry.json",
    ]:
        copy_file(source, BUNDLE / "vipragsent_current/provenance" / source.name, index)
    for run_name, seed in [
        ("finetune_visobert_weighted_cls_s1", "20260726"),
        ("finetune_phobert_weighted_cls_s2", "20260725"),
    ]:
        run_dir = ROOT / "outputs/flexible_optimization" / run_name / seed
        target = BUNDLE / "vipragsent_current/provenance/runs" / run_name
        for filename in ["history.json", "run_manifest.json"]:
            copy_file(run_dir / filename, target / filename, index)
        link_artifact(run_dir / "best.pt", target / "canonical_weights/best.pt", index)
    for name, value in systems.items():
        slug = name.lower().replace(" ", "_").replace("-", "_")
        copy_file(value["metric"], BUNDLE / "systems" / slug / "metrics_test.csv", index)
        copy_file(value["prediction"], BUNDLE / "systems" / slug / "predictions_test.jsonl", index)
        run_dir = value["run_dir"]
        if run_dir:
            for filename in ["history.json", "run_manifest.json", "selection.json"]:
                source = run_dir / filename
                if source.is_file():
                    copy_file(source, BUNDLE / "systems" / slug / "run" / filename, index)
            for filename in ["best.pt", "classification_head.pt"]:
                source = run_dir / filename
                if source.is_file():
                    link_artifact(source, BUNDLE / "systems" / slug / "canonical_weights" / filename, index)
            if (run_dir / "adapter").is_dir():
                link_artifact(run_dir / "adapter", BUNDLE / "systems" / slug / "canonical_weights" / "adapter", index)
    summary_path = BUNDLE / "comparison_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "regime", *LABELS])
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"system": name, "regime": systems[name]["kind"], **{label: f"{values[label]:.4f}" for label in LABELS}})
    index.append({"bundle_path": "comparison_metrics.csv", "kind": "generated_summary", "sha256": file_hash(summary_path), "bytes": summary_path.stat().st_size})
    rows = sorted(metrics.items(), key=lambda item: item[1]["macro_pragmatic_f1"], reverse=True)
    write_svg(rows, BUNDLE / "macro_f1_comparison.svg")
    index.append({"bundle_path": "macro_f1_comparison.svg", "kind": "generated_visualization", "sha256": file_hash(BUNDLE / "macro_f1_comparison.svg"), "bytes": (BUNDLE / "macro_f1_comparison.svg").stat().st_size})
    table_header = "| System | Implicit | Sarcasm | Irony | Idiom | Code-switch | Mocking | Macro-F1 |"
    lines = [
        "# ViPragSent comparison bundle",
        "",
        "This directory is a comparison artifact; it does not edit or replace the manuscript.",
        "",
        "The green system in `macro_f1_comparison.svg` is the locked current ViPragSent fine-tuned hybrid. The other rows are newly executed **deployment-default** runs. They use the fixed, non-searched profile in `protocol/deployment_default_baselines.yaml`; they are not replacements for historical best-tuned baseline upper bounds.",
        "",
        "## Test-set comparison",
        "",
        table_header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in rows:
        lines.append("| " + name + " | " + " | ".join(f"{values[label]:.4f}" for label in LABELS) + " |")
    lines.extend([
        "",
        "## Contents",
        "",
        "- `comparison_metrics.csv`: every metric for every system.",
        "- `macro_f1_comparison.svg`: visual macro-F1 comparison.",
        "- `systems/`: test predictions, metrics, run history, configuration manifests, and symbolic links to canonical trained weights/adapters.",
        "- `vipragsent_current/`: selected-hybrid provenance and locked current output.",
        "- `protocol/`: fixed deployment-default configuration and the 512-example sample provenance. The raw private training texts are intentionally not copied.",
        "- `artifact_index.json`: source paths and checksums for copied artifacts.",
    ])
    readme = BUNDLE / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index.append({"bundle_path": "README.md", "kind": "generated_readme", "sha256": file_hash(readme), "bytes": readme.stat().st_size})
    (BUNDLE / "artifact_index.json").write_text(json.dumps({"bundle": "answer/comparison", "artifacts": index}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "bundle": str(BUNDLE), "systems": list(systems), "artifacts": len(index)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
