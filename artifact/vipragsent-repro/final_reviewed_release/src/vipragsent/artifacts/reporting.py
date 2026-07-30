from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.experiments.constants import LOW_RESOURCE_BUDGETS, PRAGMATIC_POLARITY_CLASSES, SYSTEM_LABELS
from vipragsent.utils.io import dump_json, ensure_dir, load_json


def make_final_artifacts(
    *,
    root: str | Path = ".",
    results_dir: str | Path | None = None,
    tables_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    claim_ledger: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    results_dir = Path(results_dir) if results_dir else root / "results"
    tables_dir = ensure_dir(Path(tables_dir) if tables_dir else root / "tables")
    figures_dir = ensure_dir(Path(figures_dir) if figures_dir else root / "figures")
    claim_ledger = Path(claim_ledger) if claim_ledger else results_dir / "claim_ledger.csv"

    results = _load_results(results_dir)
    table_paths = {
        "annotation_agreement": write_annotation_agreement_table(tables_dir / "annotation_agreement.md", results.get("annotation_agreement")),
        "main_pragmatic": write_main_pragmatic_table(tables_dir / "main_pragmatic.md", results.get("main_pragmatic")),
        "ordinary_sentiment": write_ordinary_table(tables_dir / "ordinary_sentiment.md", results.get("ordinary_sentiment")),
        "multitask_ablation": write_ablation_table(tables_dir / "multitask_ablation.md", results.get("multitask_ablation")),
        "low_resource_sarcasm": write_low_resource_table(tables_dir / "low_resource_sarcasm.md", results.get("low_resource_sarcasm")),
        "cost_breakdown": write_cost_table(tables_dir / "cost_breakdown.md", results.get("cost_breakdown")),
        "calibration": write_calibration_table(tables_dir / "calibration.md", results.get("calibration")),
        "error_confusion": write_confusion_table(tables_dir / "error_confusion.md", results.get("error_confusion")),
        "learning_curves": write_learning_curves_table(tables_dir / "learning_curves.md", results.get("learning_curves")),
        "significance": write_significance_table(tables_dir / "significance.md", results.get("significance")),
        "p0_p1_p2_experiments": tables_dir / "p0_p1_p2_experiments.md",
    }
    figure_paths = {
        "pipeline": write_pipeline_svg(figures_dir / "fig1_pipeline.svg"),
        "per_phenomenon": write_per_phenomenon_svg(figures_dir / "fig2_per_phenomenon.svg", results.get("main_pragmatic")),
        "low_resource": write_low_resource_svg(figures_dir / "fig4_low_resource_sarcasm.svg", results.get("low_resource_sarcasm")),
        "confusion": write_confusion_svg(figures_dir / "fig5_confusion.svg", results.get("error_confusion")),
        "learning_curves": write_learning_curves_svg(figures_dir / "fig6_learning_curves.svg", results.get("learning_curves")),
        "calibration": write_calibration_svg(figures_dir / "fig7_calibration.svg", results.get("calibration")),
    }
    ledger_rows = write_claim_ledger(claim_ledger, results, results_dir=results_dir)
    index = {
        "schema_version": "1.1",
        "status": "ok",
        "results_dir": _portable_path(root, results_dir),
        "tables": {key: _portable_path(root, path) for key, path in table_paths.items()},
        "figures": {key: _portable_path(root, path) for key, path in figure_paths.items()},
        "claim_ledger": _portable_path(root, claim_ledger),
        "claim_rows": ledger_rows,
        "source_result_sha256": _result_hashes(results_dir),
    }
    dump_json(results_dir / "artifact_index.json", index)
    return index


def write_main_pragmatic_table(path: Path, result: dict[str, Any] | None) -> Path:
    header = ["System", *PRAGMATIC_LABELS, "macro_pragmatic_f1", "status"]
    rows = []
    for system_id, system in _systems(result).items():
        metrics = system.get("metrics") or {}
        rows.append(
            [
                system.get("label") or SYSTEM_LABELS.get(system_id, system_id),
                *[_metric_cell(metrics.get(label)) for label in PRAGMATIC_LABELS],
                _metric_cell(metrics.get("macro_pragmatic_f1")),
                system.get("status", "missing"),
            ]
        )
    _write_markdown_table(path, "Main Pragmatic Detection", header, rows, result)
    return path


def write_annotation_agreement_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    for field, payload in ((result or {}).get("fields") or {}).items():
        rows.append([
            field,
            payload.get("n", ""),
            _decimal(payload.get("mean_pairwise_percent_agreement")),
            _decimal(payload.get("mean_pairwise_cohen_kappa")),
            _decimal(payload.get("fleiss_kappa_nominal")),
            _decimal(payload.get("krippendorff_alpha_nominal")),
        ])
    macro = (result or {}).get("macro") or {}
    if macro:
        rows.append([
            "**Macro**", "", _decimal(macro.get("mean_pairwise_percent_agreement")),
            _decimal(macro.get("mean_pairwise_cohen_kappa")), _decimal(macro.get("fleiss_kappa_nominal")),
            _decimal(macro.get("krippendorff_alpha_nominal")),
        ])
    _write_markdown_table(path, "Inter-annotator Agreement", ["Label", "N", "Mean pairwise agreement", "Mean pairwise Cohen's kappa", "Fleiss' kappa", "Krippendorff's alpha"], rows, result)
    return path


def write_ordinary_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    for dataset, dataset_payload in (result or {}).get("datasets", {}).items():
        metric_name = dataset_payload.get("metric", "polarity_macro_f1")
        for system_id, system in (dataset_payload.get("systems") or {}).items():
            rows.append(
                [
                    dataset,
                    dataset_payload.get("task", "polarity"),
                    system.get("label") or SYSTEM_LABELS.get(system_id, system_id),
                    _metric_cell((system.get("metrics") or {}).get(metric_name)),
                    system.get("status", "missing"),
                ]
            )
    _write_markdown_table(path, "Public-task Retention", ["Dataset", "Task", "System", "Macro F1", "status"], rows, result)
    return path


def write_ablation_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    for name, payload in ((result or {}).get("rows") or {}).items():
        if isinstance(payload, dict):
            ordinary = payload.get("ordinary_f1", "")
            if isinstance(ordinary, dict):
                ordinary = ordinary.get("macro_across_benchmarks", "")
            rows.append([name, payload.get("macro_pragmatic_f1", ""), ordinary, payload.get("ece", ""), payload.get("status", "")])
        else:
            rows.append([name, payload, "", "", ""])
    _write_markdown_table(path, "Multi-task Ablation", ["Row", "Pragmatic F1", "Ordinary F1", "ECE", "status"], rows, result)
    return path


def write_low_resource_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    for budget in LOW_RESOURCE_BUDGETS:
        budget_payload = ((result or {}).get("budgets") or {}).get(str(budget), {})
        for system_id, system in (budget_payload.get("systems") or {}).items():
            rows.append(
                [
                    budget,
                    system.get("label") or SYSTEM_LABELS.get(system_id, system_id),
                    _metric_cell((system.get("metrics") or {}).get("sarcasm")),
                    system.get("status", ""),
                ]
            )
    _write_markdown_table(path, "Low-resource Sarcasm", ["Positive budget", "System", "sarcasm_f1", "status"], rows, result)
    return path


def write_cost_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    systems = (((result or {}).get("cost") or {}).get("systems") or {})
    for system_id, payload in systems.items():
        rows.append(
            [
                SYSTEM_LABELS.get(system_id, system_id),
                _plain_cell(payload.get("gpu_hours")),
                _plain_cell(payload.get("compute_usd")),
                _plain_cell(payload.get("api_usd")),
                _plain_cell(payload.get("total_usd")),
            ]
        )
    _write_markdown_table(path, "Measured Compute Time", ["System", "GPU h"], [row[:2] for row in rows], result)
    return path


def write_calibration_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    for system_id, payload in ((result or {}).get("systems") or {}).items():
        rows.append([payload.get("label") or SYSTEM_LABELS.get(system_id, system_id), payload.get("n", ""), _plain_cell(payload.get("ece")), payload.get("missing_confidence", ""), payload.get("status", "")])
    _write_markdown_table(path, "Calibration of Pragmatic-polarity Confidence", ["System", "Test records", "ECE", "Missing confidence", "status"], rows, result)
    return path


def write_confusion_table(path: Path, result: dict[str, Any] | None) -> Path:
    labels = (result or {}).get("labels") or []
    counts = (result or {}).get("counts") or []
    normalised = (result or {}).get("row_normalised") or []
    rows = []
    for index, label in enumerate(labels):
        count_row = counts[index] if index < len(counts) else []
        normalised_row = normalised[index] if index < len(normalised) else []
        cells = []
        for prediction_index in range(len(labels)):
            count = count_row[prediction_index] if prediction_index < len(count_row) else 0
            rate = normalised_row[prediction_index] if prediction_index < len(normalised_row) else 0.0
            cells.append(f"{count} ({float(rate) * 100:.2f}%)")
        rows.append([label, *cells])
    _write_markdown_table(path, "Pragmatic-polarity Confusion Matrix", ["Gold \\ Predicted", *labels], rows, result)
    return path


def write_learning_curves_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    for system_id, runs in ((result or {}).get("curves") or {}).items():
        for run in runs:
            history = run.get("history") or []
            dev_values = [item.get("dev_macro_pragmatic_f1") for item in history if item.get("dev_macro_pragmatic_f1") is not None]
            if not dev_values:
                continue
            rows.append([SYSTEM_LABELS.get(system_id, system_id), run.get("seed", ""), len(history), f"{max(dev_values) * 100:.2f}", f"{dev_values[-1] * 100:.2f}"])
    _write_markdown_table(path, "Main Encoder Learning Curves", ["System", "Seed", "Completed epochs", "Peak dev F1", "Final dev F1"], rows, result)
    return path


def write_significance_table(path: Path, result: dict[str, Any] | None) -> Path:
    rows = []
    for system_id, seeds in ((result or {}).get("comparisons") or {}).items():
        for seed, payload in seeds.items():
            ci = payload.get("paired_bootstrap_ci95") or []
            ci_cell = f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]" if len(ci) == 2 else ""
            rows.append([SYSTEM_LABELS.get(system_id, system_id), seed, f"{float(payload.get('delta_macro_pragmatic_f1', 0)):.4f}", ci_cell, _plain_cell(payload.get("two_sided_p"))])
    _write_markdown_table(path, "Paired Bootstrap Significance Against ViPragSent", ["Challenger", "Seed", "Delta F1", "95% paired bootstrap CI", "Raw p"], rows, result)
    return path


def write_claim_ledger(path: Path, results: dict[str, dict[str, Any] | None], *, results_dir: Path) -> int:
    ensure_dir(path.parent)
    rows = []
    main = results.get("main_pragmatic") or {}
    for system_id, system in (main.get("systems") or {}).items():
        for metric_name, metric in (system.get("metrics") or {}).items():
            rows.append(
                {
                    "anchor": f"Table2.{system_id}.{metric_name}",
                    "claim": f"{system.get('label', system_id)} {metric_name} = {_metric_cell(metric)}",
                    "source_file": "results/main_pragmatic.json",
                    "json_path": f"$.systems.{system_id}.metrics.{metric_name}.mean",
                }
            )
    calibration = results.get("calibration") or {}
    for system_id, system in (calibration.get("systems") or {}).items():
        if system.get("status") != "complete" or system.get("ece") is None:
            continue
        rows.append(
            {
                "anchor": f"Figure7.{system_id}.ece",
                "claim": f"{system.get('label', system_id)} ECE = {system.get('ece')}",
                "source_file": "results/calibration.json",
                "json_path": f"$.systems.{system_id}.ece",
            }
        )
    confusion = results.get("error_confusion") or {}
    for index, label in enumerate(confusion.get("labels") or []):
        rows.append({
            "anchor": f"Table8.confusion.diagonal.{label.replace('-', '_')}",
            "claim": f"{label} diagonal recall = {float((confusion.get('row_normalised') or [[]])[index][index]) * 100:.2f}%" if index < len(confusion.get("row_normalised") or []) else f"{label} diagonal recall unavailable",
            "source_file": "results/error_confusion.json",
            "json_path": f"$.row_normalised[{index}][{index}]",
        })
    learning = results.get("learning_curves") or {}
    for system_id, runs in (learning.get("curves") or {}).items():
        for run in runs:
            values = [item.get("dev_macro_pragmatic_f1") for item in (run.get("history") or []) if item.get("dev_macro_pragmatic_f1") is not None]
            if values:
                rows.append({"anchor": f"Figure6.{system_id}.seed_{run.get('seed')}", "claim": f"{SYSTEM_LABELS.get(system_id, system_id)} seed {run.get('seed')} peak development macro pragmatic F1 = {max(values) * 100:.2f}", "source_file": "results/learning_curves.json", "json_path": f"$.curves.{system_id}"})
    significance = results.get("significance") or {}
    for system_id, seeds in (significance.get("comparisons") or {}).items():
        for seed, payload in seeds.items():
            ci = payload.get("paired_bootstrap_ci95") or []
            rows.append({"anchor": f"Table10.{system_id}.seed_{seed}", "claim": f"{SYSTEM_LABELS.get(system_id, system_id)} delta F1 versus ViPragSent = {float(payload.get('delta_macro_pragmatic_f1', 0)):.4f} [{float(ci[0]):.4f}, {float(ci[1]):.4f}]" if len(ci) == 2 else f"{SYSTEM_LABELS.get(system_id, system_id)} delta F1 versus ViPragSent", "source_file": "results/significance.json", "json_path": f"$.comparisons.{system_id}['{seed}'].delta_macro_pragmatic_f1"})
    p0 = results.get("p0_multi_seed_ablation") or {}
    for system_id, payload in (p0.get("systems") or {}).items():
        metric = ((payload.get("aggregate") or {}).get("macro_pragmatic_f1") or {})
        if metric:
            rows.append({"anchor": f"P0.ablation.{system_id}.macro_pragmatic_f1", "claim": f"{system_id} three-seed macro pragmatic F1 = {_metric_cell(metric)}", "source_file": "results/p0_multi_seed_ablation.json", "json_path": f"$.systems.{system_id}.aggregate.macro_pragmatic_f1.mean"})
    visobert = results.get("p0_visobert_baseline") or {}
    for metric_name, metric in (((visobert.get("system") or {}).get("aggregate") or {}).items()):
        rows.append({"anchor": f"P0.visobert_finetune.{metric_name}", "claim": f"visobert_finetune three-seed {metric_name} = {_metric_cell(metric)}", "source_file": "results/p0_visobert_baseline.json", "json_path": f"$.system.aggregate.{metric_name}.mean"})
    p1 = results.get("p1_source_stratified_sensitivity") or {}
    for system_id, strata in (p1.get("systems") or {}).items():
        for stratum, payload in strata.items():
            metric = ((payload.get("aggregate") or {}).get("macro_pragmatic_f1") or {})
            if metric:
                rows.append({"anchor": f"P1.{system_id}.{stratum}.macro_pragmatic_f1", "claim": f"{system_id} {stratum} macro pragmatic F1 = {_metric_cell(metric)}", "source_file": "results/p1_source_stratified_sensitivity.json", "json_path": f"$.systems.{system_id}.{stratum}.aggregate.macro_pragmatic_f1.mean"})
    p2 = results.get("p2_multi_seed_low_resource") or {}
    for budget, systems in (p2.get("budgets") or {}).items():
        for system_id, payload in systems.items():
            metric = ((payload.get("aggregate") or {}).get("sarcasm") or {})
            if metric:
                rows.append({"anchor": f"P2.budget_{budget}.{system_id}.sarcasm", "claim": f"budget {budget} {system_id} three-seed sarcasm macro F1 = {_metric_cell(metric)}", "source_file": "results/p2_multi_seed_low_resource.json", "json_path": f"$.budgets.{budget}.{system_id}.aggregate.sarcasm.mean"})
    for system_id, datasets in (p0.get("external_benchmarks") or {}).items():
        for dataset, report in datasets.items():
            for metric_name, metric in ((report.get("aggregate") or {}).items()):
                rows.append({"anchor": f"External.p0_multi_seed_ablation.{system_id}.{dataset}.{metric_name}", "claim": f"{system_id} {dataset} three-seed {metric_name} = {_metric_cell(metric)}", "source_file": "results/p0_multi_seed_ablation.json", "json_path": f"$.external_benchmarks.{system_id}.{dataset}.aggregate.{metric_name}.mean"})
    for dataset, report in (visobert.get("external_benchmarks") or {}).items():
        for metric_name, metric in ((report.get("aggregate") or {}).items()):
            rows.append({"anchor": f"External.p0_visobert_baseline.visobert_finetune.{dataset}.{metric_name}", "claim": f"visobert_finetune {dataset} three-seed {metric_name} = {_metric_cell(metric)}", "source_file": "results/p0_visobert_baseline.json", "json_path": f"$.external_benchmarks.{dataset}.aggregate.{metric_name}.mean"})
    if not rows:
        rows.append(
            {
                "anchor": "Artifacts.status",
                "claim": "No complete metrics available yet",
                "source_file": "results/*.json",
                "json_path": "$.status",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor", "claim", "source_file", "json_path", "source_sha256"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            source = results_dir / Path(str(row["source_file"])).name
            row["source_sha256"] = f"sha256:{_sha256(source)}" if source.exists() else ""
            writer.writerow(row)
    return len(rows)


def write_pipeline_svg(path: Path) -> Path:
    boxes = [
        ("Gold JSONL", 20),
        ("Prediction JSONL", 170),
        ("Metric engine", 340),
        ("results/*.json", 500),
        ("tables/figures", 660),
    ]
    parts = [_svg_header(820, 180)]
    for text, x in boxes:
        parts.append(f'<rect x="{x}" y="55" width="120" height="55" rx="8" fill="#eaf0f9" stroke="#1f4e96"/>')
        parts.append(f'<text x="{x + 60}" y="87" text-anchor="middle" font-size="12">{_escape(text)}</text>')
    for _, x in boxes[:-1]:
        parts.append(f'<line x1="{x + 120}" y1="82" x2="{x + 145}" y2="82" stroke="#1f4e96" marker-end="url(#arrow)"/>')
    parts.append(_svg_footer())
    _write_text(path, "\n".join(parts))
    return path


def write_per_phenomenon_svg(path: Path, result: dict[str, Any] | None) -> Path:
    systems = _complete_systems(result)
    if not systems:
        return _write_empty_svg(path, "Per-phenomenon F1 is blocked until prediction files exist.")
    selected = list(systems.items())[:5]
    labels = PRAGMATIC_LABELS
    values = {
        system_id: [_metric_mean((payload.get("metrics") or {}).get(label)) for label in labels]
        for system_id, payload in selected
    }
    return _write_grouped_bar_svg(path, "Per-phenomenon macro-F1", labels, values)


def write_low_resource_svg(path: Path, result: dict[str, Any] | None) -> Path:
    budgets = (result or {}).get("budgets") or {}
    series: dict[str, list[float | None]] = {}
    for budget in LOW_RESOURCE_BUDGETS:
        systems = (budgets.get(str(budget)) or {}).get("systems") or {}
        for system_id, payload in systems.items():
            series.setdefault(system_id, []).append(_metric_mean((payload.get("metrics") or {}).get("sarcasm")))
    series = {key: values for key, values in series.items() if any(value is not None for value in values)}
    if not series:
        return _write_empty_svg(path, "Low-resource sarcasm is blocked until prediction files exist.")
    return _write_line_svg(path, "Low-resource sarcasm F1", [str(budget) for budget in LOW_RESOURCE_BUDGETS], series)


def write_confusion_svg(path: Path, result: dict[str, Any] | None) -> Path:
    matrix = (result or {}).get("row_normalised") or []
    labels = (result or {}).get("labels") or PRAGMATIC_POLARITY_CLASSES
    if not matrix:
        return _write_empty_svg(path, "Confusion matrix is blocked until prediction files exist.")
    cell = 45
    width = 170 + cell * len(labels)
    height = 120 + cell * len(labels)
    parts = [_svg_header(width, height), '<text x="20" y="28" font-size="16">Pragmatic polarity confusion</text>']
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            shade = int(255 - min(1.0, float(value)) * 170)
            parts.append(f'<rect x="{120 + j * cell}" y="{55 + i * cell}" width="{cell}" height="{cell}" fill="rgb({shade},{shade},{255})" stroke="#fff"/>')
            parts.append(f'<text x="{120 + j * cell + cell / 2}" y="{55 + i * cell + 27}" text-anchor="middle" font-size="10">{float(value):.2f}</text>')
    for i, label in enumerate(labels):
        parts.append(f'<text x="115" y="{55 + i * cell + 27}" text-anchor="end" font-size="10">{_escape(label)}</text>')
        parts.append(f'<text x="{120 + i * cell + cell / 2}" y="{50}" text-anchor="middle" font-size="10">{_escape(label)}</text>')
    parts.append(_svg_footer())
    _write_text(path, "\n".join(parts))
    return path


def write_calibration_svg(path: Path, result: dict[str, Any] | None) -> Path:
    systems = {
        system_id: payload
        for system_id, payload in ((result or {}).get("systems") or {}).items()
        if payload.get("status") == "complete" and payload.get("bins")
    }
    if not systems:
        return _write_empty_svg(path, "Calibration is blocked until confidence scores exist.")
    series = {}
    for system_id, payload in systems.items():
        values = []
        for bucket in payload["bins"]:
            values.append(bucket.get("acc") if bucket.get("n", 0) else None)
        series[system_id] = values
    return _write_line_svg(path, "Reliability diagram accuracy by confidence bin", [str(i) for i in range(10)], series, y_max=1.0)


def write_learning_curves_svg(path: Path, result: dict[str, Any] | None) -> Path:
    series: dict[str, list[float | None]] = {}
    for system_id, runs in ((result or {}).get("curves") or {}).items():
        encoder_runs = []
        for run in runs:
            values = [item.get("dev_macro_pragmatic_f1") for item in (run.get("history") or [])]
            if any(value is not None for value in values):
                encoder_runs.append(values)
        if not encoder_runs:
            continue
        length = max(len(values) for values in encoder_runs)
        series[system_id] = [sum(float(values[index]) for values in encoder_runs if index < len(values) and values[index] is not None) / sum(1 for values in encoder_runs if index < len(values) and values[index] is not None) if any(index < len(values) and values[index] is not None for values in encoder_runs) else None for index in range(length)]
    if not series:
        return _write_empty_svg(path, "Learning curves are blocked until development metrics exist.")
    return _write_line_svg(path, "Mean development macro pragmatic F1", [str(index + 1) for index in range(max(len(values) for values in series.values()))], series, y_max=1.0)


def _load_results(results_dir: Path) -> dict[str, dict[str, Any] | None]:
    names = [
        "main_pragmatic",
        "ordinary_sentiment",
        "multitask_ablation",
        "low_resource_sarcasm",
        "error_confusion",
        "calibration",
        "cost_breakdown",
        "learning_curves",
        "annotation_agreement",
        "significance",
        "p0_multi_seed_ablation",
        "p0_visobert_baseline",
        "p1_source_stratified_sensitivity",
        "p2_multi_seed_low_resource",
    ]
    out = {}
    for name in names:
        path = results_dir / f"{name}.json"
        out[name] = load_json(path) if path.exists() else None
    return out


def _portable_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_hashes(results_dir: Path) -> dict[str, str]:
    return {
        path.name: f"sha256:{_sha256(path)}"
        for path in sorted(results_dir.glob("*.json"))
        if path.name != "artifact_index.json"
    }


def _decimal(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else ""


def _systems(result: dict[str, Any] | None) -> dict[str, Any]:
    return (result or {}).get("systems") or {}


def _complete_systems(result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        system_id: payload
        for system_id, payload in _systems(result).items()
        if payload.get("status") in {"complete", "partial"} and payload.get("metrics")
    }


def _metric_cell(metric: Any) -> str:
    mean_value = _metric_mean(metric)
    if mean_value is None:
        return ""
    ci = metric.get("ci95") if isinstance(metric, dict) else None
    if isinstance(ci, list) and len(ci) == 2:
        return f"{mean_value:.2f} [{float(ci[0]):.2f}, {float(ci[1]):.2f}]"
    return f"{mean_value:.2f}"


def _metric_mean(metric: Any) -> float | None:
    if isinstance(metric, dict):
        for key in ("mean", "value"):
            if metric.get(key) is not None:
                return float(metric[key])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def _plain_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_markdown_table(path: Path, title: str, header: list[str], rows: list[list[Any]], result: dict[str, Any] | None) -> None:
    lines = [f"# {title}", ""]
    if result:
        lines.append(f"Status: `{result.get('status', 'missing')}`")
        if result.get("blocked_reason"):
            lines.append(f"Blocked reason: {result['blocked_reason']}")
        lines.append("")
    if not rows:
        lines.append("No rows available yet.")
    else:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_escape_markdown(str(cell)) for cell in row) + " |")
    _write_text(path, "\n".join(lines) + "\n")


def _write_grouped_bar_svg(path: Path, title: str, labels: list[str], values: dict[str, list[float | None]]) -> Path:
    width, height = 900, 360
    plot_x, plot_y, plot_w, plot_h = 80, 50, 760, 240
    y_max = max([value for vals in values.values() for value in vals if value is not None] or [1.0])
    y_max = max(1.0, math.ceil(y_max / 10) * 10)
    systems = list(values)
    bar_w = plot_w / max(1, len(labels)) / (len(systems) + 1)
    parts = [_svg_header(width, height), f'<text x="20" y="25" font-size="16">{_escape(title)}</text>']
    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#333"/>')
    for i, label in enumerate(labels):
        group_x = plot_x + i * (plot_w / len(labels))
        for j, system_id in enumerate(systems):
            value = values[system_id][i]
            if value is None:
                continue
            h = (value / y_max) * plot_h
            x = group_x + 8 + j * bar_w
            y = plot_y + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" fill="{_color(j)}"/>')
        parts.append(f'<text x="{group_x + 45}" y="{plot_y + plot_h + 35}" transform="rotate(25 {group_x + 45},{plot_y + plot_h + 35})" font-size="10">{_escape(label)}</text>')
    for j, system_id in enumerate(systems):
        parts.append(f'<rect x="{plot_x + plot_w - 160}" y="{plot_y + j * 18}" width="10" height="10" fill="{_color(j)}"/>')
        parts.append(f'<text x="{plot_x + plot_w - 145}" y="{plot_y + 9 + j * 18}" font-size="10">{_escape(SYSTEM_LABELS.get(system_id, system_id))}</text>')
    parts.append(_svg_footer())
    _write_text(path, "\n".join(parts))
    return path


def _write_line_svg(
    path: Path,
    title: str,
    labels: list[str],
    series: dict[str, list[float | None]],
    *,
    y_max: float | None = None,
) -> Path:
    width, height = 760, 330
    plot_x, plot_y, plot_w, plot_h = 70, 45, 600, 220
    values = [value for vals in series.values() for value in vals if value is not None]
    y_max = y_max or max(1.0, math.ceil(max(values or [1.0]) / 10) * 10)
    parts = [_svg_header(width, height), f'<text x="20" y="25" font-size="16">{_escape(title)}</text>']
    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#333"/>')
    for j, (system_id, vals) in enumerate(series.items()):
        points = []
        for i, value in enumerate(vals):
            if value is None:
                continue
            x = plot_x + (i / max(1, len(labels) - 1)) * plot_w
            y = plot_y + plot_h - (float(value) / y_max) * plot_h
            points.append((x, y))
        if len(points) >= 2:
            points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{points_attr}" fill="none" stroke="{_color(j)}" stroke-width="2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{_color(j)}"/>')
        parts.append(f'<rect x="{plot_x + plot_w + 20}" y="{plot_y + j * 18}" width="10" height="10" fill="{_color(j)}"/>')
        parts.append(f'<text x="{plot_x + plot_w + 35}" y="{plot_y + 9 + j * 18}" font-size="10">{_escape(SYSTEM_LABELS.get(system_id, system_id))}</text>')
    for i, label in enumerate(labels):
        x = plot_x + (i / max(1, len(labels) - 1)) * plot_w
        parts.append(f'<text x="{x:.1f}" y="{plot_y + plot_h + 20}" text-anchor="middle" font-size="10">{_escape(label)}</text>')
    parts.append(_svg_footer())
    _write_text(path, "\n".join(parts))
    return path


def _write_empty_svg(path: Path, message: str) -> Path:
    parts = [
        _svg_header(760, 180),
        '<rect x="20" y="35" width="720" height="100" rx="8" fill="#f7f7f7" stroke="#999"/>',
        f'<text x="380" y="90" text-anchor="middle" font-size="14">{_escape(message)}</text>',
        _svg_footer(),
    ]
    _write_text(path, "\n".join(parts))
    return path


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
        '<path d="M0,0 L10,4 L0,8 z" fill="#1f4e96"/></marker></defs>'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}</style>'
    )


def _svg_footer() -> str:
    return "</svg>"


def _color(idx: int) -> str:
    colors = ["#5b8ec9", "#7aa17a", "#a892c9", "#f0a04b", "#1f4e96", "#c98686", "#8c8c8c"]
    return colors[idx % len(colors)]


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def _write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")
