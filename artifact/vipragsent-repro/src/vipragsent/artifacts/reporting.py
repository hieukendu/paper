from __future__ import annotations

import csv
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
        "main_pragmatic": write_main_pragmatic_table(tables_dir / "main_pragmatic.md", results.get("main_pragmatic")),
        "ordinary_sentiment": write_ordinary_table(tables_dir / "ordinary_sentiment.md", results.get("ordinary_sentiment")),
        "multitask_ablation": write_ablation_table(tables_dir / "multitask_ablation.md", results.get("multitask_ablation")),
        "low_resource_sarcasm": write_low_resource_table(tables_dir / "low_resource_sarcasm.md", results.get("low_resource_sarcasm")),
        "cost_breakdown": write_cost_table(tables_dir / "cost_breakdown.md", results.get("cost_breakdown")),
    }
    figure_paths = {
        "pipeline": write_pipeline_svg(figures_dir / "fig1_pipeline.svg"),
        "per_phenomenon": write_per_phenomenon_svg(figures_dir / "fig2_per_phenomenon.svg", results.get("main_pragmatic")),
        "low_resource": write_low_resource_svg(figures_dir / "fig4_low_resource_sarcasm.svg", results.get("low_resource_sarcasm")),
        "confusion": write_confusion_svg(figures_dir / "fig5_confusion.svg", results.get("error_confusion")),
        "calibration": write_calibration_svg(figures_dir / "fig7_calibration.svg", results.get("calibration")),
    }
    ledger_rows = write_claim_ledger(claim_ledger, results)
    index = {
        "status": "ok",
        "results_dir": str(results_dir),
        "tables": {key: str(path) for key, path in table_paths.items()},
        "figures": {key: str(path) for key, path in figure_paths.items()},
        "claim_ledger": str(claim_ledger),
        "claim_rows": ledger_rows,
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
    _write_markdown_table(path, "Cost Breakdown", ["System", "GPU h", "Compute USD", "API USD", "Total USD"], rows, result)
    return path


def write_claim_ledger(path: Path, results: dict[str, dict[str, Any] | None]) -> int:
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
        rows.append(
            {
                "anchor": f"Figure7.{system_id}.ece",
                "claim": f"{system.get('label', system_id)} ECE = {system.get('ece')}",
                "source_file": "results/calibration.json",
                "json_path": f"$.systems.{system_id}.ece",
            }
        )
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
        writer = csv.DictWriter(handle, fieldnames=["anchor", "claim", "source_file", "json_path"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
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
    ]
    out = {}
    for name in names:
        path = results_dir / f"{name}.json"
        out[name] = load_json(path) if path.exists() else None
    return out


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
