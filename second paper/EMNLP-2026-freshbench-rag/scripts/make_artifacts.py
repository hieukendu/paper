#!/usr/bin/env python3
"""Generate fixed-seed simulated artifacts for FreshBench-RAG."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

DATA = {'paper': 'FreshBench-RAG', 'synthetic_label': 'Synthetic example result - replace before submission', 'rows': [{'method': 'Control', 'quality': 47.6, 'risk': 32.1, 'cost': 2.17}, {'method': 'Cheap baseline', 'quality': 53.5, 'risk': 21.6, 'cost': 1.36}, {'method': 'Strong API', 'quality': 54.3, 'risk': 18.1, 'cost': 1.54}, {'method': 'FreshBench-RAG', 'quality': 58.2, 'risk': 11.1, 'cost': 1.19}], 'ablation': [{'component': 'Full system', 'score': 58.2}, {'component': 'No protocol control', 'score': 56.1}, {'component': 'No uncertainty', 'score': 54.5}, {'component': 'No validation filter', 'score': 56.0}], 'robustness': [{'slice': 'Seen-style', 'score': 57.3}, {'slice': 'Hard-language/domain', 'score': 56.2}, {'slice': 'Noisy evidence/labels', 'score': 55.9}, {'slice': 'Fresh/shifted', 'score': 55.2}], 'calibration': [{'bucket': '0.0-0.25', 'expected': 0.18, 'observed': 0.14}, {'bucket': '0.25-0.50', 'expected': 0.38, 'observed': 0.4}, {'bucket': '0.50-0.75', 'expected': 0.62, 'observed': 0.57}, {'bucket': '0.75-1.00', 'expected': 0.86, 'observed': 0.8}], 'dataset_matrix': [{'slice': 'Primary benchmark', 'items': 1862, 'validation': 'automatic + audit', 'purpose': 'main comparison'}, {'slice': 'Human/expert subset', 'items': 386, 'validation': 'human adjudication', 'purpose': 'calibration'}, {'slice': 'Robustness slice', 'items': 513, 'validation': 'filtered synthetic', 'purpose': 'stress test'}, {'slice': 'Ablation slice', 'items': 913, 'validation': 'cached replay', 'purpose': 'component isolation'}], 'error_table': [{'error': 'unsupported output', 'share': 8.4}, {'error': 'semantic drift', 'share': 9.3}, {'error': 'retrieval/selection miss', 'share': 17.4}, {'error': 'calibration failure', 'share': 6.8}]}

def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def write_pdf(path, title, lines, bars=None):
    bars = bars or []
    commands = ["BT /F1 14 Tf 40 360 Td (" + pdf_escape(title) + ") Tj ET"]
    y = 330
    for line in lines:
        commands.append(f"BT /F1 9 Tf 40 {y} Td (" + pdf_escape(line) + ") Tj ET")
        y -= 16
    x0, y0, width, height = 55, 75, 300, 150
    commands.append(f"0.85 0.85 0.85 RG 0.5 w {x0} {y0} {width} 0 l S")
    if bars:
        maxv = max(v for _, v in bars) or 1
        barw = 42
        for i, (label, value) in enumerate(bars):
            x = x0 + i * 68
            h = height * value / maxv
            commands.append(f"0.12 0.31 0.55 rg {x} {y0} {barw} {h:.2f} re f")
            commands.append(f"BT /F1 7 Tf {x} 55 Td (" + pdf_escape(label[:12]) + ") Tj ET")
            commands.append(f"BT /F1 7 Tf {x} {y0 + h + 6:.2f} Td (" + pdf_escape(f"{value:.1f}") + ") Tj ET")
    stream = "\n".join(commands).encode("latin-1", "replace")
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 420 420] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(out)

def main():
    (RES / "main_results.json").write_text(json.dumps(DATA, indent=2), encoding="utf-8")
    with (RES / "claim_ledger.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "value", "source", "status"])
        for row in DATA["rows"]:
            writer.writerow([row["method"], row["quality"], "fixed-seed simulator", DATA["synthetic_label"]])
        for row in DATA["ablation"]:
            writer.writerow([row["component"], row["score"], "fixed-seed simulator", DATA["synthetic_label"]])
        for row in DATA["robustness"]:
            writer.writerow([row["slice"], row["score"], "fixed-seed simulator", DATA["synthetic_label"]])
    (RES / "ablations.json").write_text(json.dumps({"synthetic_label": DATA["synthetic_label"], "ablation": DATA["ablation"]}, indent=2), encoding="utf-8")
    (RES / "robustness.json").write_text(json.dumps({"synthetic_label": DATA["synthetic_label"], "robustness": DATA["robustness"]}, indent=2), encoding="utf-8")
    (RES / "calibration.json").write_text(json.dumps({"synthetic_label": DATA["synthetic_label"], "calibration": DATA["calibration"]}, indent=2), encoding="utf-8")
    with (RES / "dataset_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slice", "items", "validation", "purpose"])
        writer.writeheader()
        writer.writerows(DATA["dataset_matrix"])
    with (RES / "error_taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["error", "share"])
        writer.writeheader()
        writer.writerows(DATA["error_table"])
    write_pdf(
        FIG / "fig1_tradeoff.pdf",
        "FreshBench-RAG tradeoff",
        [DATA["synthetic_label"], "Illustrative quality scores from simulated data."],
        [(row["method"], row["quality"]) for row in DATA["rows"]],
    )
    write_pdf(
        FIG / "fig2_ablation.pdf",
        "FreshBench-RAG ablation",
        [DATA["synthetic_label"], "Illustrative component scores from simulated data."],
        [(row["component"], row["score"]) for row in DATA["ablation"]],
    )
    write_pdf(
        FIG / "fig3_robustness.pdf",
        "FreshBench-RAG robustness",
        [DATA["synthetic_label"], "Illustrative shifted-slice scores from simulated data."],
        [(row["slice"], row["score"]) for row in DATA["robustness"]],
    )
    write_pdf(
        FIG / "fig4_calibration.pdf",
        "FreshBench-RAG calibration",
        [DATA["synthetic_label"], "Illustrative observed calibration by bucket."],
        [(row["bucket"], row["observed"] * 100) for row in DATA["calibration"]],
    )

if __name__ == "__main__":
    main()
