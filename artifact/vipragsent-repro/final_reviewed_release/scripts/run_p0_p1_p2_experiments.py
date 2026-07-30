from __future__ import annotations

"""Sequential, resumable P0/P1/P2 paper-strengthening experiment runner."""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import HfApi, snapshot_download
from vipragsent.experiments.evaluator import aggregate_seed_reports, evaluate_emotion_records, evaluate_polarity_records, evaluate_pragmatic_records, load_gold_records
from vipragsent.experiments.predictions import join_gold_predictions, load_prediction_records

SEEDS = (20260520, 20260521, 20260522)
NEW_SEEDS = (20260521, 20260522)
BUDGETS = (64, 128, 256, 512, 1024)
PHOBERT = Path("/root/hf_cache/models/phobert-base")
VISOBERT = ROOT / "models/huggingface/visobert"
VISOBERT_REPO = "uitnlp/visobert"
VISOBERT_REVISION = "196a62afad9cbe4f52a54aabad828b13f0eec59a"
ABLATIONS = {
    "vipragsent_no_rationale": ("--no-rationale",),
    "vipragsent_no_emotion": ("--no-emotion",),
    "vipragsent_no_polarity": ("--no-polarity",),
    "vipragsent_no_uncertainty": ("--no-uncertainty",),
}
BASELINE_FLAGS = ("--no-rationale", "--no-emotion", "--no-polarity")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_logged(name: str, command: list[str], trace: dict) -> None:
    logs = ROOT / "logs/p0_p1_p2"; logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}.log"
    if trace.get("completed", {}).get(name):
        return
    env = os.environ.copy(); env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"})
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"started_at": now(), "cwd": str(ROOT), "command": command}, ensure_ascii=False) + "\n\n")
        code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env, check=False).returncode
        handle.write("\n" + json.dumps({"finished_at": now(), "returncode": code}) + "\n")
    if code:
        raise RuntimeError(f"{name} failed (return code {code}); see {log}")
    trace.setdefault("completed", {})[name] = {"log": str(log), "completed_at": now()}
    (ROOT / "results/p0p1p2_runs.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(name: str, system: str, seed: int, model: Path, flags: tuple[str, ...], output_root: Path, prediction_root: Path, trace: dict, train_path: Path | None = None) -> Path:
    manifest = output_root / system / str(seed) / "run_manifest.json"
    prediction = prediction_root / system / f"{seed}.jsonl"
    if manifest.exists() and prediction.exists():
        trace.setdefault("completed", {})[name] = {"existing": True, "manifest": str(manifest), "prediction": str(prediction)}
        return prediction
    cmd = [sys.executable, "scripts/train_multitask_encoder.py", "--model-id", str(model), "--system", system, "--seed", str(seed), "--output-root", str(output_root), "--prediction-root", str(prediction_root), "--batch-size", "16", "--grad-accum", "2", "--epochs", "10", "--patience", "2", "--bf16", *flags]
    if train_path: cmd += ["--train", str(train_path)]
    run_logged(name, cmd, trace)
    if not (manifest.exists() and prediction.exists()): raise RuntimeError(f"{name} did not produce manifest and prediction")
    return prediction


def predict(name: str, system: str, seed: int, model: Path, flags: tuple[str, ...], checkpoint: Path, dataset: str, trace: dict) -> Path:
    out = ROOT / "results/predictions/ordinary_sentiment" / dataset / system / f"{seed}.jsonl"
    if out.exists():
        trace.setdefault("completed", {})[name] = {"existing": True, "prediction": str(out)}; return out
    data = ROOT / f"data/processed/public_eval/{dataset}_test.jsonl"
    cmd = [sys.executable, "scripts/train_multitask_encoder.py", "--model-id", str(model), "--system", system, "--seed", str(seed), "--checkpoint", str(checkpoint), "--predict-data", str(data), "--prediction-output", str(out), "--batch-size", "16", "--bf16", *flags]
    run_logged(name, cmd, trace)
    if not out.exists(): raise RuntimeError(f"{name} did not produce {out}")
    return out


def ensure_visobert(trace: dict) -> None:
    if not (VISOBERT / "pytorch_model.bin").exists():
        snapshot_download(VISOBERT_REPO, revision=VISOBERT_REVISION, local_dir=VISOBERT, allow_patterns=["config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "README.md", ".gitattributes"])
    info = HfApi().model_info(VISOBERT_REPO, revision=VISOBERT_REVISION)
    trace["visobert_model"] = {"repo_id": VISOBERT_REPO, "revision": VISOBERT_REVISION, "resolved_revision": info.sha, "local_path": str(VISOBERT)}


def pragmatic(system: str, root: Path, gold: list[dict], resamples: int, metrics: list[str] | None = None) -> dict:
    reports = []
    for seed in SEEDS:
        path = root / system / f"{seed}.jsonl"
        joined = join_gold_predictions(gold, load_prediction_records(path))
        item = evaluate_pragmatic_records(joined, bootstrap_resamples=resamples); item.update({"seed": seed, "prediction": str(path)}); reports.append(item)
    names = metrics or ["macro_pragmatic_f1"]
    return {"per_seed": reports, "aggregate": aggregate_seed_reports(reports, names)}


def ordinary(system: str, seed: int, dataset: str, resamples: int) -> dict:
    task = "emotion" if dataset == "uit_vsmec" else "polarity"
    gold = load_gold_records(ROOT / f"data/processed/public_eval/{dataset}_test.jsonl", task=task, require_adjudicated=False)
    pred = load_prediction_records(ROOT / f"results/predictions/ordinary_sentiment/{dataset}/{system}/{seed}.jsonl")
    joined = join_gold_predictions(gold, pred)
    return evaluate_emotion_records(joined, bootstrap_resamples=resamples) if task == "emotion" else evaluate_polarity_records(joined, bootstrap_resamples=resamples)


def ordinary_three(system: str, dataset: str, resamples: int) -> dict:
    reports = []
    metric = "emotion_macro_f1" if dataset == "uit_vsmec" else "polarity_macro_f1"
    for seed in SEEDS:
        report = ordinary(system, seed, dataset, resamples); report["seed"] = seed; reports.append(report)
    return {"per_seed": reports, "aggregate": aggregate_seed_reports(reports, [metric])}


def source_of(record: dict) -> str:
    return str((record.get("source") or {}).get("dataset", "unknown"))


def write_table(results: dict) -> None:
    def cell(item: dict, metric: str = "macro_pragmatic_f1") -> str:
        x = item["aggregate"][metric]; return f"{x['mean']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"
    lines = ["# P0/P1/P2 strengthening experiments", "", "All values are macro F1 (%), mean across three seeds with normal 95% CI across seeds.", "", "## P0: multi-seed ablation and ViSoBERT", "", "| System | Macro pragmatic F1 |", "|---|---:|"]
    for k, v in results["p0_multi_seed_ablation"]["systems"].items(): lines.append(f"| {k} | {cell(v)} |")
    lines.append(f"| visobert_finetune | {cell(results['p0_visobert_baseline']['system'])} |")
    lines += ["", "## P1: source-stratified sensitivity", "", "| System | ViSoBERT-local test | VIVID-derived test |", "|---|---:|---:|"]
    for k, v in results["p1_source_stratified_sensitivity"]["systems"].items(): lines.append(f"| {k} | {cell(v['visobert_local'])} | {cell(v['VIVID_seed_and_irony_generation'])} |")
    lines += ["", "## P2: multi-seed low-resource sarcasm", "", "| Positive budget | System | Sarcasm macro F1 |", "|---:|---|---:|"]
    for b, systems in results["p2_multi_seed_low_resource"]["budgets"].items():
        for k, v in systems.items(): lines.append(f"| {b} | {k} | {cell(v, 'sarcasm')} |")
    lines += ["", "P1 is descriptive only: VIVID rows retain `unknown_or_dataset_card` licensing in the stored source metadata and require authorization/license review before public release or causal source claims.", ""]
    (ROOT / "tables/p0_p1_p2_experiments.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--bootstrap-resamples", type=int, default=1000); args = parser.parse_args()
    if not PHOBERT.exists(): raise RuntimeError("PhoBERT cache missing. Run scripts/16_download_models.py --skip-7b.")
    trace_path = ROOT / "results/p0p1p2_runs.json"
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {"started_at": now(), "completed": {}, "protocol": "P0/P1/P2 sequential runner"}
    ensure_visobert(trace)

    # P0: only the eight missing ablation seeds, then their 24 external predictions.
    for system, flags in ABLATIONS.items():
        for seed in NEW_SEEDS:
            train(f"p0_train_{system}_{seed}", system, seed, PHOBERT, flags, ROOT / "outputs", ROOT / "results/predictions/main_pragmatic", trace)
            checkpoint = ROOT / "outputs" / system / str(seed) / "best.pt"
            for dataset in ("uit_vsfc", "uit_vsmec", "aivivn_2019"):
                predict(f"p0_predict_{dataset}_{system}_{seed}", system, seed, PHOBERT, flags, checkpoint, dataset, trace)
    for seed in SEEDS:
        train(f"p0_train_visobert_{seed}", "visobert_finetune", seed, VISOBERT, BASELINE_FLAGS, ROOT / "outputs", ROOT / "results/predictions/main_pragmatic", trace)
        checkpoint = ROOT / "outputs/visobert_finetune" / str(seed) / "best.pt"
        for dataset in ("uit_vsfc", "uit_vsmec", "aivivn_2019"):
            predict(f"p0_predict_{dataset}_visobert_{seed}", "visobert_finetune", seed, VISOBERT, BASELINE_FLAGS, checkpoint, dataset, trace)

    # P2: the twenty missing registered seeds (5 budgets x 2 systems x 2 seeds).
    for budget in BUDGETS:
        subset = ROOT / f"data/processed/low_resource/sarcasm_{budget}.jsonl"
        for system, flags in (("phobert_finetune", BASELINE_FLAGS), ("vipragsent_full", ())):
            for seed in NEW_SEEDS:
                train(f"p2_train_{budget}_{system}_{seed}", system, seed, PHOBERT, flags, ROOT / f"outputs/low_resource/{budget}", ROOT / f"results/predictions/low_resource_sarcasm/{budget}", trace, subset)

    gold = load_gold_records(ROOT / "data/processed/vipragsent_test.jsonl", task="pragmatic", require_adjudicated=True)
    p0_names = ["vipragsent_full", "phobert_finetune", *ABLATIONS]
    p0_systems = {system: pragmatic(system, ROOT / "results/predictions/main_pragmatic", gold, args.bootstrap_resamples) for system in p0_names}
    p0_ordinary = {system: {dataset: ordinary_three(system, dataset, args.bootstrap_resamples) for dataset in ("uit_vsfc", "uit_vsmec", "aivivn_2019")} for system in [*ABLATIONS, "visobert_finetune"]}
    viso = pragmatic("visobert_finetune", ROOT / "results/predictions/main_pragmatic", gold, args.bootstrap_resamples)
    source_systems = {}
    for system in ["vipragsent_full", "phobert_finetune", *ABLATIONS, "visobert_finetune"]:
        source_systems[system] = {source: pragmatic(system, ROOT / "results/predictions/main_pragmatic", [r for r in gold if source_of(r) == source], args.bootstrap_resamples) for source in ("visobert_local", "VIVID_seed_and_irony_generation")}
    low = {}
    for budget in BUDGETS:
        low[str(budget)] = {system: pragmatic(system, ROOT / f"results/predictions/low_resource_sarcasm/{budget}", gold, args.bootstrap_resamples, ["sarcasm"]) for system in ("phobert_finetune", "vipragsent_full")}
    results = {
        "schema_version": "1.0", "status": "complete", "created_at": now(), "seeds": list(SEEDS), "bootstrap_resamples": args.bootstrap_resamples,
        "p0_multi_seed_ablation": {"new_finetunes": 8, "new_external_inferences": 24, "systems": p0_systems, "external_benchmarks": {system: p0_ordinary[system] for system in ABLATIONS}},
        "p0_visobert_baseline": {"new_finetunes": 3, "new_external_inferences": 9, "model": trace["visobert_model"], "system": viso, "external_benchmarks": p0_ordinary["visobert_finetune"]},
        "p1_source_stratified_sensitivity": {"fine_tunes": 0, "test_source_counts": dict(Counter(source_of(r) for r in gold)), "license_note": "VIVID source metadata is unknown_or_dataset_card; authorization/license review remains required.", "systems": source_systems},
        "p2_multi_seed_low_resource": {"new_finetunes": 20, "budgets": low},
    }
    for key in ("p0_multi_seed_ablation", "p0_visobert_baseline", "p1_source_stratified_sensitivity", "p2_multi_seed_low_resource"):
        (ROOT / f"results/{key}.json").write_text(json.dumps(results[key], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results/p0_p1_p2_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_table(results); trace.update({"finished_at": now(), "status": "complete"}); trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "result": "results/p0_p1_p2_summary.json", "runs": len(trace["completed"])}, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
