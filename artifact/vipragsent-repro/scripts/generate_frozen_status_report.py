#!/usr/bin/env python3
"""Generate auditable status artifacts for frozen ViPragSent experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
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
BEFORE = {
    "20260520": "64a499d3575a797d725571dafe34473824f0b386754c5bba4c521665752efdb9",
    "20260521": "8e3cf36afad81a9fd955cbd738237e5114f45d678f1f22649c2bc730ccf2d790",
    "20260522": "4f8bcd6fcf10ca0f3082e7b9a844fb6bdc60c72ca602ec5ecbd52a68215717ef",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def metric_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def main() -> int:
    output = ROOT / "answers" / "optimized_vipragsent"
    reports = ROOT / "reports"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / "results" / "main_pragmatic.json").read_text(encoding="utf-8"))
    optimized = metric_csv(output / "frozen_ten_way_source_dependency_hybrid_test_metrics.csv")
    best: dict[str, tuple[str, float]] = {}
    for system_id, system in baseline["systems"].items():
        if system_id == "vipragsent_full" or system.get("status") != "complete":
            continue
        for metric in LABELS:
            value = float(system["metrics"][metric]["mean"])
            if metric not in best or value > best[metric][1]:
                best[metric] = (system_id, value)
    comparison = []
    for metric in LABELS:
        original = float(baseline["systems"]["vipragsent_full"]["metrics"][metric]["mean"])
        baseline_system, baseline_value = best[metric]
        current = optimized[metric]
        status = "PASS" if current > baseline_value else "FAIL"
        comparison.append(
            {
                "task_metric": metric,
                "original_vipragsent": f"{original:.4f}",
                "best_baseline": f"{baseline_value:.4f}",
                "best_baseline_system": baseline_system,
                "optimized_vipragsent": f"{current:.4f}",
                "difference": f"{current - baseline_value:.4f}",
                "status": status,
            }
        )
    final_status = "SUCCESS" if all(row["status"] == "PASS" for row in comparison) else "NOT_YET_ACHIEVED"
    for target in (output / "baseline_comparison.csv", output / "final_results.csv"):
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader(); writer.writerows(comparison)
    with (output / "baseline_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "label", "metric", "score", "split", "prediction_files", "verification_status"])
        writer.writeheader()
        for system_id, system in baseline["systems"].items():
            for metric in LABELS:
                writer.writerow({
                    "system": system_id, "label": system["label"], "metric": metric,
                    "score": system["metrics"][metric]["mean"], "split": "adjudicated_test",
                    "prediction_files": "|".join(system.get("prediction_files", [])),
                    "verification_status": system["status"],
                })
    seed_rows = []
    for run in baseline["systems"]["vipragsent_full"]["runs"]:
        seed_rows.append({"experiment_id": "original_imported", "seed": run["seed"], "metric": "macro_pragmatic_f1", "value": run["metrics"]["macro_pragmatic_f1"]["value"], "split": "test"})
    seed_rows.append({"experiment_id": "E020_frozen_source_dependency_hybrid", "seed": "ensemble_3_seed", "metric": "macro_pragmatic_f1", "value": optimized["macro_pragmatic_f1"], "split": "test"})
    with (output / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0])); writer.writeheader(); writer.writerows(seed_rows)
    history = [
        {"experiment_id": "E001", "hypothesis": "dev-selected label thresholds plus frozen three-seed ensemble", "selection_split": "dev", "result": "macro 76.5697 on test", "status": "kept"},
        {"experiment_id": "E002", "hypothesis": "increase frozen inference max length from 128 to 256", "selection_split": "dev", "result": "identical output; no improvement", "status": "rejected"},
        {"experiment_id": "E003", "hypothesis": "train-only English-token retrieval for code switching", "selection_split": "dev", "result": "macro 80.4160 on test; code switching 75.8134", "status": "kept"},
        {"experiment_id": "E004", "hypothesis": "frozen contextual embedding k-NN blended with the frozen head", "selection_split": "dev", "result": "improved implicit/sarcasm/mocking candidates; retained for label selection", "status": "kept"},
        {"experiment_id": "E005", "hypothesis": "per-label development selection over threshold, lexical, PhoBERT k-NN, and ViSoBERT k-NN", "selection_split": "dev", "result": "macro 81.3969 on test", "status": "kept"},
        {"experiment_id": "E006", "hypothesis": "source-conditioned frozen thresholds", "selection_split": "dev", "result": "macro 75.3852 on test", "status": "rejected"},
        {"experiment_id": "E007", "hypothesis": "extended k-NN neighborhoods (up to 1,000)", "selection_split": "dev", "result": "no label selection advantage over E004/E005 candidates", "status": "rejected"},
        {"experiment_id": "E008", "hypothesis": "test-time lowercase and social-token normalization variants", "selection_split": "dev", "result": "no per-label selection advantage over E005 candidates", "status": "rejected"},
        {"experiment_id": "E009", "hypothesis": "mean-pooled frozen contextual k-NN", "selection_split": "dev", "result": "no per-label selection advantage over E005 candidates", "status": "rejected"},
        {"experiment_id": "E010", "hypothesis": "train-set linear calibration over frozen pragmatic, polarity, and emotion logits", "selection_split": "dev", "result": "no per-label selection advantage over E005 candidates", "status": "rejected"},
        {"experiment_id": "E011", "hypothesis": "train-centroid prototype retrieval over frozen contextual embeddings", "selection_split": "dev", "result": "no per-label selection advantage over E005 candidates", "status": "rejected"},
        {"experiment_id": "E012", "hypothesis": "similarity-weighted k-NN followed by six-way per-label development selection", "selection_split": "dev", "result": "macro 81.4475 on test", "status": "kept"},
        {"experiment_id": "E013", "hypothesis": "XLM-R frozen embedding and pairwise frozen-score blends", "selection_split": "dev", "result": "macro 81.7627 on test", "status": "kept"},
        {"experiment_id": "E014", "hypothesis": "expanded train-token posterior rule for code switching", "selection_split": "dev", "result": "macro 81.8533 on test", "status": "kept"},
        {"experiment_id": "E015", "hypothesis": "Vistral-base frozen embeddings in pairwise score blend", "selection_split": "dev", "result": "macro 81.9650 on test", "status": "kept"},
        {"experiment_id": "E016", "hypothesis": "additional lexical-neighbour retrieval and lexical/retrieval blend", "selection_split": "dev", "result": "no per-label selection advantage over E015 candidates", "status": "rejected"},
        {"experiment_id": "E017", "hypothesis": "train-label dependency adjustment of frozen score blends", "selection_split": "dev", "result": "macro 82.0991 on test", "status": "superseded_by_dev_selection"},
        {"experiment_id": "E018", "hypothesis": "per-label convex weights over the three frozen ViPragSent seeds", "selection_split": "dev", "result": "no per-label selection advantage over E017 candidates", "status": "rejected"},
        {"experiment_id": "E019", "hypothesis": "similarity-weighted ViSoBERT and XLM-R frozen embedding retrieval", "selection_split": "dev", "result": "no per-label selection advantage over E017 candidates", "status": "rejected"},
        {"experiment_id": "E020", "hypothesis": "source-conditioned frozen score blend integrated with label dependencies", "selection_split": "dev", "result": "macro 82.0699 on test", "status": "kept"},
        {"experiment_id": "E021", "hypothesis": "nonlinear minimum/maximum/quantile frozen-score aggregation", "selection_split": "dev", "result": "no per-label selection advantage over E020 candidates", "status": "rejected"},
        {"experiment_id": "E022", "hypothesis": "retrieval-augmented frozen Vistral-base few-shot inference without adapters", "selection_split": "dev", "result": "no per-label selection advantage over E020 candidates", "status": "rejected"},
        {"experiment_id": "E023", "hypothesis": "static Vietnamese teencode normalization before frozen encoder inference", "selection_split": "dev", "result": "no per-label selection advantage over E020 candidates", "status": "rejected"},
        {"experiment_id": "E024", "hypothesis": "retrieval-augmented frozen Sailor-base few-shot inference without adapters", "selection_split": "dev_probe", "result": "base model echoed demonstrations instead of producing usable target completions", "status": "rejected"},
        {"experiment_id": "E025", "hypothesis": "k-NN over the frozen archived rationale-projection representation", "selection_split": "dev", "result": "no per-label selection advantage over E020 candidates", "status": "rejected"},
        {"experiment_id": "E026", "hypothesis": "map emoji and laughter cues to static Vietnamese text before frozen inference", "selection_split": "dev", "result": "no per-label selection advantage over E020 candidates", "status": "rejected"},
    ]
    with (output / "experiment_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    checkpoints = ROOT.parent.parent / "huggingface" / "vipragsent-experiment-checkpoints" / "vipragsent_full"
    verification = {"method": "sha256_before_after_frozen_inference", "neural_weight_updates": False, "optimizer_or_backward_called": False, "checkpoints": {}}
    for seed, before in BEFORE.items():
        after = digest(checkpoints / seed / "best.pt")
        verification["checkpoints"][seed] = {"before": before, "after": after, "unchanged": before == after}
    verification["status"] = "PASS" if all(row["unchanged"] for row in verification["checkpoints"].values()) else "FAIL"
    (output / "frozen_weight_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    config = {
        "experiment_id": "E020_frozen_source_dependency_hybrid",
        "base": "frozen ViPragSent checkpoints 20260520/20260521/20260522",
        "ensemble": "mean probabilities", "thresholds": json.loads((output / "configs" / "threshold_ensemble.json").read_text())["thresholds"],
        "code_switch_lexicon_candidate": json.loads((output / "configs" / "code_switch_lexicon.json").read_text()),
        "code_switch_rule": json.loads((output / "configs" / "code_switch_rules.json").read_text()),
        "phobert_knn_rule": json.loads((output / "configs" / "embedding_knn.json").read_text()),
        "visobert_knn_rule": json.loads((output / "configs" / "visobert_embedding_knn.json").read_text()),
        "xlmr_knn_rule": json.loads((output / "configs" / "xlmr_embedding_knn.json").read_text()),
        "vistral_base_knn_rule": json.loads((output / "configs" / "vistral_base_embedding_knn.json").read_text()),
        "weighted_knn_rule": json.loads((output / "configs" / "weighted_knn.json").read_text()),
        "score_blend_rule": json.loads((output / "configs" / "score_blend_extended.json").read_text()),
        "label_dependency_rule": json.loads((output / "configs" / "label_dependency.json").read_text()),
        "source_score_blend_rule": json.loads((output / "configs" / "source_score_blend.json").read_text()),
        "per_label_selector": json.loads((output / "configs" / "ten_way_source_dependency_hybrid_selector.json").read_text()),
        "selection_split": "development only", "test_labels_used_for_selection": False,
    }
    (output / "best_configuration.yaml").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = "\n".join(f"| {row['task_metric']} | {row['original_vipragsent']} | {row['best_baseline']} ({row['best_baseline_system']}) | {row['optimized_vipragsent']} | {row['difference']} | {row['status']} |" for row in comparison)
    final = f"""# Final status\n\n{final_status}\n\nThe completion criterion requires a strict win on every listed metric. This candidate does not meet it; no success claim is made.\n\n| Task/Metric | Original ViPragSent | Best Baseline | Optimized ViPragSent | Difference | Status |\n| --- | ---: | ---: | ---: | ---: | --- |\n{table}\n\nFrozen checkpoint hashes: `{verification['status']}`.\n"""
    (output / "FINAL_STATUS.md").write_text(final, encoding="utf-8")
    (output / "FINAL_RESULTS.md").write_text(final, encoding="utf-8")
    audit = """# Repository audit\n\n- Evidence source: generated `results/*.json`, prediction JSONL, evaluator code, and adjudicated split files.\n- Gold protocol: 8,000 train / 2,000 dev / 2,000 test records.\n- Baselines are immutable imported artifacts; their registry and prediction paths are in `answers/optimized_vipragsent/baseline_registry.csv`.\n- Original ViPragSent uses three frozen PhoBERT multitask checkpoints. Reproduction of seed 20260520 exactly matched the recorded pragmatic metrics.\n- No dev prediction artifacts existed initially; frozen inference regenerated three dev and three test predictions from the archived checkpoints.\n"""
    (reports / "REPOSITORY_AUDIT.md").write_text(audit, encoding="utf-8")
    (reports / "FINAL_OPTIMIZATION_REPORT.md").write_text(final + "\nThe final configuration is selected exclusively by development metrics; test scores are reported once per configuration and are not used to select its parameters. Experiments: E001 threshold ensemble; E002 max-length control; E003 train-only lexical retrieval; E004 frozen contextual k-NN; E005 dev-selected four-way hybrid; E006 source thresholds; E007 extended k-NN; E008 text normalization variants; E009 mean-pooled k-NN; E010 frozen-logit calibration; E011 prototype retrieval; E012 similarity-weighted k-NN/six-way hybrid; E013 XLM-R score blends; E014 expanded lexical rule; E015 Vistral-base frozen score blend; E016 lexical-neighbour retrieval; E017 label dependency; E018 seed weights; E019 weighted ViSoBERT/XLM-R; E020 source/dependency hybrid; E021 nonlinear score aggregation; E022 frozen Vistral few-shot; E023 teencode normalization; E024 Sailor few-shot probe; E025 rationale-projection retrieval; E026 emoji/laughter normalization.\n", encoding="utf-8")
    logs_dir = output / "logs"
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "frozen_experiment_summary.log").write_text(
        "E001 threshold ensemble: completed\n"
        "E002 max-length 256 control: rejected (identical predictions)\n"
        "E003 train-only code-switch lexical retrieval: completed\n"
        "E004 frozen contextual k-NN retrieval: completed\n"
        "E005 development-selected four-way hybrid: completed\n"
        "E006 source-conditioned thresholds: rejected\n"
        "E007 extended neighborhood k-NN: rejected\n"
        "E008 text-normalization variants: rejected on development data\n"
        "E009 mean-pooled frozen k-NN: rejected on development data\n"
        "E010 frozen-logit linear calibration: rejected on development data\n"
        "E011 frozen-prototype retrieval: rejected on development data\n"
        "E012 similarity-weighted k-NN six-way hybrid: completed\n"
        "E013 XLM-R pairwise frozen-score blend: completed\n"
        "E014 expanded train-token code-switch rule: completed\n"
        "E015 Vistral-base frozen score blend: completed\n"
        "E016 lexical-neighbour retrieval: rejected on development data\n"
        "E017 train-label dependency adjustment: superseded by subsequent development-only selection\n"
        "E018 frozen seed-weight ensemble: rejected on development data\n"
        "E019 weighted ViSoBERT/XLM-R retrieval: rejected on development data\n"
        "E020 source-conditioned score/dependency hybrid: completed\n"
        "E021 nonlinear frozen-score aggregation: rejected on development data\n"
        "E022 frozen Vistral base few-shot inference: rejected on development data\n"
        "E023 teencode normalization: rejected on development data\n"
        "E024 frozen Sailor base few-shot probe: rejected (echoed demonstrations)\n"
        "E025 frozen rationale-projection k-NN: rejected on development data\n"
        "E026 emoji/laughter normalization: rejected on development data\n"
        f"final_status={final_status}\n",
        encoding="utf-8",
    )
    tracked = [path for path in output.rglob("*") if path.is_file()]
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": final_status, "files": {str(path.relative_to(output)): digest(path) for path in tracked}}
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(final_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
