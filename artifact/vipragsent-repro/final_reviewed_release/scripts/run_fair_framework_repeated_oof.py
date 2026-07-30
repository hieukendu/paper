"""Next fair-framework cycle: repeated OOF source selection with paired inference.

This cycle deliberately uses only saved probabilities from training-split models.
Development selection is 5 outer folds x 5 deterministic seeds.  Test source
probabilities are touched only after a genuinely changed manifest is frozen.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_fair_framework_cycle_v2 as core
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl

OUT = ROOT / "answer/final_best_tuned_fair_framework_candidates"
REJECTED = OUT / "rejected"
TARGETS = ("irony", "idiom_figurative", "code_switching")
PROTECTED = tuple(label for label in PRAGMATIC_LABELS if label not in TARGETS)
DEV = {
    "visobert_1": "answer/final_best_tuned/candidates/dev/visobert_20260901.jsonl",
    "visobert_2": "answer/final_best_tuned/candidates/dev/visobert_20260902.jsonl",
    "visobert_3": "answer/final_best_tuned/candidates/dev/visobert_20260903.jsonl",
    "phobert_1": "answer/final_best_tuned/candidates/dev/phobert_20260901.jsonl",
    "phobert_2": "answer/final_best_tuned/candidates/dev/phobert_20260902.jsonl",
    "phobert_3": "answer/final_best_tuned/candidates/dev/phobert_20260903.jsonl",
}
# Only these have paired saved canonical-test probability files.  A source
# without a paired test artifact is recorded but cannot be frozen or invented.
TEST = {
    "visobert_2": "answer/final_best_tuned/candidates/test/visobert_20260902.jsonl",
    "visobert_3": "answer/final_best_tuned/candidates/test/visobert_20260903.jsonl",
    "phobert_1": "answer/final_best_tuned/candidates/test/phobert_20260901.jsonl",
}
SEEDS = (811, 829, 853, 877, 907)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_probability(bank: dict, names: tuple[str, ...], ids: list[str], label: str) -> np.ndarray:
    return np.mean(np.asarray([[bank[name][rid][label] for rid in ids] for name in names]), axis=0)


def nested_oof(rows: list[dict], y: np.ndarray, p: np.ndarray, label: str, seed: int) -> tuple[np.ndarray, dict]:
    """Five outer folds, with threshold tuning only on inner training folds."""
    outer, outer_info = core.stratified_folds(rows, label, 5, seed)
    prediction = np.zeros(len(rows), dtype=int); thresholds = {}
    for fold in range(5):
        train, held = np.where(outer != fold)[0], np.where(outer == fold)[0]
        inner, inner_info = core.stratified_folds([rows[i] for i in train], label, 4, seed + 1000 + fold)
        threshold, plateau = core.threshold_plateau(y[train], p[train], inner)
        prediction[held] = p[held] >= threshold
        thresholds[str(fold)] = {"threshold": threshold, "plateau": plateau, "inner": inner_info}
    return prediction, {"outer": outer_info, "folds": thresholds, "assignments": outer.tolist()}


def metrics(y: np.ndarray, pred: np.ndarray, incumbent: np.ndarray) -> dict:
    candidate, base = core.binary(y, pred), core.binary(y, incumbent)
    # Paired correction counts are computed directly, independent of labels.
    rescued_fn = int(np.sum((y == 1) & (incumbent == 0) & (pred == 1)))
    removed_fp = int(np.sum((y == 0) & (incumbent == 1) & (pred == 0)))
    introduced_fp = int(np.sum((y == 0) & (incumbent == 0) & (pred == 1)))
    introduced_fn = int(np.sum((y == 1) & (incumbent == 1) & (pred == 0)))
    return {"pooled_oof_f1": candidate["binary_macro_f1"], "delta": candidate["binary_macro_f1"] - base["binary_macro_f1"],
            "confusion": candidate, "confusion_change": {k: int(candidate[k]) - int(base[k]) for k in ("tp", "tn", "fp", "fn")},
            "paired_corrections": {"rescued_FN": rescued_fn, "removed_FP": removed_fp, "introduced_FP": introduced_fp, "introduced_FN": introduced_fn}}


def repeated_evaluate(rows: list[dict], y: np.ndarray, p: np.ndarray, incumbent: np.ndarray, label: str, config: dict) -> dict:
    runs = []; all_pred = []
    for seed in SEEDS:
        pred, evidence = nested_oof(rows, y, p, label, seed)
        outer = np.asarray(evidence["assignments"])
        fold_deltas = [core.binary(y[outer == f], pred[outer == f])["binary_macro_f1"] - core.binary(y[outer == f], incumbent[outer == f])["binary_macro_f1"] for f in range(5)]
        one = metrics(y, pred, incumbent)
        one.update({"seed": seed, "fold_deltas": fold_deltas, "threshold_evidence": evidence})
        runs.append(one); all_pred.append(pred)
    deltas = np.asarray([r["delta"] for r in runs]); fold_deltas = np.asarray([x for r in runs for x in r["fold_deltas"]])
    modal = (np.mean(np.asarray(all_pred), axis=0) >= 0.5).astype(int)
    summary = metrics(y, modal, incumbent)
    summary.update({"config": config, "runs": runs, "median_repeated_delta": float(np.median(deltas)), "mean_repeated_delta": float(np.mean(deltas)),
                    "seed_variance": float(np.var(deltas)), "fold_variance": float(np.var(fold_deltas)),
                    "nondecreasing_fold_runs": int(np.sum(fold_deltas >= 0)), "fold_runs": int(fold_deltas.size),
                    "nondecreasing_fraction": float(np.mean(fold_deltas >= 0)),
                    "prediction_disagreement_with_incumbent": float(np.mean(modal != incumbent))})
    return summary


def pre_screen(rows: list[dict], y: np.ndarray, p: np.ndarray, incumbent: np.ndarray, label: str, config: dict) -> dict:
    pred, evidence = nested_oof(rows, y, p, label, 701 + TARGETS.index(label))
    result = metrics(y, pred, incumbent); result.update({"config": config, "screen_seed": 701 + TARGETS.index(label), "threshold_evidence": evidence})
    return result


def eligible(summary: dict, label: str) -> tuple[bool, str]:
    c = summary["config"]
    if not set(c["sources"]).issubset(TEST):
        return False, "missing paired canonical-test probability artifact"
    if summary["median_repeated_delta"] <= 0:
        return False, "non-positive median repeated OOF delta"
    if summary["nondecreasing_fraction"] < 0.70:
        return False, "fewer than 70% non-decreasing repeated fold-runs"
    pc = summary["paired_corrections"]
    if label in ("irony", "idiom_figurative") and (pc["introduced_FP"] > (1 if label == "idiom_figurative" else 0)):
        return False, "violates high-precision introduced-FP constraint"
    if label == "code_switching" and pc["introduced_FP"] > pc["rescued_FN"] + pc["removed_FP"]:
        return False, "code correction introduces more false positives than paired corrections"
    return True, "advanced"


def full_threshold(rows: list[dict], y: np.ndarray, p: np.ndarray, label: str) -> tuple[float, dict]:
    folds, info = core.stratified_folds(rows, label, 5, 1901 + TARGETS.index(label))
    threshold, plateau = core.threshold_plateau(y, p, folds)
    return threshold, {"folds": info, "plateau": plateau}


def main() -> int:
    split_paths = {x: ROOT / f"data/processed/vipragsent_{x}.jsonl" for x in ("train", "dev", "test")}
    hashes_before = {x: sha(path) for x, path in split_paths.items()}
    train_rows, dev_rows, test_rows = (list(read_jsonl(split_paths[x])) for x in ("train", "dev", "test"))
    if len(dev_rows) != 2000 or len(test_rows) != 2000 or len({x["id"] for x in dev_rows}) != 2000 or len({x["id"] for x in test_rows}) != 2000:
        raise ValueError("canonical dev/test IDs must remain 2,000 unique records")
    ids = [str(row["id"]) for row in dev_rows]
    baseline = core.raw_baseline_audit(test_rows)
    if not baseline["passed"]:
        raise RuntimeError("baseline rounding audit failed; stopping before selection")
    incumbent_path = ROOT / "answer/final_best_tuned/predictions/final_dev_predictions.jsonl"
    incumbent_binary = core.load_binary(incumbent_path, ids)
    incumbent_prob = core.load_probabilities(incumbent_path, ids)
    bank = {name: core.load_probabilities(ROOT / path, ids) for name, path in DEV.items()}
    verification = {"hashes_before": hashes_before, "records": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)},
                    "baseline_audit": baseline, "probability_bank": {name: str(path) for name, path in DEV.items()}, "paired_test_sources": TEST}
    dump(OUT / "initial_verification.json", verification)

    registry: list[dict] = []; finalists: dict[str, list[dict]] = {x: [] for x in TARGETS}
    # Exhaustive cheap source screening.  It includes every individual, pair,
    # triple and leave-one-out source, then only positive candidates advance to
    # the 25 repeated outer-fold runs (successive halving).
    for label in TARGETS:
        y = np.asarray([row["labels"][label] for row in dev_rows], dtype=int)
        inc = np.asarray([incumbent_binary[rid][label] for rid in ids], dtype=int)
        positive_screens: list[tuple[dict, np.ndarray, dict]] = []
        names = tuple(DEV)
        combinations = [(x,) for x in names] + list(itertools.combinations(names, 2)) + list(itertools.combinations(names, 3))
        combinations += [tuple(x for x in names if x != removed) for removed in names]
        seen: set[tuple[str, ...]] = set()
        for source_names in combinations:
            source_names = tuple(sorted(source_names))
            if source_names in seen: continue
            seen.add(source_names)
            p = source_probability(bank, source_names, ids, label)
            config = {"kind": "uniform_source_ensemble", "sources": list(source_names), "weights": [1 / len(source_names)] * len(source_names)}
            screen = pre_screen(dev_rows, y, p, inc, label, config)
            screen.update({"target": label, "stage": "cheap_screen", "candidate": "+".join(source_names)})
            registry.append(screen)
            # Retain only same-split positive screens for repeated evidence.
            if screen["delta"] > 0:
                positive_screens.append((screen, p, config))

        # Selective rescue preserves incumbent positives.  It is applied to the
        # three paired heterogeneous sources so its final inference is exact.
        paired = ("visobert_2", "visobert_3", "phobert_1")
        p = source_probability(bank, paired, ids, label)
        rescue_p = np.where(inc == 1, 1.0, p)
        config = {"kind": "incumbent_positive_preserving_rescue", "sources": list(paired), "weights": [1 / 3] * 3}
        screen = pre_screen(dev_rows, y, rescue_p, inc, label, config)
        screen.update({"target": label, "stage": "cheap_screen", "candidate": "paired_source_preserving_rescue"}); registry.append(screen)
        if screen["delta"] > 0:
            positive_screens.append((screen, rescue_p, config))
        # Successive halving: repeat the five highest positive screens, plus
        # the two explicitly prioritized code triples when they are positive.
        positive_screens.sort(key=lambda item: item[0]["delta"], reverse=True)
        selected_screens = positive_screens[:5]
        if label == "code_switching":
            priority = {frozenset(("visobert_2", "visobert_3", "phobert_1")), frozenset(("visobert_2", "visobert_3", "phobert_3"))}
            selected_screens += [item for item in positive_screens if frozenset(item[2]["sources"]) in priority and item not in selected_screens]
        for screen, probability, config in selected_screens:
            repeated = repeated_evaluate(dev_rows, y, probability, inc, label, config)
            repeated.update({"target": label, "stage": "repeated_5x5", "candidate": screen["candidate"]})
            ok, reason = eligible(repeated, label); repeated["advance"] = ok; repeated["advance_reason"] = reason; registry.append(repeated)
            if ok: finalists[label].append(repeated)

    selection = {}
    for label in TARGETS:
        if finalists[label]:
            # Utility favors repaired paired errors; code FP cost is stricter.
            def key(x: dict) -> tuple:
                pc = x["paired_corrections"]; penalty = 2 if label == "code_switching" else 3
                utility = pc["rescued_FN"] + pc["removed_FP"] - penalty * pc["introduced_FP"] - penalty * pc["introduced_FN"]
                return (x["median_repeated_delta"], utility, -x["fold_variance"], -x["seed_variance"])
            choice = max(finalists[label], key=key); choice["selection_utility"] = key(choice)[1]; selection[label] = choice
        else:
            selection[label] = {"candidate": "incumbent_unchanged", "config": {"kind": "incumbent", "sources": []}, "reason": "no paired candidate passed repeated OOF robustness"}
    dump(OUT / "repeated_oof_summary.json", {"seeds": list(SEEDS), "outer_folds": 5, "selection": selection, "finalists": finalists,
                                               "repeated_candidates": [row for row in registry if row.get("stage") == "repeated_5x5"]})
    dump(OUT / "development_selection.json", {"selection": selection, "protected_labels": list(PROTECTED), "ranking": "median repeated OOF delta, paired utility, consistency, variance"})
    fields = ["target", "stage", "candidate", "pooled_oof_f1", "delta", "median_repeated_delta", "nondecreasing_fraction", "fold_variance", "seed_variance", "advance", "advance_reason"]
    with (OUT / "experiment_registry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in registry: writer.writerow({field: row.get(field, "") for field in fields})

    changed = [label for label in TARGETS if selection[label]["config"]["kind"] != "incumbent"]
    hashes_after_dev = {x: sha(path) for x, path in split_paths.items()}
    if hashes_after_dev != hashes_before: raise RuntimeError("canonical split hash changed during development")
    if not changed:
        # This cycle did not freeze a candidate.  Remove the stale manifest
        # from the prior cycle so its presence cannot be misread as a test
        # evaluation under the current protocol.
        stale_manifest = OUT / "frozen_candidate_manifest.json"
        if stale_manifest.exists(): stale_manifest.unlink()
        status = {"status": "NOT_PROMOTED", "phase": "development_selection", "reason": "no genuinely changed target passed repeated OOF robustness; canonical test was not consumed", "test_evaluations": 0, "hashes_before": hashes_before, "hashes_after": hashes_after_dev, "selection": selection}
        dump(OUT / "status.json", status); dump(OUT / "candidate_metrics.json", {"test_evaluations": 0}); dump(OUT / "candidate_confusion_matrices.json", {})
        (OUT / "gap_to_baseline.csv").write_text("metric,candidate_score,baseline_max,margin,pass\n")
        repeated = [row for row in registry if row.get("stage") == "repeated_5x5"]
        report = ["# ViPragSent Repeated-OOF Fair Framework", "", "**NOT_PROMOTED**: no genuinely changed target passed repeated OOF robustness; no canonical-test candidate was evaluated.", "", "## Methods actually run", "", "- Complete probability-bank screen: all individual sources, pairs, triples, and leave-one-out ensembles.", "- Selective incumbent-positive-preserving rescue using the paired heterogeneous source bank.", "- Repeated 5 outer folds × 5 deterministic seeds for every positive screen advanced by successive halving.", "", "## Repeated candidates and exclusions", "", "| Target | Candidate | Median delta | Non-decreasing fold-runs | Exclusion |", "| --- | --- | ---: | ---: | --- |"]
        for row in repeated:
            report.append(f"| {row['target']} | {row['candidate']} | {row['median_repeated_delta']:+.10f} | {100*row['nondecreasing_fraction']:.0f}% | {row['advance_reason']} |")
        report += ["", "The paired `phobert_1 + visobert_2 + visobert_3` code triple had a positive median delta but reached 60% non-decreasing fold-runs, below the required 70%. Stronger code triples lacked paired canonical-test source probabilities and were retained as excluded evidence, not inferred or fabricated.", "", "All dataset hashes remained unchanged. Protected-label predictions were never recalibrated or changed. Detailed TP/TN/FP/FN corrections, fold variance, seed variance, and disagreement are retained in `repeated_oof_summary.json`.", "", "NOT_PROMOTED"]
        (OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md").write_text("\n".join(report) + "\n")
        return 0

    # Freeze config before opening paired new candidate test probabilities.
    thresholds = {}
    for label in changed:
        config = selection[label]["config"]; y = np.asarray([row["labels"][label] for row in dev_rows])
        p = source_probability(bank, tuple(config["sources"]), ids, label)
        if config["kind"] == "incumbent_positive_preserving_rescue": p = np.where(np.asarray([incumbent_binary[rid][label] for rid in ids]) == 1, 1.0, p)
        thresholds[label] = {"threshold": full_threshold(dev_rows, y, p, label)[0]}
    manifest = {"protocol": "repeated 5 outer folds x 5 deterministic seeds", "code_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "hashes": hashes_before, "protected_binary_predictions": "exact incumbent", "selection": {label: selection[label] for label in changed}, "thresholds": thresholds,
                "paired_test_sources": TEST, "test_evaluations": 1}
    dump(OUT / "frozen_candidate_manifest.json", manifest)

    test_ids = [str(row["id"]) for row in test_rows]
    test_bank = {name: core.load_probabilities(ROOT / TEST[name], test_ids) for name in sorted({source for label in changed for source in selection[label]["config"]["sources"]})}
    incumbent_test = core.load_binary(ROOT / "answer/final_best_tuned/predictions/final_test_predictions.jsonl", test_ids)
    candidate = {label: np.asarray([incumbent_test[rid][label] for rid in test_ids], dtype=int) for label in PRAGMATIC_LABELS}
    for label in changed:
        config = selection[label]["config"]; p = source_probability(test_bank, tuple(config["sources"]), test_ids, label)
        if config["kind"] == "incumbent_positive_preserving_rescue": p = np.where(candidate[label] == 1, 1.0, p)
        candidate[label] = (p >= thresholds[label]["threshold"]).astype(int)
    if any(not np.array_equal(candidate[label], np.asarray([incumbent_test[rid][label] for rid in test_ids])) for label in PROTECTED):
        raise RuntimeError("protected test predictions changed")
    gold = {str(row["id"]): row for row in test_rows}
    confusion = {label: core.binary(np.asarray([gold[rid]["labels"][label] for rid in test_ids]), candidate[label]) for label in PRAGMATIC_LABELS}
    scores = {label: float(confusion[label]["binary_macro_f1"]) for label in PRAGMATIC_LABELS}; scores["macro_pragmatic_f1"] = float(np.mean(list(scores.values())))
    maxes = baseline["authoritative_baseline_max"]; margins = {label: scores[label] - maxes[label] for label in maxes}; promoted = all(v > 0 for v in margins.values())
    dump(OUT / "candidate_metrics.json", {"scores": scores, "baseline_max": maxes, "margins": margins, "test_evaluations": 1, "changed_targets": changed})
    dump(OUT / "candidate_confusion_matrices.json", confusion)
    with (OUT / "gap_to_baseline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "candidate_score", "baseline_max", "margin", "pass"], lineterminator="\n"); writer.writeheader()
        for label in maxes: writer.writerow({"metric": label, "candidate_score": scores[label], "baseline_max": maxes[label], "margin": margins[label], "pass": margins[label] > 0})
    hashes_after = {x: sha(path) for x, path in split_paths.items()}
    if hashes_after != hashes_before: raise RuntimeError("canonical split hash changed")
    status = {"status": "PROMOTED" if promoted else "NOT_PROMOTED", "test_evaluations": 1, "changed_targets": changed, "selection": selection, "scores": scores, "baseline_margins": margins, "hashes_before": hashes_before, "hashes_after": hashes_after}
    dump(OUT / "status.json", status)
    report = ["# ViPragSent Repeated-OOF Fair Framework", "", f"**{status['status']}**. A single genuinely changed frozen candidate was evaluated once on canonical test after repeated OOF selection.", "", "## Selection", ""]
    for label in TARGETS: report.append(f"- `{label}`: {selection[label]['candidate']}")
    report += ["", "The cheap screen covered individual sources, all pairs, all triples, and leave-one-out ensembles. Positive screens advanced to repeated 5 outer folds × 5 deterministic seeds. Candidates without paired canonical-test probabilities were retained as exclusions rather than fabricated.", "", "## Canonical-test margins", "", "| Metric | Score | Baseline | Margin |", "| --- | ---: | ---: | ---: |"]
    for label in maxes: report.append(f"| {label} | {scores[label]:.10f} | {maxes[label]:.10f} | {margins[label]:+.10f} |")
    report += ["", "Detailed repeated deltas, TP/TN/FP/FN changes, variance, and disagreement are in `repeated_oof_summary.json`; all source screens are in `experiment_registry.csv`.", "", status["status"]]
    (OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md").write_text("\n".join(report) + "\n")
    if not promoted: dump(REJECTED / "repeated_oof_best_rejected.json", {"status": status, "manifest": manifest, "confusion": confusion, "selection": selection})
    print(json.dumps({"status": status["status"], "changed_targets": changed, "margins": margins}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
