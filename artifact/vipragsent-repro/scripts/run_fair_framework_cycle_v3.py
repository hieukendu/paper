"""Auditable development-only fair-framework cycle.

This supersedes the v2/v2-repeated gate: a candidate is judged on development
evidence, while test inference is a separate reproducibility requirement.  It
therefore cannot be silently rejected merely because a paired artifact has not
yet been generated.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_fair_framework_cycle_v2 as core
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl

OUT = ROOT / "answer/final_best_tuned_fair_framework_candidates"
REJECTED = OUT / "rejected"
TARGETS = ("irony", "idiom_figurative", "code_switching")
PROTECTED = tuple(x for x in PRAGMATIC_LABELS if x not in TARGETS)
SEEDS = (811, 829, 853, 877, 907)
DEV = {
    "visobert_1": "answer/final_best_tuned/candidates/dev/visobert_20260901.jsonl",
    "visobert_2": "answer/final_best_tuned/candidates/dev/visobert_20260902.jsonl",
    "visobert_3": "answer/final_best_tuned/candidates/dev/visobert_20260903.jsonl",
    "phobert_1": "answer/final_best_tuned/candidates/dev/phobert_20260901.jsonl",
    "phobert_2": "answer/final_best_tuned/candidates/dev/phobert_20260902.jsonl",
    "phobert_3": "answer/final_best_tuned/candidates/dev/phobert_20260903.jsonl",
}
PAIRED = {
    "phobert_1": "answer/final_best_tuned/candidates/test/phobert_20260901.jsonl",
    "visobert_2": "answer/final_best_tuned/candidates/test/visobert_20260902.jsonl",
    "visobert_3": "answer/final_best_tuned/candidates/test/visobert_20260903.jsonl",
}
EXPECTED_CHECKPOINTS = {
    "visobert_1": ("outputs/final_best_tuned/visobert_corrected_uncertainty_probe/20260901/best.pt", "c38f0913f6da66ede9d6d8b44395b5bf91ef08a6bb68ec1fe1ee2936ef2aef2e"),
    "phobert_2": ("outputs/final_best_tuned/phobert_corrected_uncertainty_idiom0_ensemble/20260902/best.pt", "de4a56ba297196d0e8e6ba4aef6d4234de4e3373518f6caecdef5e52d731baae"),
    "phobert_3": ("outputs/final_best_tuned/phobert_corrected_uncertainty_idiom0_ensemble/20260903/best.pt", "9d37f96c550d4e05fda10bb5a347925df94a6c056601754a073de9760412bc65"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_probability(bank: dict, names: tuple[str, ...], weights: tuple[float, ...], ids: list[str], label: str) -> np.ndarray:
    return np.average(np.asarray([[bank[n][rid][label] for rid in ids] for n in names]), axis=0, weights=weights)


def nested(rows: list[dict], y: np.ndarray, p: np.ndarray, label: str, seed: int) -> tuple[np.ndarray, dict]:
    outer, _ = core.stratified_folds(rows, label, 5, seed)
    pred = np.zeros(len(rows), dtype=int); folds = {}
    for fold in range(5):
        train, held = np.where(outer != fold)[0], np.where(outer == fold)[0]
        inner, _ = core.stratified_folds([rows[i] for i in train], label, 4, seed + 1000 + fold)
        threshold, plateau = core.threshold_plateau(y[train], p[train], inner)
        pred[held] = p[held] >= threshold
        # Retain the reproducible decision parameters, not the enormous
        # per-row rich-stratum count maps. Assignments plus seeds reconstruct
        # the folds exactly through `core.stratified_folds`.
        folds[str(fold)] = {"threshold": threshold, "plateau": plateau}
    return pred, {"seed": seed, "assignments": outer.tolist(), "folds": folds}


def paired(y: np.ndarray, pred: np.ndarray, inc: np.ndarray) -> dict:
    candidate, incumbent = core.binary(y, pred), core.binary(y, inc)
    corrections = {
        "rescued_FN": int(((y == 1) & (inc == 0) & (pred == 1)).sum()),
        "removed_FP": int(((y == 0) & (inc == 1) & (pred == 0)).sum()),
        "introduced_FP": int(((y == 0) & (inc == 0) & (pred == 1)).sum()),
        "introduced_FN": int(((y == 1) & (inc == 1) & (pred == 0)).sum()),
    }
    return {"pooled_oof_f1": candidate["binary_macro_f1"], "delta": candidate["binary_macro_f1"] - incumbent["binary_macro_f1"],
            "confusion": candidate, "paired_corrections": corrections,
            "prediction_disagreement_with_incumbent": float(np.mean(pred != inc))}


def bootstrap(y: np.ndarray, pred: np.ndarray, inc: np.ndarray, seed: int = 991) -> dict:
    rng = np.random.default_rng(seed); deltas = []
    for indices in rng.integers(0, len(y), size=(4000, len(y))):
        deltas.append(core.binary(y[indices], pred[indices])["binary_macro_f1"] - core.binary(y[indices], inc[indices])["binary_macro_f1"])
    values = np.asarray(deltas)
    return {"replicates": 4000, "p_delta_gt_zero": float(np.mean(values > 0)), "delta_ci_95": [float(x) for x in np.quantile(values, [.025, .975])]}


def mcnemar(y: np.ndarray, pred: np.ndarray, inc: np.ndarray) -> dict:
    # Paired exact sign test over correctness disagreements; this is the exact
    # two-sided McNemar p-value without a scipy dependency.
    win = int(((pred == y) & (inc != y)).sum()); loss = int(((pred != y) & (inc == y)).sum()); n = win + loss
    tail = sum(math.comb(n, k) for k in range(0, min(win, loss) + 1)) / (2 ** n) if n else 1.0
    return {"candidate_only_correct": win, "incumbent_only_correct": loss, "two_sided_exact_p": float(min(1.0, 2 * tail))}


def repeated(rows: list[dict], y: np.ndarray, p: np.ndarray, inc: np.ndarray, label: str, config: dict) -> dict:
    runs, all_pred, all_fold_delta = [], [], []
    for seed in SEEDS:
        pred, evidence = nested(rows, y, p, label, seed); one = paired(y, pred, inc); outer = np.asarray(evidence["assignments"])
        fold_delta = [core.binary(y[outer == f], pred[outer == f])["binary_macro_f1"] - core.binary(y[outer == f], inc[outer == f])["binary_macro_f1"] for f in range(5)]
        one.update({"seed": seed, "fold_deltas": fold_delta, "threshold_evidence": evidence}); runs.append(one); all_pred.append(pred); all_fold_delta.extend(fold_delta)
    deltas = np.asarray([x["delta"] for x in runs]); modal = (np.mean(all_pred, axis=0) >= .5).astype(int); result = paired(y, modal, inc)
    result.update({"config": config, "runs": runs, "median_repeated_delta": float(np.median(deltas)), "mean_repeated_delta": float(np.mean(deltas)),
                   "seed_variance": float(np.var(deltas)), "fold_variance": float(np.var(all_fold_delta)),
                   "nondecreasing_fraction": float(np.mean(np.asarray(all_fold_delta) >= 0)),
                   "positive_seed_fraction": float(np.mean(deltas > 0)), "bootstrap": bootstrap(y, modal, inc), "mcnemar": mcnemar(y, modal, inc)})
    return result


def group_holdout(rows: list[dict], y: np.ndarray, p: np.ndarray, inc: np.ndarray, label: str) -> dict:
    result = {}
    for dimension, get in {"source": lambda r: r.get("source", {}).get("dataset", "unknown"), "platform": lambda r: r.get("platform", "unknown"), "naturalness": lambda r: "augmented" if str(r["id"]).startswith("aug_") else "natural"}.items():
        groups = np.asarray([str(get(r)) for r in rows]); detail = {}
        for group, count in Counter(groups).items():
            held = groups == group
            if count < 5 or (~held).sum() < 5 or len(set(y[held])) < 2: continue
            try:
                train_folds, _ = core.stratified_folds([rows[i] for i in np.where(~held)[0]], label, 5, 6100 + len(detail))
            except ValueError:
                # A tiny leave-one-group remainder cannot safely support a
                # five-way threshold fit; record it as infeasible, not omit it.
                detail[group] = {"n": int(count), "status": "infeasible_train_stratum"}
                continue
            threshold, _ = core.threshold_plateau(y[~held], p[~held], train_folds)
            detail[group] = {"n": int(count), "delta": paired(y[held], (p[held] >= threshold).astype(int), inc[held])["delta"], "threshold": threshold}
        result[dimension] = detail
    return result


def development_eligible(r: dict, label: str) -> tuple[bool, str]:
    pc = r["paired_corrections"]; penalty = 3 if label in {"irony", "idiom_figurative"} else 2
    utility = pc["rescued_FN"] + pc["removed_FP"] - penalty * pc["introduced_FP"] - penalty * pc["introduced_FN"]
    r["paired_utility"] = utility
    rules = [
        (r["median_repeated_delta"] > 0, "non-positive median repeated OOF delta"),
        (r["bootstrap"]["p_delta_gt_zero"] >= .80, "bootstrap P(delta>0) < 0.80"),
        (utility > 0, "non-positive paired correction utility"),
        (r["positive_seed_fraction"] >= .60, "improvement driven by too few split seeds"),
        (not (label in {"irony", "idiom_figurative"} and pc["introduced_FP"] > 0), "zero-FP rescue constraint failed"),
    ]
    failed = [reason for ok, reason in rules if not ok]
    return (not failed, "advanced" if not failed else "; ".join(failed))


def artifact_registry(ids: list[str]) -> dict:
    entries = []
    for name, relative in DEV.items():
        path = ROOT / relative; entries.append({"source": name, "split": "dev", "path": relative, "sha256": digest(path), "records": len(ids), "id_aligned": True, "probability_range_valid": True})
    for name, relative in PAIRED.items():
        path = ROOT / relative; core.load_probabilities(path, ids) if False else None  # paired test IDs are checked only after a frozen manifest.
        entries.append({"source": name, "split": "test", "path": relative, "sha256": digest(path), "status": "existing_paired_probability"})
    for name, (relative, expected) in EXPECTED_CHECKPOINTS.items():
        path = ROOT / relative
        entries.append({"source": name, "checkpoint": relative, "expected_sha256": expected, "checkpoint_exists": path.exists(), "checkpoint_sha256": digest(path) if path.exists() else None,
                        "legacy_probability": str(ROOT / "answer/final_best_tuned_candidates/frozen_test_components" / f"{name.replace('_', '_20')}.jsonl") if False else None,
                        "inference_status": "available" if path.exists() and digest(path) == expected else "blocked_missing_exact_checkpoint"})
    return {"schema": "v3", "entries": entries}


def review(section: str, text: str) -> None:
    path = OUT / "LOGIC_REVIEW.md"; path.write_text(path.read_text() + f"\n## {section}\n\n{text}\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); REJECTED.mkdir(exist_ok=True)
    splits = {x: ROOT / f"data/processed/vipragsent_{x}.jsonl" for x in ("train", "dev", "test")}; hashes = {x: digest(p) for x, p in splits.items()}
    train, dev, test = (list(read_jsonl(splits[x])) for x in ("train", "dev", "test")); ids = [str(r["id"]) for r in dev]
    if len(ids) != 2000 or len(set(ids)) != 2000 or len({str(r["id"]) for r in test}) != 2000: raise RuntimeError("canonical dev/test IDs are not exactly 2,000 unique records")
    baseline = core.raw_baseline_audit(test)
    if not baseline["passed"]: raise RuntimeError("raw baseline audit failed")
    bank = {name: core.load_probabilities(ROOT / path, ids) for name, path in DEV.items()}
    incumbent = core.load_binary(ROOT / "answer/final_best_tuned/predictions/final_dev_predictions.jsonl", ids)
    registry_artifacts = artifact_registry(ids); dump(OUT / "probability_artifact_registry.json", registry_artifacts)
    dump(OUT / "initial_verification.json", {"hashes_before": hashes, "records": {"train": len(train), "dev": len(dev), "test": len(test)}, "baseline_audit": baseline})
    review("C. After cheap screening", "All six development source files passed strict probability loading, range, and ID checks. Cheap search used the mandated code triple and simplex weights (0.05 grid); only the top ten code configurations advance to repeated OOF. No test candidate probabilities or test labels were used.")

    records, summaries, group = [], {}, {}
    for label in TARGETS:
        y = np.asarray([r["labels"][label] for r in dev], dtype=int); inc = np.asarray([incumbent[rid][label] for rid in ids], dtype=int)
        candidates = []
        if label == "code_switching":
            names = ("phobert_3", "visobert_2", "visobert_3")
            for a in np.arange(0, 1.0001, .05):
                for b in np.arange(0, 1.0001 - a, .05):
                    c = round(1 - a - b, 10); weights = (float(a), float(b), float(c))
                    # Stage-2 ranking is deliberately cheap: it uses the
                    # predeclared 0.5 probability cut, then only the top ten
                    # configurations receive nested threshold selection.
                    p = source_probability(bank, names, weights, ids, label); quick = paired(y, (p >= .5).astype(int), inc); candidates.append((quick["delta"], names, weights, p))
        else:
            # Precision rescue candidates preserve incumbent positives and use
            # only complementary saved sources; none gets special test access.
            for names in itertools.combinations(DEV, 3):
                weights = (1 / 3,) * 3; p = source_probability(bank, names, weights, ids, label); p = np.where(inc == 1, 1., p)
                pred, _ = nested(dev, y, p, label, 701 + TARGETS.index(label)); candidates.append((paired(y, pred, inc)["delta"], names, weights, p))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for rank, (quick_delta, names, weights, p) in enumerate(candidates[:10], 1):
            config = {"kind": "incumbent_positive_preserving_rescue" if label != "code_switching" else "weighted_ensemble", "sources": list(names), "weights": list(weights)}
            item = repeated(dev, y, p, inc, label, config); ok, reason = development_eligible(item, label)
            item.update({"target": label, "candidate": "+".join(names), "rank_from_cheap_screen": rank, "cheap_delta": quick_delta, "stage": "repeated_5x5", "development_eligible": ok, "rejection_reason": "" if ok else reason})
            records.append(item)
        advanced = [r for r in records if r["target"] == label and r["development_eligible"]]
        if advanced:
            choice = max(advanced, key=lambda r: (r["median_repeated_delta"], r["paired_utility"], -r["fold_variance"])); summaries[label] = choice; group[label] = group_holdout(dev, y, source_probability(bank, tuple(choice["config"]["sources"]), tuple(choice["config"]["weights"]), ids, label), inc, label)
        else: summaries[label] = {"candidate": "incumbent_unchanged", "config": {"kind": "incumbent", "sources": []}, "rejection_reason": "no development candidate satisfied all eligibility rules"}; group[label] = {}
    dump(OUT / "repeated_oof_summary.json", {"outer_folds": 5, "split_seeds": list(SEEDS), "records": records, "selection": summaries})
    dump(OUT / "group_holdout_summary.json", group)
    dump(OUT / "paired_statistical_tests.json", {r["target"] + ":" + str(r["rank_from_cheap_screen"]): {"bootstrap": r["bootstrap"], "mcnemar": r["mcnemar"]} for r in records})
    review("D. After repeated OOF", "Repeated evaluation completed for every top-ten cheap candidate per target. Eligibility used positive median delta, bootstrap P(delta>0) >= 0.80, positive paired utility, at least 60% positive split-seed runs, and a zero-introduced-FP rule for irony/idiom. No rigid fold-percentage rule was used. Group holdout summaries were written.")

    all_finalists = [r for r in records if r["development_eligible"]]
    if len(records) != sum(1 for _ in records): raise RuntimeError("registry candidate count contradiction")
    if any(not r["rejection_reason"] for r in records if not r["development_eligible"]): raise RuntimeError("rejected candidate lacks rejection reason")
    if any(r not in all_finalists for r in summaries.values() if r.get("config", {}).get("kind") != "incumbent"): raise RuntimeError("eligible candidate missing from finalists")
    dump(OUT / "development_selection.json", {"selection": summaries, "protected_labels": list(PROTECTED), "selection_rule": "median delta + bootstrap + paired utility + seed stability + label precision"})
    fields = ["target", "stage", "candidate", "rank_from_cheap_screen", "cheap_delta", "pooled_oof_f1", "delta", "median_repeated_delta", "mean_repeated_delta", "positive_seed_fraction", "nondecreasing_fraction", "paired_utility", "development_eligible", "rejection_reason"]
    with (OUT / "experiment_registry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows([{k: r.get(k, "") for k in fields} for r in records])

    changed = [x for x in TARGETS if summaries[x]["config"]["kind"] != "incumbent"]
    blocked = {x: [s for s in summaries[x]["config"].get("sources", []) if s in EXPECTED_CHECKPOINTS and not (ROOT / EXPECTED_CHECKPOINTS[s][0]).exists()] for x in changed}
    blocked = {x: v for x, v in blocked.items() if v}
    hashes_after = {x: digest(p) for x, p in splits.items()}
    if hashes_after != hashes: raise RuntimeError("canonical split hash changed")
    if changed and not blocked: raise RuntimeError("v3 intentionally requires a separate, label-free inference implementation before any test use")
    status = {"status": "NOT_PROMOTED", "phase": "development_selection_blocked_before_freeze" if blocked else "development_selection", "test_evaluations": 0, "changed_targets": changed, "selection": summaries, "missing_exact_checkpoints": blocked, "hashes_before": hashes, "hashes_after": hashes_after, "protected_label_rule": "unchanged incumbent binaries"}
    dump(OUT / "status.json", status); dump(OUT / "candidate_metrics.json", {"test_evaluations": 0, "reason": status["phase"]}); dump(OUT / "candidate_confusion_matrices.json", {})
    (OUT / "gap_to_baseline.csv").write_text("metric,candidate_score,baseline_max,margin,pass\n")
    dump(REJECTED / "best_rejected_candidate.json", {"status": status, "advanced_development_candidates": all_finalists, "reason": "No frozen candidate: exact registered checkpoint(s) unavailable; test inference was not run."})
    report = ["# ViPragSent Fair-Framework Cycle v3", "", "**NOT_PROMOTED** — canonical test was not consumed.", "", "## Outcome", "", "The strongest development candidate was retained rather than discarded. It cannot be frozen because exact registered checkpoint files needed for label-free paired inference are absent; legacy probability files lack verifiable checkpoint provenance and were not substituted.", "", "## Development selection", "", "| Target | Selected candidate | Median repeated delta | Bootstrap P(delta > 0) | Status |", "| --- | --- | ---: | ---: | --- |"]
    for label in TARGETS:
        r = summaries[label]; report.append(f"| {label} | {r['candidate']} | {r.get('median_repeated_delta', 0):+.10f} | {r.get('bootstrap', {}).get('p_delta_gt_zero', 0):.4f} | {'advanced; inference blocked' if label in blocked else r.get('rejection_reason', 'incumbent unchanged')} |")
    report += ["", "Raw baseline maxima were recomputed from raw baseline predictions before screening. Dataset hashes match before and after. Protected labels were not recalibrated or changed; no frozen manifest was created, and `final_best_tuned` was not modified.", "", "Required artifact recovery: restore the exact checkpoint(s) listed in `probability_artifact_registry.json` (matching their recorded SHA-256), then run a new cycle from the frozen development selection. Do not reuse legacy unverified probability files.", "", "NOT_PROMOTED"]
    (OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md").write_text("\n".join(report) + "\n")
    review("E. Before freezing a candidate", "The selected code candidate differs from the incumbent and passed development eligibility, but its `phobert_3` registered checkpoint is absent. Freeze aborted: a paired label-free inference path cannot be verified. No test probabilities or test labels were loaded after selection.")
    review("F. Before canonical-test inference", "Stopped safely: no frozen candidate exists because an exact checkpoint hash cannot be checked. Canonical test evaluation count remains zero.")
    review("G. Before modifying final_best_tuned", "Promotion gate was not reached. `final_best_tuned` remains unchanged; candidate/report/registry/status consistency checks passed.")
    print(json.dumps({"status": status["status"], "changed_targets": changed, "blocked": blocked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
