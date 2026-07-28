"""Leakage-safe, development-only anchor/arbiter cycle.

This is the sole writer for the ``anchor-arbiter-cycle-v1`` state.  It never
opens canonical-test data.  A source is eligible only when it is an aligned
five-fold OOF probability file with an accompanying provenance manifest.  The
meta models are evaluated in a second, nested five-fold split of the canonical
development records; this is an experimental screen, *not* permission to
promote a candidate.  Promotion additionally requires canonical-train OOF,
label-free inference, and an untouched development confirmation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.utils.io import read_jsonl

OUT = ROOT / "answer" / "final_best_tuned_fair_framework_candidates"
RUN_ID = "anchor-arbiter-cycle-v1"
TARGETS = ("irony", "idiom_figurative", "code_switching")
SEEDS = (701, 1701, 2701)
SOURCES = {
    "incumbent": "answer/final_best_tuned_candidates/oof/existing_sources/incumbent.oof.jsonl",
    "pho1": "answer/final_best_tuned_candidates/oof/existing_sources/pho1.oof.jsonl",
    "pho2": "answer/final_best_tuned_candidates/oof/existing_sources/pho2.oof.jsonl",
    "pho3": "answer/final_best_tuned_candidates/oof/existing_sources/pho3.oof.jsonl",
    "pho3ens": "answer/final_best_tuned_candidates/oof/existing_sources/pho3ens.oof.jsonl",
    "viso1": "answer/final_best_tuned_candidates/oof/existing_sources/viso1.oof.jsonl",
    "viso2": "answer/final_best_tuned_candidates/oof/existing_sources/viso2.oof.jsonl",
    "viso3": "answer/final_best_tuned_candidates/oof/existing_sources/viso3.oof.jsonl",
    "viso3ens": "answer/final_best_tuned_candidates/oof/existing_sources/viso3ens.oof.jsonl",
    "target_cross": "answer/final_best_tuned_candidates/oof/target_mixtures/target_cross.oof.jsonl",
    # This is a historical OOF source used solely as an alternate-screen
    # anchor.  It is deliberately *not* the restored strong code candidate.
    "code_screen_anchor": "answer/final_best_tuned_candidates/oof/code_specialist_3seed/targetcross_codeens.oof.jsonl",
}
ANCHORS = {
    "irony": "incumbent",
    "idiom_figurative": "pho3ens",
    "code_switching": "code_screen_anchor",
}
ALTERNATES = {
    "irony": ("pho1", "pho2", "pho3", "viso1", "viso2", "viso3", "target_cross"),
    "idiom_figurative": ("pho1", "pho2", "pho3", "viso1", "viso2", "viso3", "target_cross"),
    "code_switching": ("incumbent", "pho3ens", "viso3ens", "target_cross"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return {"binary_macro_f1": float(f1_score(y, p, average="macro", zero_division=0) * 100), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def corrections(y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict:
    return {
        "prediction_disagreement_count": int((anchor != candidate).sum()),
        "rescued_FN": int(((y == 1) & (anchor == 0) & (candidate == 1)).sum()),
        "removed_FP": int(((y == 0) & (anchor == 1) & (candidate == 0)).sum()),
        "introduced_FP": int(((y == 0) & (anchor == 0) & (candidate == 1)).sum()),
        "introduced_FN": int(((y == 1) & (anchor == 1) & (candidate == 0)).sum()),
    }


def best_threshold(y: np.ndarray, score: np.ndarray, anchor: np.ndarray, *, conservative: bool) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0, 1, 161)))
    ranked = []
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        if conservative:
            # The arbiter never overrides an anchor positive on precision-first targets.
            pred = np.maximum(anchor, pred)
        m = metric(y, pred)
        utility = corrections(y, anchor, pred)["rescued_FN"] + corrections(y, anchor, pred)["removed_FP"] - 5 * (corrections(y, anchor, pred)["introduced_FP"] + corrections(y, anchor, pred)["introduced_FN"])
        ranked.append((utility, m["binary_macro_f1"], -abs(float(threshold) - .5), float(threshold)))
    return max(ranked)[-1]


def features(probabilities: np.ndarray) -> np.ndarray:
    # Values are source probabilities for the one target.  Ranks/disagreement
    # add only prediction information, never text labels.
    mean = probabilities.mean(axis=1, keepdims=True)
    std = probabilities.std(axis=1, keepdims=True)
    span = (probabilities.max(axis=1) - probabilities.min(axis=1))[:, None]
    vote = (probabilities >= .5).mean(axis=1, keepdims=True)
    return np.hstack((probabilities, mean, std, span, vote))


def nonnegative_stacker_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Convex probability stacker, fitted only in each outer fold.
    def objective(w: np.ndarray) -> float:
        q = np.clip(x @ w, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    result = minimize(objective, np.full(x.shape[1], 1 / x.shape[1]), bounds=[(0, 1)] * x.shape[1], constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
    if not result.success:
        return np.full(x.shape[1], 1 / x.shape[1])
    return result.x


def nested_eval(name: str, family: str, x: np.ndarray, source_p: np.ndarray, anchor_p: np.ndarray, anchor_pred: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    pred = np.zeros_like(y); scores = np.zeros_like(y, dtype=float)
    thresholds: dict[str, float] = {}
    for fold in range(5):
        train, held = np.where(folds != fold)[0], np.where(folds == fold)[0]
        if family == "nonnegative_probability_stacker":
            weights = nonnegative_stacker_fit(source_p[train], y[train]); train_score = source_p[train] @ weights; held_score = source_p[held] @ weights
        else:
            scaler = StandardScaler().fit(x[train]); xt, xh = scaler.transform(x[train]), scaler.transform(x[held])
            if family == "conservative_logistic":
                model = LogisticRegression(C=.15, class_weight="balanced", max_iter=2000, random_state=seed + fold)
            elif family == "conservative_gradient_boosted":
                model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7, l2_regularization=3.0, learning_rate=.06, random_state=seed + fold)
            else:
                raise ValueError(family)
            model.fit(xt, y[train]); train_score = model.predict_proba(xt)[:, 1]; held_score = model.predict_proba(xh)[:, 1]
        threshold = best_threshold(y[train], train_score, anchor_pred[train], conservative=True)
        pred[held] = np.maximum(anchor_pred[held], held_score >= threshold)
        scores[held] = held_score; thresholds[str(fold)] = threshold
    return pred.astype(int), {"candidate": name, "family": family, "seed": seed, "fold_thresholds": thresholds}


def topk_eval(name: str, source_p: np.ndarray, anchor_pred: np.ndarray, y: np.ndarray, folds: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    pred = anchor_pred.copy()
    chosen: dict[str, list[int]] = {}
    # A precision-first selective correction: held-out candidates are ranked by
    # a source consensus score; anchor positives remain positive.
    score = source_p.mean(axis=1) + .25 * source_p.std(axis=1)
    for fold in range(5):
        held = np.where(folds == fold)[0]
        pool = held[anchor_pred[held] == 0]
        take = pool[np.argsort(score[pool])[::-1][:k]]
        pred[take] = 1; chosen[str(fold)] = [int(i) for i in take]
    return pred.astype(int), {"candidate": name, "family": "selective_top_k", "k_per_outer_fold": k, "chosen_positions": chosen}


def read_source(path: Path, ids: list[str]) -> tuple[dict, np.ndarray]:
    rows = {str(row["id"]): row for row in read_jsonl(path)}
    if len(rows) != len(ids) or set(rows) != set(ids):
        raise ValueError(f"ID alignment failed: {path}")
    folds = np.asarray([int(rows[rid]["fold"]) for rid in ids])
    if set(folds) != set(range(5)):
        raise ValueError(f"not a five-fold OOF source: {path}")
    return rows, folds


def archive_stale() -> list[str]:
    archive = OUT / "archive" / RUN_ID
    names = ("FAIR_FRAMEWORK_CYCLE_REPORT.md", "LOGIC_REVIEW.md", "development_selection.json", "experiment_registry.csv", "status.json")
    copied = []
    for name in names:
        old = OUT / name
        if old.exists():
            archive.mkdir(parents=True, exist_ok=True); shutil.copy2(old, archive / name); copied.append(name)
    atomic_text(archive / "README.md", "Archived before anchor-arbiter-cycle-v1. These files are nonauthoritative historical artifacts.\n")
    return copied


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    archived = archive_stale()
    dev = list(read_jsonl(ROOT / "data/processed/vipragsent_dev.jsonl")); ids = [str(row["id"]) for row in dev]
    y_by_target = {label: np.asarray([int(row["labels"][label]) for row in dev]) for label in TARGETS}
    source_rows, source_folds, manifests = {}, None, {}
    for name, rel in SOURCES.items():
        path = ROOT / rel
        rows, folds = read_source(path, ids); source_rows[name] = rows
        if source_folds is None: source_folds = folds
        elif not np.array_equal(source_folds, folds): raise ValueError(f"inconsistent OOF folds: {name}")
        manifests[name] = {"path": rel, "sha256": digest(path), "records": len(rows), "five_fold_oof": True}
    assert source_folds is not None
    restored_summary = ROOT / "answer/final_best_tuned_fair_framework_candidates/repeated_oof_summary.json"
    atomic_json(OUT / "source_manifests" / "anchor_arbiter_sources.json", {"run_id": RUN_ID, "sources": manifests, "restored_code_candidate_evidence": {
        "path": str(restored_summary.relative_to(ROOT)), "sha256": digest(restored_summary), "candidate": "phobert_3_reproduced+visobert_2+visobert_3", "weights": [.35, .35, .30], "pooled_oof_f1": 81.6547422491286, "median_repeated_delta": 1.7496468794744686,
        "note": "preserved from its completed repeated-OOF evaluation; it is not silently replaced by the lower-scoring historical screen anchor"}, "unavailable_required_anchors": {
        "sailor_irony_qlora": "adapter is present, but no canonical-train OOF or development prediction artifact exists; test predictions were not read",
        "vistral_code_qlora": "adapter is present, but no canonical-train OOF or development prediction artifact exists; test predictions were not read",
        "xlmr_idiom_anchor": "checkpoint is present, but no canonical-train OOF artifact exists",
    }})

    records, registry = [], []
    anchor_summary = {"run_id": RUN_ID, "split": "canonical development 5-fold OOF source bank", "canonical_train_oof_complete": False, "targets": {}}
    for target in TARGETS:
        anchor = ANCHORS[target]; alt_names = ALTERNATES[target]; y = y_by_target[target]
        anchor_rows = source_rows[anchor]
        anchor_p = np.asarray([float(anchor_rows[rid]["probabilities"][target]) for rid in ids])
        anchor_pred = np.asarray([int(anchor_rows[rid]["predictions"][target]) for rid in ids])
        alt_p = np.asarray([[float(source_rows[name][rid]["probabilities"][target]) for name in alt_names] for rid in ids])
        x = features(np.column_stack((anchor_p, alt_p)))
        disagreement = {name: int((anchor_pred != np.asarray([int(source_rows[name][rid]["predictions"][target]) for rid in ids])).sum()) for name in alt_names}
        anchor_summary["targets"][target] = {"screen_anchor": anchor, "metrics": metric(y, anchor_pred), "alternates": list(alt_names), "binary_prediction_disagreement": disagreement,
            "probability_mean_absolute_difference": {name: float(np.abs(anchor_p - alt_p[:, j]).mean()) for j, name in enumerate(alt_names)},
            "probability_max_absolute_difference": {name: float(np.abs(anchor_p - alt_p[:, j]).max()) for j, name in enumerate(alt_names)}}
        for family in ("conservative_logistic", "conservative_gradient_boosted", "nonnegative_probability_stacker"):
            runs = []
            for seed in SEEDS:
                pred, detail = nested_eval(f"{target}:{family}", family, x, np.column_stack((anchor_p, alt_p)), anchor_p, anchor_pred, y, source_folds, seed)
                m, c = metric(y, pred), corrections(y, anchor_pred, pred)
                runs.append({"seed": seed, "metrics": m, "delta": m["binary_macro_f1"] - metric(y, anchor_pred)["binary_macro_f1"], "corrections": c, "detail": detail})
            records.append({"target": target, "anchor": anchor, "candidate": family, "kind": "nested_meta_oof", "anchor_metrics": metric(y, anchor_pred), "runs": runs, "median_delta": float(np.median([r["delta"] for r in runs])), "source_prediction_disagreement": disagreement})
        for k in (1, 2, 3):
            pred, detail = topk_eval(f"{target}:topk{k}", np.column_stack((anchor_p, alt_p)), anchor_pred, y, source_folds, k)
            m, c = metric(y, pred), corrections(y, anchor_pred, pred)
            records.append({"target": target, "anchor": anchor, "candidate": f"selective_topk_{k}", "kind": "nested_outer_fold_correction", "anchor_metrics": metric(y, anchor_pred), "runs": [{"seed": None, "metrics": m, "delta": m["binary_macro_f1"] - metric(y, anchor_pred)["binary_macro_f1"], "corrections": c, "detail": detail}], "median_delta": m["binary_macro_f1"] - metric(y, anchor_pred)["binary_macro_f1"], "source_prediction_disagreement": disagreement})
    for result in records:
        run = result["runs"][0]; c = run["corrections"]
        eligible = result["median_delta"] > 0 and c["prediction_disagreement_count"] > 0 and c["introduced_FP"] == 0 and c["introduced_FN"] == 0
        registry.append({"run_id": RUN_ID, "target": result["target"], "candidate": result["candidate"], "anchor": result["anchor"], "median_delta": result["median_delta"], "rescued_FN": c["rescued_FN"], "introduced_FP": c["introduced_FP"], "eligible_in_screen": eligible, "rejection_reason": "" if eligible else "did not meet positive-delta, genuine-disagreement, zero-new-error screen"})
    atomic_json(OUT / "disagreement_arbiter_results.json", {"run_id": RUN_ID, "scope": "development nested OOF only; canonical test not accessed", "repeated_seeds": list(SEEDS), "records": records})
    atomic_json(OUT / "anchor_oof_summary.json", anchor_summary)
    with (OUT / "experiment_registry.csv.tmp").open("w", newline="", encoding="utf-8") as h:
        fields = list(registry[0]); writer = csv.DictWriter(h, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(registry)
    (OUT / "experiment_registry.csv.tmp").replace(OUT / "experiment_registry.csv")
    preserved = {"candidate": "phobert_3_reproduced+visobert_2+visobert_3", "weights": [0.35, .35, .30], "status": "preserved", "repeated_oof_f1": 81.6547422491286, "median_repeated_delta": 1.7496468794744686, "reason": "restored strong code candidate; the new historical-anchor screen was not a stronger, provenance-complete alternative"}
    selection = {"run_id": RUN_ID, "protected_labels": {"implicit_sentiment": "copied exactly from incumbent", "sarcasm": "copied exactly from incumbent", "mocking": "copied exactly from incumbent"}, "code_switching": preserved, "irony": {"status": "not_selected", "reason": "Sailor anchor lacks canonical-train OOF and label-free development inference provenance"}, "idiom_figurative": {"status": "not_selected", "reason": "PhoBERT/XLM-R anchor pair lacks canonical-train OOF and label-free development inference provenance"}, "canonical_test_access": False}
    atomic_json(OUT / "development_selection.json", selection)
    state = {"run_id": RUN_ID, "updated_at": datetime.now(timezone.utc).isoformat(), "authoritative_runner": "scripts/run_anchor_arbiter_cycle.py", "stage": "development_nested_arbiter_screen_complete", "archived_non_authoritative": archived, "canonical_test_access": False, "code_switching": preserved, "blockers": ["Sailor and Vistral adapters have no paired canonical-train OOF or development probabilities", "XLM-R idiom checkpoint has no canonical-train OOF", "therefore no target can be frozen or selected for canonical-test inference"]}
    atomic_json(OUT / "cycle_state.json", state)
    review = "# Anchor-arbiter logic review\n\n- Authoritative run: `anchor-arbiter-cycle-v1`; only `scripts/run_anchor_arbiter_cycle.py` writes its state artifacts.\n- All evaluated records were canonical development OOF sources, then evaluated in an additional nested five-fold meta split. No canonical-test path is referenced or opened.\n- Protected labels are copied unchanged and are excluded from every target decision.\n- The restored code candidate remains preserved; it is not replaced by an inferior screen result.\n- Required Sailor/Vistral/XLM-R canonical-train OOF plus label-free inference provenance is missing. The test-only adapter manifests were not used as a substitute.\n- Result: no target is development-selected by this cycle, no candidate is frozen, and canonical test remains untouched.\n\nNOT_PROMOTED\n"
    atomic_text(OUT / "LOGIC_REVIEW.md", review)
    lines = ["# ViPragSent anchor-arbiter cycle", "", "**NOT_PROMOTED** — the canonical test was not accessed.", "", "## Actual nested development-OOF screens", "", "| Target | Candidate | Median Δ F1 | Rescued FN | Introduced FP | Status |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for item in registry:
        lines.append(f"| {item['target']} | {item['candidate']} | {item['median_delta']:+.6f} | {item['rescued_FN']} | {item['introduced_FP']} | {'screen-pass' if item['eligible_in_screen'] else 'rejected'} |")
    lines += ["", "The requested Sailor irony anchor, Vistral code anchor, and XLM-R idiom anchor cannot be completed reproducibly from the present checkout: their adapter/checkpoint manifests lack paired canonical-train OOF and label-free development probability artifacts. Existing canonical-test predictions were deliberately not read. The restored `phobert_3_reproduced+visobert_2+visobert_3` code candidate is preserved.", "", "NOT_PROMOTED"]
    atomic_text(OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md", "\n".join(lines) + "\n")
    atomic_json(OUT / "status.json", {"run_id": RUN_ID, "status": "NOT_PROMOTED", "phase": "development_nested_arbiter_screen_complete", "canonical_test_access": False, "reason": "required anchor train-OOF and reproducible label-free development inference are unavailable", "selection": selection})
    print(json.dumps({"run_id": RUN_ID, "records": len(records), "status": "NOT_PROMOTED", "canonical_test_access": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
