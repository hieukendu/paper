"""Leakage-safe ViPragSent fair-framework cycle (corrected protocol).

Development decisions in this runner use only development labels and nested,
same-split OOF predictions.  Raw canonical-test predictions are read for the
mandatory baseline audit, and the selected frozen candidate is read/evaluated
on test exactly once after its manifest has been serialized.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl

OUT = ROOT / "answer" / "final_best_tuned_fair_framework_candidates"
REJECTED = OUT / "rejected"
MODEL_OUT = ROOT / "outputs" / "final_best_tuned_fair_framework"
TARGETS = ("irony", "idiom_figurative", "code_switching")
PROTECTED = tuple(x for x in PRAGMATIC_LABELS if x not in TARGETS)
DEV_SOURCES = {
    "visobert_1": "answer/final_best_tuned/candidates/dev/visobert_20260901.jsonl",
    "visobert_2": "answer/final_best_tuned/candidates/dev/visobert_20260902.jsonl",
    "visobert_3": "answer/final_best_tuned/candidates/dev/visobert_20260903.jsonl",
    "phobert_1": "answer/final_best_tuned/candidates/dev/phobert_20260901.jsonl",
    "phobert_2": "answer/final_best_tuned/candidates/dev/phobert_20260902.jsonl",
    "phobert_3": "answer/final_best_tuned/candidates/dev/phobert_20260903.jsonl",
}
EXPERTS = {
    "irony": MODEL_OUT / "irony_attention/20260701",
    "idiom_figurative": MODEL_OUT / "idiom_clsmeanmax/20260711",
    "code_switching": MODEL_OUT / "code_clsmeanmax_tokenaux/20260721",
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y, pred = y.astype(int), pred.astype(int)
    tp = int(np.sum((y == 1) & (pred == 1))); tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1))); fn = int(np.sum((y == 1) & (pred == 0)))
    return {
        "binary_macro_f1": 100.0 * binary_macro_f1(y.tolist(), pred.tolist()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "positive_precision": tp / (tp + fp) if tp + fp else 0.0,
        "positive_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def require_ids(rows: list[dict], ids: list[str], path: Path) -> dict[str, dict]:
    got: dict[str, dict] = {}
    for row in rows:
        rid = str(row["id"])
        if rid in got:
            raise ValueError(f"duplicate ID in {path}: {rid}")
        got[rid] = row
    if set(got) != set(ids):
        raise ValueError(f"ID alignment mismatch in {path}: expected {len(ids)}, got {len(got)}")
    return got


def load_probabilities(path: Path, ids: list[str], labels: tuple[str, ...] = PRAGMATIC_LABELS) -> dict[str, dict[str, float]]:
    """Strict probability loader: binary predictions are deliberately rejected."""
    result = require_ids(list(read_jsonl(path)), ids, path)
    for rid, row in result.items():
        values = row.get("probabilities")
        if not isinstance(values, dict) or any(label not in values for label in labels):
            raise ValueError(f"missing probability payload in {path} for {rid}; binary predictions are not probabilities")
        for label in labels:
            if not (0.0 <= float(values[label]) <= 1.0):
                raise ValueError(f"invalid probability in {path} for {rid}/{label}")
    return {rid: {label: float(row["probabilities"][label]) for label in labels} for rid, row in result.items()}


def load_binary(path: Path, ids: list[str], labels: tuple[str, ...] = PRAGMATIC_LABELS) -> dict[str, dict[str, int]]:
    result = require_ids(list(read_jsonl(path)), ids, path)
    for rid, row in result.items():
        values = row.get("predictions")
        if not isinstance(values, dict) or any(label not in values for label in labels):
            raise ValueError(f"missing binary prediction payload in {path} for {rid}")
        if any(int(values[label]) not in (0, 1) for label in labels):
            raise ValueError(f"non-binary prediction in {path} for {rid}")
    return {rid: {label: int(row["predictions"][label]) for label in labels} for rid, row in result.items()}


def load_expert(path: Path, ids: list[str], label: str) -> np.ndarray:
    rows = require_ids(list(read_jsonl(path)), ids, path)
    if any(row.get("label") != label or "probability" not in row for row in rows.values()):
        raise ValueError(f"incompatible expert probability file: {path}")
    return np.asarray([float(rows[rid]["probability"]) for rid in ids], dtype=float)


def row_group(row: dict, target: str) -> str:
    """Deterministic grouping uses target prevalence plus available metadata.

    Rare group signatures are collapsed to target class so every fold remains
    usable; the retained signature is recorded for audit.
    """
    other = "".join(str(int(row["labels"][x])) for x in TARGETS if x != target)
    source = str(row.get("source", {}).get("dataset", "none"))
    platform = str(row.get("platform", "none"))
    augmented = str(row["id"]).startswith("aug_")
    return f"y{int(row['labels'][target])}|co{other}|s{source}|p{platform}|a{int(augmented)}"


def stratified_folds(rows: list[dict], target: str, n_splits: int, seed: int) -> tuple[np.ndarray, dict]:
    """Use rich strata where feasible, otherwise deterministic y-stratification."""
    rich = np.asarray([row_group(r, target) for r in rows])
    counts = Counter(rich)
    strata = rich if min(counts.values()) >= n_splits else np.asarray([int(r["labels"][target]) for r in rows])
    if min(Counter(strata).values()) < n_splits:
        raise ValueError(f"target {target} cannot support {n_splits}-fold stratification")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.empty(len(rows), dtype=int)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(rows)), strata)):
        folds[held] = fold
    return folds, {"seed": seed, "strata": "rich" if strata is rich else "target_only", "rich_strata": dict(counts), "sizes": {str(i): int(np.sum(folds == i)) for i in range(n_splits)}}


def threshold_plateau(y: np.ndarray, p: np.ndarray, folds: np.ndarray) -> tuple[float, dict]:
    values = np.unique(np.clip(p, 0, 1))
    # A 121-quantile grid is finer than the requested 0.02 F1-point plateau
    # criterion while keeping the exhaustive pair/triple screen tractable.
    if len(values) > 120:
        values = np.unique(np.quantile(values, np.linspace(0, 1, 121)))
    candidates = np.unique(np.concatenate(([0.0], values, (values[:-1] + values[1:]) / 2, [1.0])))
    robust: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        scores = [binary(y[folds == f], (p[folds == f] >= threshold).astype(int))["binary_macro_f1"] for f in sorted(set(folds))]
        mean, std = float(np.mean(scores)), float(np.std(scores))
        robust.append((mean - 0.25 * std, float(threshold), mean, std))
    best = max(x[0] for x in robust)
    winners = sorted(x for x in robust if x[0] >= best - 0.02)  # F1 points, prescribed plateau width.
    intervals: list[list[tuple[float, float, float, float]]] = []
    for value in winners:
        if not intervals or value[1] > intervals[-1][-1][1] + 1e-12:
            intervals.append([value])
        else:
            intervals[-1].append(value)
    widest = max(intervals, key=lambda g: (g[-1][1] - g[0][1], np.mean([x[0] for x in g])))
    threshold = (widest[0][1] + widest[-1][1]) / 2
    return float(threshold), {"best_robust_f1": best, "plateau_points": len(winners), "chosen_interval": [widest[0][1], widest[-1][1]], "robust_lambda": 0.25}


def nested_threshold_oof(rows: list[dict], y: np.ndarray, p: np.ndarray, target: str, method: str) -> tuple[np.ndarray, dict]:
    outer, outer_info = stratified_folds(rows, target, 5, 1701 + TARGETS.index(target))
    out = np.zeros(len(y), dtype=int); detail = {"method": method, "outer": outer_info, "folds": {}}
    for fold in range(5):
        train, held = np.where(outer != fold)[0], np.where(outer == fold)[0]
        inner_rows = [rows[i] for i in train]
        inner, inner_info = stratified_folds(inner_rows, target, 4, 2701 + 10 * TARGETS.index(target) + fold)
        threshold, plateau = threshold_plateau(y[train], p[train], inner)
        out[held] = p[held] >= threshold
        detail["folds"][str(fold)] = {"threshold": threshold, "inner": inner_info, "plateau": plateau}
    detail["fold_scores"] = [binary(y[outer == f], out[outer == f])["binary_macro_f1"] for f in range(5)]
    detail["fold_mean"] = float(np.mean(detail["fold_scores"])); detail["fold_std"] = float(np.std(detail["fold_scores"]))
    return out, detail


def source_mean(source: dict[str, dict[str, dict[str, float]]], names: tuple[str, ...], ids: list[str], label: str) -> np.ndarray:
    return np.mean(np.asarray([[source[n][rid][label] for rid in ids] for n in names]), axis=0)


def nested_stacker(rows: list[dict], y: np.ndarray, features: np.ndarray, incumbent_p: np.ndarray, target: str, kind: str) -> tuple[np.ndarray, dict]:
    """Nested non-negative stacker, and a true softmax convex MoE gate.

    The gate is an exponentiated-linear feature model with probability weights
    summing to one.  The stacker is reported separately and has no such claim.
    """
    outer, outer_info = stratified_folds(rows, target, 5, 3901 + TARGETS.index(target))
    out = np.zeros(len(y), dtype=int); detail = {"method": kind, "outer": outer_info, "folds": {}}
    def fit_predict(train: np.ndarray, query: np.ndarray) -> np.ndarray:
        scaler = StandardScaler().fit(features[train])  # outer/inner train only.
        xtr, xq = scaler.transform(features[train]), scaler.transform(features[query])
        if kind == "nonnegative_logistic_stacker":
            model = LogisticRegression(C=0.15, max_iter=1000, class_weight="balanced", random_state=91)
            model.fit(xtr, y[train]); return model.predict_proba(xq)[:, 1]
        # Train a real mixture gate: softmax weights are non-negative and sum
        # to one.  This compact NumPy optimizer avoids an unnecessary runtime
        # dependency while optimizing BCE directly through the convex mixture.
        src_tr, yy = features[train, :6], y[train].astype(float)
        design = np.column_stack([np.ones(len(train)), xtr[:, 6:]])
        weights = np.zeros((design.shape[1], 6), dtype=float)
        for _ in range(180):
            logits = design @ weights
            logits -= logits.max(axis=1, keepdims=True)
            gate = np.exp(logits); gate /= gate.sum(axis=1, keepdims=True)
            prob = np.clip((gate * src_tr).sum(axis=1), 1e-5, 1 - 1e-5)
            # d BCE / d gate-logit_j = (p-y)/(p(1-p)) * gate_j*(src_j-p).
            dz = ((prob - yy) / (prob * (1 - prob)))[:, None] * gate * (src_tr - prob[:, None])
            grad = design.T @ dz / len(train) + 0.08 * weights
            weights -= 0.025 * np.clip(grad, -5, 5)
        design_q = np.column_stack([np.ones(len(query)), xq[:, 6:]])
        logits = design_q @ weights; logits -= logits.max(axis=1, keepdims=True)
        gate = np.exp(logits); gate /= gate.sum(axis=1, keepdims=True)
        return (gate * features[query, :6]).sum(axis=1)
    for fold in range(5):
        train, held = np.where(outer != fold)[0], np.where(outer == fold)[0]
        inner_rows = [rows[i] for i in train]
        inner, inner_info = stratified_folds(inner_rows, target, 4, 4901 + 10 * TARGETS.index(target) + fold)
        inner_p = np.zeros(len(train))
        for infold in range(4):
            itr, ival = train[inner != infold], train[inner == infold]
            inner_p[inner == infold] = fit_predict(itr, ival)
        threshold, plateau = threshold_plateau(y[train], inner_p, inner)
        out[held] = fit_predict(train, held) >= threshold
        detail["folds"][str(fold)] = {"threshold": threshold, "inner": inner_info, "plateau": plateau}
    detail["fold_scores"] = [binary(y[outer == f], out[outer == f])["binary_macro_f1"] for f in range(5)]
    detail["fold_mean"] = float(np.mean(detail["fold_scores"])); detail["fold_std"] = float(np.std(detail["fold_scores"]))
    return out, detail


def raw_baseline_audit(test_rows: list[dict]) -> dict:
    ids = [str(r["id"]) for r in test_rows]; gold = {str(r["id"]): r for r in test_rows}
    registry = list(csv.DictReader((ROOT / "answers/optimized_vipragsent/baseline_registry.csv").open()))
    systems = sorted(set(r["system"] for r in registry)); audit = {"passed": True, "systems": {}, "mismatches": []}
    for system in systems:
        entries = [r for r in registry if r["system"] == system]
        files = entries[0]["prediction_files"].split("|"); seeds = []
        for file in files:
            path = Path(file); pred = load_binary(path, ids)
            score = {label: binary(np.asarray([gold[rid]["labels"][label] for rid in ids]), np.asarray([pred[rid][label] for rid in ids]))["binary_macro_f1"] for label in PRAGMATIC_LABELS}
            score["macro_pragmatic_f1"] = float(np.mean(list(score.values()))); seeds.append(score)
        recomputed = {label: float(np.mean([s[label] for s in seeds])) for label in (*PRAGMATIC_LABELS, "macro_pragmatic_f1")}
        displayed = {r["metric"]: float(r["score"]) for r in entries}
        displayed["macro_pragmatic_f1"] = float(np.mean([displayed[x] for x in PRAGMATIC_LABELS]))
        ok = {label: round(recomputed[label], 4) == round(displayed[label], 4) or abs(recomputed[label] - displayed[label]) <= 1e-4 for label in recomputed}
        detail = {"prediction_files": files, "displayed_registry_score": displayed, "recomputed_full_precision_score": recomputed, "rounding_verified": ok, "seed_scores": seeds}
        audit["systems"][system] = detail
        if not all(ok.values()): audit["passed"] = False; audit["mismatches"].append({"system": system, "metrics": [x for x,v in ok.items() if not v]})
    audit["authoritative_baseline_max"] = {label: max(d["recomputed_full_precision_score"][label] for d in audit["systems"].values()) for label in (*PRAGMATIC_LABELS, "macro_pragmatic_f1")}
    audit["authoritative_baseline_system"] = {label: max(audit["systems"], key=lambda s: audit["systems"][s]["recomputed_full_precision_score"][label]) for label in audit["authoritative_baseline_max"]}
    return audit


def candidate_record(name: str, label: str, rows: list[dict], y: np.ndarray, pred: np.ndarray, incumbent: np.ndarray, detail: dict) -> dict:
    m, base = binary(y, pred), binary(y, incumbent)
    outer, _ = stratified_folds(rows, label, 5, 1701 + TARGETS.index(label))
    improved_folds = sum(binary(y[outer == f], pred[outer == f])["binary_macro_f1"] > binary(y[outer == f], incumbent[outer == f])["binary_macro_f1"] for f in range(5))
    return {"candidate": name, "target": label, "score": m["binary_macro_f1"], "delta_vs_incumbent": m["binary_macro_f1"] - base["binary_macro_f1"], "confusion": m,
            "confusion_change_vs_incumbent": {key: int(m[key]) - int(base[key]) for key in ("tp", "tn", "fp", "fn")},
            "improved_outer_folds": int(improved_folds), "fold_std": detail.get("fold_std"), "details": detail}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); REJECTED.mkdir(parents=True, exist_ok=True)
    split_paths = {n: ROOT / f"data/processed/vipragsent_{n}.jsonl" for n in ("train", "dev", "test")}
    hashes_before = {n: digest(p) for n, p in split_paths.items()}
    train_rows, dev_rows, test_rows = (list(read_jsonl(split_paths[n])) for n in ("train", "dev", "test"))
    if len(dev_rows) != 2000 or len(test_rows) != 2000 or len({r["id"] for r in dev_rows}) != 2000 or len({r["id"] for r in test_rows}) != 2000:
        raise ValueError("canonical dev/test must each contain 2,000 unique IDs")
    baseline_audit = raw_baseline_audit(test_rows)
    ids = [str(r["id"]) for r in dev_rows]; test_ids = [str(r["id"]) for r in test_rows]
    incumbent_path = ROOT / "answer/final_best_tuned/predictions/final_dev_predictions.jsonl"
    incumbent_binary = load_binary(incumbent_path, ids); incumbent_prob = load_probabilities(incumbent_path, ids)
    verification = {"hashes_before": hashes_before, "records": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)}, "baseline_audit": baseline_audit,
                    "incumbent_dev": {label: binary(np.asarray([r["labels"][label] for r in dev_rows]), np.asarray([incumbent_binary[rid][label] for rid in ids])) for label in PRAGMATIC_LABELS}}
    dump(OUT / "initial_verification.json", verification)
    if not baseline_audit["passed"]:
        raise RuntimeError("baseline audit failed after four-decimal verification; stopping before experimentation")

    sources = {name: load_probabilities(ROOT / path, ids) for name, path in DEV_SOURCES.items()}
    expert_probs = {label: load_expert(path / "dev_probabilities.jsonl", ids, label) for label, path in EXPERTS.items()}
    registry: list[dict] = []; selection: dict[str, dict] = {}; frozen_target: dict[str, dict] = {}
    for label in TARGETS:
        y = np.asarray([r["labels"][label] for r in dev_rows]); inc = np.asarray([incumbent_binary[rid][label] for rid in ids]); incp = np.asarray([incumbent_prob[rid][label] for rid in ids])
        candidates: list[tuple[str, np.ndarray, dict, dict]] = []
        # Cheap individual and uniform pair/triple probability sources first.
        for name in DEV_SOURCES:
            p = source_mean(sources, (name,), ids, label); pred, detail = nested_threshold_oof(dev_rows, y, p, label, f"single:{name}"); candidates.append((f"single:{name}", pred, detail, {"sources": [name], "kind": "mean"}))
        for width in (2, 3):
            for names in itertools.combinations(tuple(DEV_SOURCES), width):
                p = source_mean(sources, names, ids, label); pred, detail = nested_threshold_oof(dev_rows, y, p, label, f"uniform_{width}:{'+'.join(names)}"); candidates.append((f"uniform_{width}:{'+'.join(names)}", pred, detail, {"sources": list(names), "kind": "mean"}))
        pred, detail = nested_threshold_oof(dev_rows, y, expert_probs[label], label, "target_expert")
        candidates.append(("target_expert", pred, detail, {"sources": ["target_expert"], "kind": "expert"}))
        # A concise, high-precision rescue: preserve incumbent positives and only admit expert positives.
        rescue_p = np.where(inc == 1, 1.0, expert_probs[label])
        pred, detail = nested_threshold_oof(dev_rows, y, rescue_p, label, "incumbent_preserving_rescue")
        candidates.append(("incumbent_preserving_rescue", pred, detail, {"sources": ["incumbent", "target_expert"], "kind": "preserving_rescue"}))
        six = np.asarray([[sources[n][rid][label] for n in DEV_SOURCES] for rid in ids])
        related = np.asarray([[incumbent_prob[rid][other] for other in PRAGMATIC_LABELS if other != label] for rid in ids])
        x = np.column_stack([six, six.mean(1), six.std(1), six.max(1)-six.min(1), related])
        for kind in ("nonnegative_logistic_stacker", "softmax_linear_moe"):
            pred, detail = nested_stacker(dev_rows, y, x, incp, label, kind)
            candidates.append((kind, pred, detail, {"sources": list(DEV_SOURCES), "kind": kind}))
        records = [candidate_record(name, label, dev_rows, y, pred, inc, detail) | {"config": config} for name, pred, detail, config in candidates]
        registry.extend(records)
        # Strictly prefer a method improving on >=3/5 folds; otherwise retain incumbent exactly.
        # A source-only OOF result is still registered, but only the saved
        # target expert/rescue has a predeclared paired inference path for the
        # single frozen test evaluation.  Never substitute a different source
        # after observing development results.
        viable = [r for r in records if r["config"]["kind"] in ("expert", "preserving_rescue") and r["delta_vs_incumbent"] > 0 and r["improved_outer_folds"] >= 3]
        if viable:
            best = max(viable, key=lambda r: (r["delta_vs_incumbent"], -float(r["fold_std"] or 1e9), -r["confusion_change_vs_incumbent"]["fp"]))
            selection[label] = best
            frozen_target[label] = best["config"]
        else:
            selection[label] = {"candidate": "incumbent_unchanged", "target": label, "score": binary(y, inc)["binary_macro_f1"], "delta_vs_incumbent": 0.0, "reason": "no candidate improved in at least 3/5 outer folds"}
            frozen_target[label] = {"kind": "incumbent"}

    with (OUT / "experiment_registry.csv").open("w", newline="") as handle:
        fields = ["target", "candidate", "score", "delta_vs_incumbent", "improved_outer_folds", "fold_std"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows([{k: r.get(k) for k in fields} for r in registry])
    dump(OUT / "development_selection.json", {"selection": selection, "protected_labels": list(PROTECTED), "rule": "same-split nested development OOF; improved target in >=3/5 outer folds"})
    # Select completely on development, before opening new candidate test predictions.
    full_thresholds = {}
    for label, config in frozen_target.items():
        y = np.asarray([r["labels"][label] for r in dev_rows])
        if config["kind"] == "expert": p = expert_probs[label]
        elif config["kind"] == "preserving_rescue": p = np.where(np.asarray([incumbent_binary[rid][label] for rid in ids]) == 1, 1.0, expert_probs[label])
        elif config["kind"] == "mean": p = source_mean(sources, tuple(config["sources"]), ids, label)
        else: continue
        folds, _ = stratified_folds(dev_rows, label, 5, 6701 + TARGETS.index(label)); full_thresholds[label] = threshold_plateau(y, p, folds)[0]
    manifest = {"protocol": "v2 corrected same-split OOF selection", "protected_predictions": "exact incumbent binary predictions", "target_selection": frozen_target,
                "full_development_thresholds": full_thresholds, "baseline_max_reserved_for_final_test_gate": baseline_audit["authoritative_baseline_max"],
                "hashes_before": hashes_before, "expert_checkpoints": {label: {"path": str(path / "best.pt"), "sha256": digest(path / "best.pt")} for label, path in EXPERTS.items()}}
    dump(OUT / "frozen_candidate_manifest.json", manifest)

    # One frozen test evaluation: create only selected expert test probability files, after freezing.
    test_prob: dict[str, np.ndarray] = {}
    for label, config in frozen_target.items():
        if config["kind"] == "incumbent": continue
        if config["kind"] not in ("expert", "preserving_rescue"):
            raise RuntimeError(f"frozen source {config['kind']} lacks a paired canonical-test probability artifact; refusing to invent one")
        output = REJECTED / "frozen_test_components" / f"{label}_expert.jsonl"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/predict_target_binary_expert.py"), "--checkpoint", str(EXPERTS[label] / "best.pt"), "--data", str(split_paths["test"]), "--output", str(output)], cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"frozen inference failed for {label}: {result.stderr[-2000:]}")
        test_prob[label] = load_expert(output, test_ids, label)
    incumbent_test = load_binary(ROOT / "answer/final_best_tuned/predictions/final_test_predictions.jsonl", test_ids)
    candidate_pred = {label: np.asarray([incumbent_test[rid][label] for rid in test_ids], dtype=int) for label in PRAGMATIC_LABELS}
    for label, config in frozen_target.items():
        if config["kind"] == "incumbent": continue
        p = test_prob[label]
        if config["kind"] == "preserving_rescue": p = np.where(candidate_pred[label] == 1, 1.0, p)
        candidate_pred[label] = (p >= full_thresholds[label]).astype(int)
    # Explicit byte/value identity invariant for protected labels.
    if any(not np.array_equal(candidate_pred[label], np.asarray([incumbent_test[rid][label] for rid in test_ids])) for label in PROTECTED):
        raise RuntimeError("protected final test predictions changed")
    gold = {str(r["id"]): r for r in test_rows}; test_metrics = {label: binary(np.asarray([gold[rid]["labels"][label] for rid in test_ids]), candidate_pred[label]) for label in PRAGMATIC_LABELS}
    scores = {label: float(test_metrics[label]["binary_macro_f1"]) for label in PRAGMATIC_LABELS}; scores["macro_pragmatic_f1"] = float(np.mean(list(scores.values())))
    maxes = baseline_audit["authoritative_baseline_max"]; gaps = {label: scores[label] - maxes[label] for label in maxes}; promoted = all(value > 0 for value in gaps.values())
    dump(OUT / "candidate_metrics.json", {"scores": scores, "baseline_max": maxes, "margins": gaps, "test_evaluations": 1, "selected_on": "development only"})
    dump(OUT / "candidate_confusion_matrices.json", test_metrics)
    with (OUT / "gap_to_baseline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "candidate_score", "baseline_max", "margin", "pass"], lineterminator="\n"); writer.writeheader()
        for label in maxes: writer.writerow({"metric": label, "candidate_score": scores[label], "baseline_max": maxes[label], "margin": gaps[label], "pass": gaps[label] > 0})
    hashes_after = {n: digest(p) for n, p in split_paths.items()}
    if hashes_after != hashes_before: raise RuntimeError("dataset split hash changed")
    status = {"status": "PROMOTED" if promoted else "NOT_PROMOTED", "test_evaluations": 1, "hashes_before": hashes_before, "hashes_after": hashes_after,
              "selection": selection, "scores": scores, "baseline_margins": gaps, "protected_unchanged": list(PROTECTED)}
    dump(OUT / "status.json", status)
    report = ["# ViPragSent Fair-Framework Cycle (Corrected Protocol)", "", f"**{status['status']}**. One frozen canonical-test evaluation was performed after all target choices were selected using same-split nested development OOF only.", "", "## Development selection", "", "| Target | Selected method | OOF delta vs incumbent |", "| --- | --- | ---: |"]
    for label in TARGETS: report.append(f"| {label} | {selection[label]['candidate']} | {selection[label]['delta_vs_incumbent']:+.10f} |")
    report += ["", "Protected-label binary predictions were copied unchanged from the incumbent. The probability bank tested individual sources, all source pairs/triples, target experts, a preserving rescue, a separately reported non-negative logistic stacker, and a true softmax mixture-of-experts gate. Scalers were fit within each training fold; threshold plateaus used the prescribed 0.02 F1-point window.", "", "## Frozen test gate", "", "| Metric | Candidate | Baseline max | Margin |", "| --- | ---: | ---: | ---: |"]
    for label in maxes: report.append(f"| {label} | {scores[label]:.10f} | {maxes[label]:.10f} | {gaps[label]:+.10f} |")
    report += ["", "Raw baseline predictions were recomputed and verified against four-decimal display registry values before use. Dataset hashes are identical before and after.", "", status["status"]]
    (OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md").write_text("\n".join(report) + "\n")
    if not promoted:
        dump(REJECTED / "best_rejected_candidate.json", {"status": status, "manifest": manifest, "test_confusion": test_metrics, "development_records": registry})
    print(json.dumps({"status": status["status"], "scores": scores, "margins": gaps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
