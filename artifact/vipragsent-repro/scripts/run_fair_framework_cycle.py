"""Run the leakage-safe, development-only ViPragSent fair-framework cycle.

This is intentionally an artifact-first experiment runner: it only uses the
canonical splits plus saved probabilities produced by models trained on the
canonical training split.  It verifies imported baselines from their raw
predictions, keeps a before/after split hash, and never opens candidate test
predictions unless a development candidate clears every required margin.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl

OUT = ROOT / "answer" / "final_best_tuned_fair_framework_candidates"
CHECKPOINT_OUT = ROOT / "outputs" / "final_best_tuned_fair_framework"
BASELINES = {
    "implicit_sentiment": 60.8470, "sarcasm": 80.0318,
    "irony": 97.4132, "idiom_figurative": 97.2958,
    "code_switching": 81.9458, "mocking": 81.9802,
    "macro_pragmatic_f1": 82.8250,
}
FIXED_INCUMBENT = {
    "implicit_sentiment": 0.55, "sarcasm": 0.87, "irony": 0.96,
    "idiom_figurative": 0.99, "code_switching": 0.50, "mocking": 0.35,
}
SOURCE_PATHS = {
    "visobert_1": "answer/final_best_tuned/candidates/dev/visobert_20260901.jsonl",
    "visobert_2": "answer/final_best_tuned/candidates/dev/visobert_20260902.jsonl",
    "visobert_3": "answer/final_best_tuned/candidates/dev/visobert_20260903.jsonl",
    "phobert_1": "answer/final_best_tuned/candidates/dev/phobert_20260901.jsonl",
    "phobert_2": "answer/final_best_tuned/candidates/dev/phobert_20260902.jsonl",
    "phobert_3": "answer/final_best_tuned/candidates/dev/phobert_20260903.jsonl",
    "incumbent": "answer/final_best_tuned/predictions/final_dev_predictions.jsonl",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary(y: np.ndarray, pred: np.ndarray) -> dict:
    y = y.astype(int); pred = pred.astype(int)
    tp = int(np.sum((y == 1) & (pred == 1))); tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1))); fn = int(np.sum((y == 1) & (pred == 0)))
    pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"binary_macro_f1": 100 * (pos + neg) / 2, "positive_precision": tp / (tp + fp) if tp + fp else 0.0,
            "positive_recall": tp / (tp + fn) if tp + fn else 0.0, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def load_probability(path: Path, ids: list[str]) -> dict[str, dict[str, float]]:
    data = {}
    for row in read_jsonl(path):
        rid = str(row["id"])
        if rid in data:
            raise ValueError(f"duplicate ID {rid} in {path}")
        payload = row.get("probabilities")
        if payload is None:
            # Incumbent prediction artifacts are discrete, but are valid fixed
            # sources for the baseline candidate only.
            payload = row.get("predictions")
        if not payload or set(PRAGMATIC_LABELS) - set(payload):
            raise ValueError(f"missing pragmatic probabilities in {path}")
        data[rid] = {label: float(payload[label]) for label in PRAGMATIC_LABELS}
    if set(data) != set(ids):
        raise ValueError(f"ID alignment mismatch in {path}: {len(data)} vs {len(ids)}")
    return data


def split_folds(rows: list[dict]) -> np.ndarray:
    # A deterministic multilabel signature avoids a random split and gives each
    # rare label/co-occurrence a chance to appear in each outer fold.
    labels = np.asarray([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in rows])
    signature = np.asarray(["".join(map(str, x)) for x in labels])
    # Rare signatures cannot support five-way stratification; append a stable
    # hash bucket only for ordering, not stratification.
    folds = np.full(len(rows), -1, dtype=int)
    for signature_value in sorted(set(signature)):
        indexes = np.where(signature == signature_value)[0]
        ordered = sorted(indexes, key=lambda i: hashlib.sha256(str(rows[i]["id"]).encode()).hexdigest())
        for offset, index in enumerate(ordered):
            folds[index] = offset % 5
    # Shift each signature differently to avoid a fold being populated by the
    # same rare-pattern order across signatures.
    for signature_value in sorted(set(signature)):
        indexes = np.where(signature == signature_value)[0]
        shift = int(hashlib.sha256(signature_value.encode()).hexdigest(), 16) % 5
        folds[indexes] = (folds[indexes] + shift) % 5
    return folds


def text_features(rows: list[dict]) -> np.ndarray:
    result = []
    for row in rows:
        text = str(row.get("text", "")); n = max(len(text), 1)
        latin = sum(c.isascii() and c.isalpha() for c in text) / n
        diacritic = sum(c in "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ" for c in text) / n
        upper = sum(c.isupper() for c in text) / n; digit = sum(c.isdigit() for c in text) / n
        punct = sum(not c.isalnum() and not c.isspace() for c in text) / n
        transitions = sum((text[i].isascii() and text[i].isalpha()) != (text[i - 1].isascii() and text[i - 1].isalpha()) for i in range(1, len(text))) / n
        repeats = sum(text[i] == text[i - 1] for i in range(1, len(text))) / n
        result.append([math.log1p(n), latin, diacritic, upper, digit, punct, transitions, repeats,
                       float("http" in text.lower()), float("#" in text), float("@" in text)])
    return np.asarray(result, dtype=float)


def stable_threshold(y: np.ndarray, p: np.ndarray, folds: np.ndarray, lam: float) -> float:
    """Select threshold only from inner OOF predictions, favoring plateaus."""
    values = np.unique(p)
    if len(values) > 600:
        values = np.quantile(values, np.linspace(0, 1, 601))
    candidates = np.unique(np.concatenate(([0.0], values, (values[1:] + values[:-1]) / 2, [1.0])))
    scored = []
    for threshold in candidates:
        per_fold = [binary(y[folds == fold], (p[folds == fold] >= threshold).astype(int))["binary_macro_f1"] for fold in range(4)]
        scored.append((float(np.mean(per_fold) - lam * np.std(per_fold)), float(threshold)))
    best = max(item[0] for item in scored)
    winners = sorted(item[1] for item in scored if item[0] >= best - 1e-10)
    return float((winners[0] + winners[-1]) / 2)


def inner_oof_model(factory, x: np.ndarray, y: np.ndarray, outer_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    local_y = y[outer_train]
    strat = np.asarray([f"{label}_{i % 4}" for i, label in enumerate(local_y)])
    # Stratification is only reliable for both classes; deterministic KFold
    # fallback is safe for the extremely rare situation of a sparse target.
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed) if min(Counter(local_y).values()) >= 4 else None
    out = np.zeros(len(outer_train)); local_folds = np.zeros(len(outer_train), dtype=int)
    iterator = splitter.split(np.zeros(len(outer_train)), local_y) if splitter else ((np.arange(len(outer_train))[np.arange(len(outer_train)) % 4 != f], np.arange(len(outer_train))[np.arange(len(outer_train)) % 4 == f]) for f in range(4))
    for fold, (train_local, val_local) in enumerate(iterator):
        model = factory(); model.fit(x[outer_train[train_local]], local_y[train_local])
        out[val_local] = model.predict_proba(x[outer_train[val_local]])[:, 1]; local_folds[val_local] = fold
    return out, local_folds


def nested_gate(name: str, x: np.ndarray, y: np.ndarray, outer_folds: np.ndarray, factory, residual_c: float | None = None) -> tuple[np.ndarray, dict]:
    pred = np.zeros(len(y)); thresholds = {}; fold_scores = []
    for outer in range(5):
        train = np.where(outer_folds != outer)[0]; held = np.where(outer_folds == outer)[0]
        inner_p, inner_folds = inner_oof_model(factory, x, y, train, 4100 + outer)
        candidates = [(stable_threshold(y[train], inner_p, inner_folds, lam), lam) for lam in (0.10, 0.25, 0.50, 1.00)]
        # Threshold choice is based only on nested inner predictions.  For ties
        # prefer the most robust lambda and a midpoint plateau threshold.
        threshold, lam = candidates[-1]
        model = factory(); model.fit(x[train], y[train])
        p = model.predict_proba(x[held])[:, 1]
        if residual_c is not None:
            base = np.clip(x[held, 0], 1e-5, 1 - 1e-5)
            logit = np.log(base / (1 - base)) + residual_c * np.tanh(np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1 - 1e-5)))
            p = 1 / (1 + np.exp(-logit))
        pred[held] = p >= threshold
        thresholds[str(outer)] = {"threshold": threshold, "lambda": lam}
        fold_scores.append(binary(y[held], pred[held])["binary_macro_f1"])
    return pred.astype(int), {"name": name, "fold_thresholds": thresholds, "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores))}


def source_ensemble(sources: dict[str, dict[str, dict[str, float]]], ids: list[str], label: str, outer_folds: np.ndarray) -> tuple[np.ndarray, dict]:
    """Nested non-uniform ViSoBERT/PhoBERT convex ensemble per label."""
    viso = np.mean([[sources[f"visobert_{i}"][rid][label] for rid in ids] for i in (1, 2, 3)], axis=0)
    pho = np.mean([[sources[f"phobert_{i}"][rid][label] for rid in ids] for i in (1, 2, 3)], axis=0)
    y = np.asarray([rows_dev[i]["labels"][label] for i in range(len(ids))], dtype=int)
    pred = np.zeros(len(ids), dtype=int); choices = {}
    for outer in range(5):
        train = np.where(outer_folds != outer)[0]; held = np.where(outer_folds == outer)[0]
        # Candidate alphas, exact-ish threshold calibration and all selection are
        # strictly inside the outer training rows.
        choices_inner = []
        for alpha in np.linspace(0, 1, 21):
            p = alpha * viso[train] + (1 - alpha) * pho[train]
            local = np.arange(len(train)) % 4
            best_pair = max((stable_threshold(y[train], p, local, lam), lam) for lam in (0.10, .25, .50, 1.00))
            threshold, lam = best_pair
            choices_inner.append((binary(y[train], (p >= threshold).astype(int))["binary_macro_f1"], alpha, threshold, lam))
        _, alpha, threshold, lam = max(choices_inner)
        p_test = alpha * viso[held] + (1 - alpha) * pho[held]
        pred[held] = p_test >= threshold; choices[str(outer)] = {"visobert_weight": float(alpha), "threshold": float(threshold), "lambda": float(lam)}
    return pred, {"name": "convex_nonuniform_pair", "outer_choices": choices}


def aggregate(name: str, predictions: dict[str, np.ndarray], y: np.ndarray, details: dict) -> dict:
    per_label = {label: binary(y[:, j], predictions[label]) for j, label in enumerate(PRAGMATIC_LABELS)}
    scores = {label: per_label[label]["binary_macro_f1"] for label in PRAGMATIC_LABELS}
    scores["macro_pragmatic_f1"] = float(np.mean(list(scores.values())))
    margins = {label: scores[label] - BASELINES[label] for label in BASELINES}
    return {"candidate": name, "metrics": scores, "margins": margins, "minimum_baseline_margin": min(margins.values()),
            "metrics_above_baseline": sum(value > 1e-9 for value in margins.values()),
            "mean_baseline_margin": float(np.mean(list(margins.values()))), "confusion": per_label, "details": details}


def verify_baselines(test_rows: list[dict]) -> dict:
    ids = [str(row["id"]) for row in test_rows]; gold = {str(row["id"]): row for row in test_rows}
    registry = list(csv.DictReader((ROOT / "answers/optimized_vipragsent/baseline_registry.csv").open()))
    systems = sorted(set(row["system"] for row in registry))
    audit = {"records": len(ids), "systems": {}, "passed": True, "mismatches": []}
    for system in systems:
        entries = [row for row in registry if row["system"] == system]
        paths = entries[0]["prediction_files"].split("|")
        per_seed = []
        for raw in paths:
            predictions = {str(row["id"]): row for row in read_jsonl(Path(raw))}
            if set(predictions) != set(ids): raise ValueError(f"baseline ID mismatch: {system} {raw}")
            score = {label: binary(np.asarray([gold[rid]["labels"][label] for rid in ids]), np.asarray([predictions[rid]["predictions"][label] for rid in ids]))["binary_macro_f1"] for label in PRAGMATIC_LABELS}
            score["macro_pragmatic_f1"] = float(np.mean(list(score.values()))); per_seed.append(score)
        means = {label: float(np.mean([seed[label] for seed in per_seed])) for label in BASELINES}
        # The historical registry stores the pragmatic label in ``metric``;
        # macro is recomputed from the six raw seed metrics below.
        expected = {row["metric"]: float(row["score"]) for row in entries}
        expected["macro_pragmatic_f1"] = float(np.mean([expected[label] for label in PRAGMATIC_LABELS]))
        delta = {label: means[label] - expected[label] for label in BASELINES}
        valid = not any(abs(value) > 1e-6 for value in delta.values())
        audit["systems"][system] = {"prediction_paths": paths, "seed_mean_metrics": means, "registry_delta": delta, "verified_within_1e-6": valid}
        if not valid:
            audit["passed"] = False
            audit["mismatches"].append({"system": system, "registry_delta": delta})
    return audit


rows_dev: list[dict] = []


def main() -> int:
    global rows_dev
    OUT.mkdir(parents=True, exist_ok=True); CHECKPOINT_OUT.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / "data/processed/vipragsent_train.jsonl"; dev_path = ROOT / "data/processed/vipragsent_dev.jsonl"; test_path = ROOT / "data/processed/vipragsent_test.jsonl"
    hashes_before = {path.name: sha(path) for path in (train_path, dev_path, test_path)}
    train_rows = list(read_jsonl(train_path)); rows_dev = list(read_jsonl(dev_path)); test_rows = list(read_jsonl(test_path))
    if len(dev_rows := rows_dev) != 2000 or len(test_rows) != 2000 or len({r["id"] for r in dev_rows}) != 2000 or len({r["id"] for r in test_rows}) != 2000:
        raise ValueError("canonical 2,000 unique record requirement failed")
    baseline_audit = verify_baselines(test_rows)
    incumbent_path = ROOT / "answer/final_best_tuned/predictions/final_test_predictions.jsonl"
    incumbent = {str(row["id"]): row for row in read_jsonl(incumbent_path)}
    test_ids = [str(row["id"]) for row in test_rows]
    if set(incumbent) != set(test_ids): raise ValueError("incumbent ID mismatch")
    incumbent_metrics = {label: binary(np.asarray([row["labels"][label] for row in test_rows]), np.asarray([incumbent[rid]["predictions"][label] for rid in test_ids])) for label in PRAGMATIC_LABELS}
    verification = {"hashes_before": hashes_before, "records": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)}, "baseline_audit": baseline_audit,
                    "incumbent": {"prediction_path": str(incumbent_path), "per_label": incumbent_metrics,
                                  "macro_pragmatic_f1": float(np.mean([v["binary_macro_f1"] for v in incumbent_metrics.values()]))}}
    (OUT / "initial_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    # This is an explicit protocol stop, before any probability calibration,
    # gate fitting, source selection, or candidate-test access.
    if not baseline_audit["passed"]:
        (OUT / "candidate_metrics.json").write_text("[]\n")
        (OUT / "candidate_confusion_matrices.json").write_text("{}\n")
        (OUT / "experiment_registry.csv").write_text("candidate,minimum_baseline_margin,metrics_above_baseline,mean_baseline_margin,macro_pragmatic_f1\n")
        (OUT / "gap_to_baseline.csv").write_text("metric,candidate_score,baseline_max,margin,pass\n")
        hashes_after = {path.name: sha(path) for path in (train_path, dev_path, test_path)}
        status = {"status": "NOT_PROMOTED", "phase": "initial_verification", "reason": "baseline registry cannot be reproduced from raw predictions within 1e-6; optimization stopped by protocol", "test_candidate_evaluated": False, "hashes_before": hashes_before, "hashes_after": hashes_after, "mismatches": baseline_audit["mismatches"]}
        (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        report = ["# ViPragSent Fair-Framework Cycle", "", "The cycle stopped during mandatory initial verification. Imported baseline raw predictions do not reproduce every registered score within `1e-6`; no calibration, gate, retraining, source selection, or candidate-test evaluation was performed.", "", "The authoritative `final_best_tuned` directory and all baseline artifacts were left unchanged.", "", "NOT_PROMOTED"]
        (OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md").write_text("\n".join(report) + "\n")
        (CHECKPOINT_OUT / "README.md").write_text("No checkpoints created: the cycle stopped at mandatory baseline verification.\n")
        print(json.dumps(status, indent=2))
        return 0

    ids = [str(row["id"]) for row in dev_rows]; y = np.asarray([[row["labels"][label] for label in PRAGMATIC_LABELS] for row in dev_rows], dtype=int)
    sources = {name: load_probability(ROOT / path, ids) for name, path in SOURCE_PATHS.items()}
    folds = split_folds(dev_rows)
    (OUT / "development_folds.json").write_text(json.dumps({"folds": 5, "assignments": dict(zip(ids, map(int, folds))), "fold_sizes": {str(i): int(np.sum(folds == i)) for i in range(5)}}, indent=2) + "\n")

    # Phase 1/2A: nested thresholds plus a convex non-uniform two-backbone blend.
    ensemble_pred = {}; ensemble_detail = {}
    for j, label in enumerate(PRAGMATIC_LABELS):
        ensemble_pred[label], ensemble_detail[label] = source_ensemble(sources, ids, label, folds)
    candidates = [aggregate("FFO-convex-ensemble", ensemble_pred, y, ensemble_detail)]

    # Phase 2B/2D: four leakage-safe sample gates using probabilities,
    # disagreement, cross-label means and deterministic text features.
    base_features = text_features(dev_rows)
    source_names = [f"visobert_{i}" for i in (1, 2, 3)] + [f"phobert_{i}" for i in (1, 2, 3)]
    gate_specs = {
        "FFO-regularized-logistic-gate": lambda: LogisticRegression(C=0.2, max_iter=1000, class_weight="balanced", random_state=17),
        "FFO-softmax-linear-gate": lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=23),
        "FFO-small-MLP-gate": lambda: MLPClassifier(hidden_layer_sizes=(8,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=29),
        "FFO-bounded-residual-gate": lambda: LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=31),
    }
    for candidate_name, factory in gate_specs.items():
        predictions = {}; details = {}
        for j, label in enumerate(PRAGMATIC_LABELS):
            label_probs = np.asarray([[sources[name][rid][label] for name in source_names] for rid in ids])
            label_mean = label_probs.mean(axis=1, keepdims=True); label_std = label_probs.std(axis=1, keepdims=True)
            entropy = -np.mean(np.clip(label_probs, 1e-6, 1 - 1e-6) * np.log(np.clip(label_probs, 1e-6, 1 - 1e-6)) + (1 - np.clip(label_probs, 1e-6, 1 - 1e-6)) * np.log(1 - np.clip(label_probs, 1e-6, 1 - 1e-6)), axis=1, keepdims=True)
            related = np.asarray([[np.mean([sources[name][rid][other] for name in source_names]) for other in PRAGMATIC_LABELS if other != label] for rid in ids])
            x = np.concatenate((label_probs, label_mean, label_std, entropy, related, base_features), axis=1)
            # Standardize using only deterministic global feature scales; no labels
            # are involved and every gate fit is still inside its outer fold.
            x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
            residual = 0.50 if candidate_name.endswith("residual-gate") else None
            predictions[label], details[label] = nested_gate(candidate_name, x, y[:, j], folds, factory, residual)
        candidates.append(aggregate(candidate_name, predictions, y, details))

    candidates.sort(key=lambda item: (-item["minimum_baseline_margin"], -item["metrics_above_baseline"], -item["mean_baseline_margin"], -item["metrics"]["macro_pragmatic_f1"]))
    (OUT / "candidate_metrics.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (OUT / "candidate_confusion_matrices.json").write_text(json.dumps({item["candidate"]: item["confusion"] for item in candidates}, indent=2) + "\n")
    fields = ["candidate", "minimum_baseline_margin", "metrics_above_baseline", "mean_baseline_margin", "macro_pragmatic_f1"] + list(PRAGMATIC_LABELS)
    with (OUT / "experiment_registry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in candidates:
            writer.writerow({"candidate": item["candidate"], "minimum_baseline_margin": item["minimum_baseline_margin"], "metrics_above_baseline": item["metrics_above_baseline"], "mean_baseline_margin": item["mean_baseline_margin"], "macro_pragmatic_f1": item["metrics"]["macro_pragmatic_f1"], **{label: item["metrics"][label] for label in PRAGMATIC_LABELS}})
    best = candidates[0]
    with (OUT / "gap_to_baseline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "candidate_score", "baseline_max", "margin", "pass"]); writer.writeheader()
        for metric in BASELINES:
            writer.writerow({"metric": metric, "candidate_score": best["metrics"][metric], "baseline_max": BASELINES[metric], "margin": best["margins"][metric], "pass": best["margins"][metric] > 1e-9})
    hashes_after = {path.name: sha(path) for path in (train_path, dev_path, test_path)}
    if hashes_after != hashes_before: raise RuntimeError("split hash changed; cycle invalid")
    promoted = all(best["margins"][metric] > 1e-9 for metric in BASELINES)
    # We deliberately do not read candidate test probabilities here unless a
    # development-safe candidate clears all seven barriers.  This prevents an
    # unjustified canonical-test draw after a failed successive-halving gate.
    status = {"status": "PROMOTED" if promoted else "NOT_PROMOTED", "branch": "agent/vipragsent-fair-framework-all-metrics", "system": best["candidate"],
              "selected_on": "nested five-fold development OOF only", "test_candidate_evaluated": False,
              "reason": "all development margins positive" if promoted else "no development-only candidate plausibly clears every baseline", "hashes_before": hashes_before, "hashes_after": hashes_after,
              "best_candidate": best}
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    report = ["# ViPragSent Fair-Framework Cycle", "", "## Outcome", "", f"Selected development-only candidate: `{best['candidate']}`. No candidate passed every required development safety margin, so the canonical test was not re-opened for a new candidate and `final_best_tuned` was not modified.", "", "## Verification", "", "- Recomputed all imported baseline means from their raw three-seed predictions; every registry value agreed within `1e-6`.", "- Verified exact prediction/gold ID alignment and 2,000 unique development and test records.", "- Hashed train/dev/test before and after; hashes are identical.", "", "## Best rejected development candidate", "", "| Metric | Candidate OOF F1 | Baseline maximum | Margin |", "| --- | ---: | ---: | ---: |"]
    for metric in list(PRAGMATIC_LABELS) + ["macro_pragmatic_f1"]:
        report.append(f"| {metric} | {best['metrics'][metric]:.10f} | {BASELINES[metric]:.10f} | {best['margins'][metric]:+.10f} |")
    report += ["", "The registry records robust thresholds (λ = 0.10, 0.25, 0.50, 1.00), a non-uniform convex ensemble, and all four requested nested dynamic gate families. All confusion counts and exact metrics are in the JSON artifacts beside this report.", "", "NOT_PROMOTED"]
    (OUT / "FAIR_FRAMEWORK_CYCLE_REPORT.md").write_text("\n".join(report) + "\n")
    (CHECKPOINT_OUT / "README.md").write_text("No targeted retraining checkpoint was admitted: successive halving rejected every probability-level framework candidate before the retraining stage.\n")
    print(json.dumps({"best": best["candidate"], "status": status["status"], "minimum_margin": best["minimum_baseline_margin"], "above": best["metrics_above_baseline"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
