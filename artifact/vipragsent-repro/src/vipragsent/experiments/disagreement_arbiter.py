"""Leakage-safe, bidirectional binary disagreement arbitration.

An arbiter is not a target-label classifier.  It is trained exclusively where
an anchor and alternate disagree, with target ``1`` precisely when the
alternate prediction is correct and ``0`` when the anchor is correct.  On
agreement records the common prediction is copied unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


REQUIRED_ANCHORS = {
    "irony": "sailor_7b_sft_qlora",
    "idiom_figurative": "phobert_anchor+xlmr_large_anchor",
    "code_switching": "vistral_7b_sft_qlora",
}


def repeated_folds(y: np.ndarray, seeds: Iterable[int]) -> dict[int, np.ndarray]:
    """Five genuinely shuffled, label-stratified outer-fold assignments."""
    y = np.asarray(y, dtype=int)
    if min(np.bincount(y)) < 5:
        raise ValueError("both classes need at least five records")
    assignments = {}
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
        fold = np.empty(len(y), dtype=int)
        for number, (_, held) in enumerate(splitter.split(np.zeros(len(y)), y)):
            fold[held] = number
        assignments[int(seed)] = fold
    return assignments


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    tp = int(((y == 1) & (prediction == 1)).sum()); tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum()); fn = int(((y == 1) & (prediction == 0)).sum())
    return {"binary_macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0) * 100), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def paired_corrections(y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    return {
        "disagreement_count": int((anchor != candidate).sum()),
        "rescued_FN": int(((y == 1) & (anchor == 0) & (candidate == 1)).sum()),
        "removed_FP": int(((y == 0) & (anchor == 1) & (candidate == 0)).sum()),
        "introduced_FP": int(((y == 0) & (anchor == 0) & (candidate == 1)).sum()),
        "introduced_FN": int(((y == 1) & (anchor == 1) & (candidate == 0)).sum()),
    }


def disagreement_target(y: np.ndarray, anchor: np.ndarray, alternate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return disagreement indexes and the exact *alternate-is-correct* target."""
    y, anchor, alternate = (np.asarray(value, dtype=int) for value in (y, anchor, alternate))
    indexes = np.flatnonzero(anchor != alternate)
    if indexes.size and not np.all(anchor[indexes] != alternate[indexes]):
        raise ValueError("binary disagreement invariant violated")
    # For binary disagreement exactly one prediction equals gold.
    target = (alternate[indexes] == y[indexes]).astype(int)
    if indexes.size and not np.all((anchor[indexes] == y[indexes]) != (alternate[indexes] == y[indexes])):
        raise ValueError("disagreement target is not exclusive")
    return indexes, target


def source_features(anchor_probability: np.ndarray, alternate_probabilities: np.ndarray, extras: np.ndarray | None = None) -> np.ndarray:
    """Features allowed before fitting; all are source/text-derived, never gold."""
    anchor_probability = np.asarray(anchor_probability, dtype=float).reshape(-1, 1)
    alternate_probabilities = np.asarray(alternate_probabilities, dtype=float)
    if alternate_probabilities.ndim == 1:
        alternate_probabilities = alternate_probabilities[:, None]
    source = np.hstack((anchor_probability, alternate_probabilities))
    margin = alternate_probabilities.mean(axis=1, keepdims=True) - anchor_probability
    variance = source.var(axis=1, keepdims=True)
    spread = (source.max(axis=1) - source.min(axis=1))[:, None]
    entropy = -np.clip(source, 1e-6, 1 - 1e-6) * np.log(np.clip(source, 1e-6, 1 - 1e-6)) - (1 - np.clip(source, 1e-6, 1 - 1e-6)) * np.log(np.clip(1 - source, 1e-6, 1 - 1e-6))
    agreement = (source >= .5).mean(axis=1, keepdims=True)
    pieces = [source, margin, variance, spread, entropy.mean(axis=1, keepdims=True), agreement]
    if extras is not None:
        extra = np.asarray(extras, dtype=float)
        if extra.shape[0] != source.shape[0]:
            raise ValueError("extra feature rows do not align")
        pieces.append(extra if extra.ndim == 2 else extra[:, None])
    return np.hstack(pieces)


def model_factory(name: str, seed: int):
    if name == "logistic":
        return LogisticRegression(C=.2, class_weight="balanced", max_iter=4000, random_state=seed)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=2.0, learning_rate=.05, random_state=seed)
    if name == "shallow_mlp":
        return MLPClassifier(hidden_layer_sizes=(12,), alpha=.02, early_stopping=True, max_iter=500, random_state=seed)
    raise ValueError(f"unknown arbiter family: {name}")


@dataclass(frozen=True)
class ArbiterRun:
    seed: int
    family: str
    prediction: np.ndarray
    fold_details: list[dict]
    disagreement_accuracy: float


def crossfit_arbiter(*, features: np.ndarray, gold: np.ndarray, anchor_prediction: np.ndarray, alternate_prediction: np.ndarray, folds: np.ndarray, family: str, seed: int) -> ArbiterRun:
    """Cross-fit an arbiter; fit only outer-training disagreement records."""
    features = np.asarray(features, dtype=float); gold = np.asarray(gold, dtype=int)
    anchor_prediction = np.asarray(anchor_prediction, dtype=int); alternate_prediction = np.asarray(alternate_prediction, dtype=int); folds = np.asarray(folds, dtype=int)
    if len(features) != len(gold) or set(folds) != set(range(5)):
        raise ValueError("expected aligned five-fold records")
    result = np.where(anchor_prediction == alternate_prediction, anchor_prediction, anchor_prediction).astype(int)
    decisions, details = [], []
    for fold in range(5):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        train_disagreement, choice = disagreement_target(gold[train], anchor_prediction[train], alternate_prediction[train])
        train_disagreement = train[train_disagreement]
        held_disagreement, held_choice = disagreement_target(gold[held], anchor_prediction[held], alternate_prediction[held])
        held_disagreement = held[held_disagreement]
        if len(held_disagreement) == 0:
            chosen = np.zeros(0, dtype=int); status = "no_held_disagreements"
        elif len(train_disagreement) < 8 or len(np.unique(choice)) < 2:
            # A one-sided arbiter cannot be trained; preserve the anchor on this fold.
            chosen = np.zeros(len(held_disagreement), dtype=int); status = "insufficient_disagreement_classes"
        else:
            scaler = StandardScaler().fit(features[train_disagreement])
            model = model_factory(family, seed + fold)
            model.fit(scaler.transform(features[train_disagreement]), choice)
            chosen = model.predict(scaler.transform(features[held_disagreement])).astype(int); status = "ok"
        result[held_disagreement] = np.where(chosen == 1, alternate_prediction[held_disagreement], anchor_prediction[held_disagreement])
        decisions.extend((chosen == held_choice).tolist())
        details.append({"fold": fold, "train_disagreement_records": int(len(train_disagreement)), "held_disagreement_records": int(len(held_disagreement)), "arbiter_status": status})
    return ArbiterRun(seed=seed, family=family, prediction=result, fold_details=details, disagreement_accuracy=float(np.mean(decisions)) if decisions else 0.0)


def bootstrap_probability_positive_delta(gold: np.ndarray, anchor: np.ndarray, candidate: np.ndarray, *, seed: int, replicates: int = 2000) -> float:
    rng = np.random.default_rng(seed); gold, anchor, candidate = (np.asarray(item, dtype=int) for item in (gold, anchor, candidate))
    wins = 0
    for _ in range(replicates):
        indexes = rng.integers(0, len(gold), len(gold))
        wins += metric(gold[indexes], candidate[indexes])["binary_macro_f1"] > metric(gold[indexes], anchor[indexes])["binary_macro_f1"]
    return wins / replicates


def eligibility(runs: Iterable[dict], *, minimum_positive_fraction: float = .60, bootstrap_probability: float | None = None) -> tuple[bool, str]:
    """Aggregate all repeated runs; never inspect merely the first result."""
    items = list(runs)
    if not items:
        return False, "no repeated runs"
    deltas = np.asarray([float(item["delta"]) for item in items])
    corrections = [item["corrections"] for item in items]
    net = np.asarray([c["rescued_FN"] + c["removed_FP"] - c["introduced_FP"] - c["introduced_FN"] for c in corrections])
    if float(np.median(deltas)) <= 0:
        return False, "non-positive median same-split delta"
    if float((deltas > 0).mean()) < minimum_positive_fraction:
        return False, "positive delta is not stable across split seeds"
    if bootstrap_probability is not None and bootstrap_probability < .80:
        return False, "paired bootstrap probability below 0.80"
    if float(np.median(net)) <= 0:
        return False, "non-positive paired net correction"
    return True, "eligible"


def preserve_strong_code(existing: dict, challenger: dict) -> dict:
    """A later screen cannot overwrite the known strong code candidate unless stronger."""
    if float(challenger.get("median_delta", float("-inf"))) > float(existing.get("median_delta", float("inf"))) and challenger.get("eligible"):
        return challenger
    return existing
