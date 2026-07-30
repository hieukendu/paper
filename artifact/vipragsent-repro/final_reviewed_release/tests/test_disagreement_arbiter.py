import numpy as np

from vipragsent.experiments.disagreement_arbiter import (
    REQUIRED_ANCHORS, crossfit_arbiter, disagreement_target, eligibility,
    preserve_strong_code, repeated_folds, source_features,
)


def test_anchor_contract_is_the_real_anchor_not_proxy():
    assert REQUIRED_ANCHORS == {"irony": "sailor_7b_sft_qlora", "idiom_figurative": "phobert_anchor+xlmr_large_anchor", "code_switching": "vistral_7b_sft_qlora"}


def test_disagreement_target_contains_only_disagreements_and_is_exclusive():
    gold = np.array([0, 1, 0, 1]); anchor = np.array([0, 0, 1, 1]); alternate = np.array([0, 1, 0, 0])
    indexes, target = disagreement_target(gold, anchor, alternate)
    assert indexes.tolist() == [1, 2, 3]
    assert target.tolist() == [1, 1, 0]


def test_arbiter_changes_in_both_directions_and_keeps_agreement():
    # Probability features identify whether alternate is correct; both 0->1 and 1->0 occur.
    n = 50; gold = np.tile([1, 0, 1, 0, 1], 10)
    # Each block has alternate-correct rescue/veto, anchor-correct rescue/veto,
    # and one agreement record.
    anchor = np.tile([0, 1, 1, 0, 1], 10); alternate = np.tile([1, 0, 0, 1, 1], 10)
    alternate_correct = alternate == gold
    anchor_p = np.where(alternate_correct, .2, .9); alternate_p = np.where(alternate_correct, .9, .2)
    features = source_features(anchor_p, alternate_p)
    run = crossfit_arbiter(features=features, gold=gold, anchor_prediction=anchor, alternate_prediction=alternate, folds=np.arange(n) % 5, family="logistic", seed=17)
    changed = np.flatnonzero(run.prediction != anchor)
    assert changed.size > 0
    assert set(run.prediction[changed]) == {0, 1}
    agreement = anchor == alternate
    assert np.array_equal(run.prediction[agreement], anchor[agreement])


def test_eligibility_uses_all_runs_and_allows_bounded_new_errors():
    runs = [
        {"delta": .3, "corrections": {"rescued_FN": 3, "removed_FP": 1, "introduced_FP": 1, "introduced_FN": 0}},
        {"delta": .2, "corrections": {"rescued_FN": 2, "removed_FP": 0, "introduced_FP": 1, "introduced_FN": 0}},
        {"delta": -.1, "corrections": {"rescued_FN": 2, "removed_FP": 0, "introduced_FP": 1, "introduced_FN": 0}},
    ]
    assert eligibility(runs, bootstrap_probability=.9) == (True, "eligible")
    # The first run alone is positive, but the aggregate is not robust.
    weak = [runs[0], {**runs[2]}, {**runs[2]}]
    assert eligibility(weak, bootstrap_probability=.9)[0] is False


def test_repeated_split_seed_assignments_are_genuinely_different():
    folds = repeated_folds(np.tile([0, 1], 50), (11, 12))
    assert set(folds[11]) == set(range(5))
    assert not np.array_equal(folds[11], folds[12])


def test_later_weaker_screen_cannot_overwrite_restored_code():
    existing = {"name": "restored", "median_delta": 1.7496, "eligible": True}
    challenger = {"name": "later", "median_delta": .42, "eligible": True}
    assert preserve_strong_code(existing, challenger) == existing
