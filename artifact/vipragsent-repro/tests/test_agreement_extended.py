from vipragsent.annotation.agreement import cohen_kappa_nominal, krippendorff_alpha_nominal


def test_nominal_agreement_is_one_for_identical_ratings():
    values = ["a", "b", "c", "a"]
    assert cohen_kappa_nominal(values, values) == 1.0
    assert krippendorff_alpha_nominal([[value, value] for value in values]) == 1.0


def test_nominal_agreement_supports_missing_ratings():
    value = krippendorff_alpha_nominal([["a", "a"], ["b", None], ["b", "a"]])
    assert -1.0 <= value <= 1.0
