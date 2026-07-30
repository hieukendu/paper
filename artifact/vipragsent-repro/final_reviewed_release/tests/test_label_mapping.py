from vipragsent.data.schema import canonical_label_name, canonicalize_labels


def test_label_aliases_map_to_canonical_names():
    assert canonical_label_name("implicit") == "implicit_sentiment"
    assert canonical_label_name("code_switch") == "code_switching"
    assert canonical_label_name("idiom") == "idiom_figurative"


def test_canonicalize_labels_keeps_unset_fields_null():
    labels = canonicalize_labels({"implicit": 1, "code_switch": 1, "polarity_signed_3way": "positive"})
    assert labels["implicit_sentiment"] == 1
    assert labels["code_switching"] == 1
    assert labels["polarity"] == "positive"
    assert labels["sarcasm"] is None
