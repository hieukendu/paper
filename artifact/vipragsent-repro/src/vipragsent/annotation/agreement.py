from __future__ import annotations

from vipragsent.annotation.disagreement import LABEL_FIELDS
from vipragsent.data.schema import canonicalize_labels


def percent_agreement(pairs: list[tuple[dict, dict]]) -> dict[str, float]:
    totals = {field: 0 for field in LABEL_FIELDS}
    agrees = {field: 0 for field in LABEL_FIELDS}
    for left, right in pairs:
        left_labels = canonicalize_labels(left.get("labels"))
        right_labels = canonicalize_labels(right.get("labels"))
        for field in LABEL_FIELDS:
            totals[field] += 1
            agrees[field] += int(left_labels.get(field) == right_labels.get(field))
    return {field: (agrees[field] / totals[field] if totals[field] else 0.0) for field in LABEL_FIELDS}


def cohen_kappa_binary(left_values: list[int], right_values: list[int]) -> float:
    if len(left_values) != len(right_values):
        raise ValueError("left and right must have equal lengths")
    if not left_values:
        return 0.0
    observed = sum(int(a == b) for a, b in zip(left_values, right_values)) / len(left_values)
    p_left = sum(left_values) / len(left_values)
    p_right = sum(right_values) / len(right_values)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)
