from __future__ import annotations

from collections import Counter
from typing import Hashable

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
    return cohen_kappa_nominal(left_values, right_values)


def cohen_kappa_nominal(left_values: list[Hashable], right_values: list[Hashable]) -> float:
    if len(left_values) != len(right_values):
        raise ValueError("left and right must have equal lengths")
    if not left_values:
        return 0.0
    observed = sum(int(a == b) for a, b in zip(left_values, right_values)) / len(left_values)
    left_counts = Counter(left_values)
    right_counts = Counter(right_values)
    categories = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[category] / len(left_values))
        * (right_counts[category] / len(right_values))
        for category in categories
    )
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def fleiss_kappa_nominal(ratings: list[list[Hashable | None]]) -> float:
    """Fleiss' kappa for nominal ratings with a fixed number of raters."""
    usable = [[value for value in row if value is not None] for row in ratings]
    usable = [row for row in usable if len(row) >= 2]
    if not usable:
        return 0.0
    rater_count = len(usable[0])
    if rater_count < 2 or any(len(row) != rater_count for row in usable):
        raise ValueError("Fleiss' kappa requires the same number of ratings per item")
    total_items = len(usable)
    category_counts = Counter(value for row in usable for value in row)
    observed = sum(
        sum(count * (count - 1) for count in Counter(row).values()) / (rater_count * (rater_count - 1))
        for row in usable
    ) / total_items
    total_ratings = total_items * rater_count
    expected = sum((count / total_ratings) ** 2 for count in category_counts.values())
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def krippendorff_alpha_nominal(ratings: list[list[Hashable | None]]) -> float:
    """Krippendorff's alpha for nominal ratings with missing values allowed."""
    pairs: list[tuple[Hashable, Hashable]] = []
    pooled: list[Hashable] = []
    for unit in ratings:
        observed = [value for value in unit if value is not None]
        pooled.extend(observed)
        for i, left in enumerate(observed):
            for right in observed[i + 1 :]:
                pairs.append((left, right))
    if not pairs or len(pooled) < 2:
        return 0.0
    observed_disagreement = sum(left != right for left, right in pairs) / len(pairs)
    counts = Counter(pooled)
    total = len(pooled)
    expected_agreement = sum(count * (count - 1) for count in counts.values()) / (total * (total - 1))
    expected_disagreement = 1.0 - expected_agreement
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement
