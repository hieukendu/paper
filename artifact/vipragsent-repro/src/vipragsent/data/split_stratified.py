from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from vipragsent.data.schema import PRAGMATIC_LABELS


def assign_splits(
    records: list[dict],
    *,
    seed: int = 20260520,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> list[dict]:
    rng = random.Random(seed)
    output = [dict(record) for record in records]
    unassigned = [idx for idx, record in enumerate(output) if record.get("split") in (None, "unassigned")]
    rng.shuffle(unassigned)
    n = len(unassigned)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    for position, idx in enumerate(unassigned):
        if position < train_end:
            split = "train"
        elif position < dev_end:
            split = "dev"
        else:
            split = "test"
        output[idx]["split"] = split
    return output


def assign_multilabel_stratified_splits(
    records: list[dict],
    *,
    seed: int = 20260520,
    train_ratio: float = 0.6667,
    dev_ratio: float = 0.1667,
) -> list[dict]:
    """Greedy multi-label split for small/medium annotation sets.

    This keeps existing train/dev/test assignments and assigns unassigned records
    while approximately balancing size, platform and pragmatic positives.
    """
    rng = random.Random(seed)
    output = [dict(record) for record in records]
    split_names = ["train", "dev", "test"]
    ratios = {"train": train_ratio, "dev": dev_ratio, "test": max(0.0, 1.0 - train_ratio - dev_ratio)}
    unassigned_idx = [idx for idx, record in enumerate(output) if record.get("split") in (None, "unassigned")]
    if not unassigned_idx:
        return output

    target_sizes = _target_counts(len(unassigned_idx), ratios)
    target_label_counts = {
        split: {
            label: round(
                sum(int((output[idx].get("labels") or {}).get(label) in (1, True)) for idx in unassigned_idx)
                * ratios[split]
            )
            for label in PRAGMATIC_LABELS
        }
        for split in split_names
    }
    platforms = sorted({str(output[idx].get("platform") or (output[idx].get("source") or {}).get("platform") or "unknown") for idx in unassigned_idx})
    target_platform_counts = {
        split: {
            platform: round(
                sum(
                    int(str(output[idx].get("platform") or (output[idx].get("source") or {}).get("platform") or "unknown") == platform)
                    for idx in unassigned_idx
                )
                * ratios[split]
            )
            for platform in platforms
        }
        for split in split_names
    }
    current_sizes = {split: 0 for split in split_names}
    current_label_counts = {split: {label: 0 for label in PRAGMATIC_LABELS} for split in split_names}
    current_platform_counts = {split: {platform: 0 for platform in platforms} for split in split_names}

    rng.shuffle(unassigned_idx)
    unassigned_idx.sort(
        key=lambda idx: (
            -sum(int((output[idx].get("labels") or {}).get(label) in (1, True)) for label in PRAGMATIC_LABELS),
            str(output[idx].get("id")),
        )
    )
    for idx in unassigned_idx:
        record = output[idx]
        label_values = {
            label: int((record.get("labels") or {}).get(label) in (1, True))
            for label in PRAGMATIC_LABELS
        }
        platform = str(record.get("platform") or (record.get("source") or {}).get("platform") or "unknown")
        split = min(
            split_names,
            key=lambda candidate: _split_score(
                candidate,
                label_values,
                platform,
                current_sizes,
                current_label_counts,
                current_platform_counts,
                target_sizes,
                target_label_counts,
                target_platform_counts,
            ),
        )
        record["split"] = split
        current_sizes[split] += 1
        for label, value in label_values.items():
            current_label_counts[split][label] += value
        current_platform_counts[split][platform] += 1
    return output


def split_id_sets(records: Iterable[dict]) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_split[str(record.get("split", "unassigned"))].add(str(record.get("id")))
    return by_split


def find_split_overlaps(records: Iterable[dict]) -> dict[str, list[str]]:
    by_split = split_id_sets(records)
    splits = sorted(by_split)
    overlaps: dict[str, list[str]] = {}
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            shared = sorted(by_split[left] & by_split[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared
    return overlaps


def split_counts(records: Iterable[dict]) -> dict[str, dict]:
    counts: dict[str, dict] = {}
    for record in records:
        split = str(record.get("split", "unassigned"))
        bucket = counts.setdefault(
            split,
            {
                "total": 0,
                "labels": {label: 0 for label in PRAGMATIC_LABELS},
                "platforms": defaultdict(int),
            },
        )
        bucket["total"] += 1
        labels = record.get("labels") or {}
        for label in PRAGMATIC_LABELS:
            bucket["labels"][label] += int(labels.get(label) in (1, True))
        platform = str(record.get("platform") or (record.get("source") or {}).get("platform") or "unknown")
        bucket["platforms"][platform] += 1
    return {
        split: {
            "total": value["total"],
            "labels": value["labels"],
            "platforms": dict(value["platforms"]),
        }
        for split, value in counts.items()
    }


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = int(round(total * ratios["train"]))
    dev = int(round(total * ratios["dev"]))
    test = max(0, total - train - dev)
    return {"train": train, "dev": dev, "test": test}


def _split_score(
    split: str,
    label_values: dict[str, int],
    platform: str,
    current_sizes: dict[str, int],
    current_label_counts: dict[str, dict[str, int]],
    current_platform_counts: dict[str, dict[str, int]],
    target_sizes: dict[str, int],
    target_label_counts: dict[str, dict[str, int]],
    target_platform_counts: dict[str, dict[str, int]],
) -> float:
    size_after = current_sizes[split] + 1
    size_overflow = max(0, size_after - target_sizes[split]) * 100.0
    size_gap = abs(target_sizes[split] - size_after)
    label_gap = 0.0
    for label, value in label_values.items():
        after = current_label_counts[split][label] + value
        label_gap += abs(target_label_counts[split][label] - after)
    platform_after = current_platform_counts[split][platform] + 1
    platform_gap = abs(target_platform_counts[split][platform] - platform_after)
    return size_overflow + size_gap + (2.0 * label_gap) + platform_gap
