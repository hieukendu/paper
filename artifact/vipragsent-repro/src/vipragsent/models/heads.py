from __future__ import annotations

from dataclasses import dataclass

from vipragsent.data.schema import PRAGMATIC_LABELS


@dataclass(frozen=True)
class HeadSpec:
    name: str
    output_dim: int
    loss: str
    train_only: bool = False


HEAD_SPECS = [
    *[HeadSpec(label, 1, "binary_cross_entropy") for label in PRAGMATIC_LABELS],
    HeadSpec("polarity", 3, "cross_entropy"),
    HeadSpec("emotion", 7, "cross_entropy"),
    HeadSpec("rationale", -1, "token_cross_entropy", train_only=True),
]
