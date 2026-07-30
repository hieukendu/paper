from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backbone_id: str
    tokenizer: str
    training_status: str = "deferred"


@dataclass(frozen=True)
class PredictionArtifact:
    input_id: str
    seed: int
    backbone_id: str
    labels: dict
    predictions: dict
    logits: dict
    rationale: str | None
    script_hash: str
