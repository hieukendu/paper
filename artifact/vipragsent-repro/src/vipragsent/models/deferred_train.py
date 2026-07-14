from __future__ import annotations


def deferred_training_message() -> str:
    return (
        "Training is deferred in setup-only mode. Complete HUMAN_ACTIONS.md, "
        "approve adjudicated data, then run scripts/train.py with explicit guardrail flags."
    )
