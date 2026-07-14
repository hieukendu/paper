from __future__ import annotations

from vipragsent.utils.hashing import sha256_text

PRAGMATIC_LABELING_PROMPT = """You are a Vietnamese pragmatic-sentiment annotator.
Given one social-media comment, propose silver labels for:
implicit_sentiment, sarcasm, irony, idiom_figurative, code_switching,
mocking, intended polarity, and emotion. Output JSON only.
These labels are silver labels and require human review."""

POLARITY_EMOTION_PROMPT = """Decide intended polarity and UIT-VSMEC-style emotion for
the Vietnamese comment. Use intended meaning, not surface praise."""

RATIONALE_PROMPT = """Given a Vietnamese social-media comment and gold labels, write
a 1-2 sentence Vietnamese explanation naming the cues. Do not repeat the comment
verbatim and do not output labels."""

AUDIT_RATIONALE_PROMPT = """Check whether the rationale faithfully supports the gold
labels and cites real cues in the comment. Output accepted/rejected plus notes."""


def prompt_hash(prompt: str) -> str:
    return sha256_text(prompt)
