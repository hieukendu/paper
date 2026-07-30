#!/usr/bin/env python3
"""Create deterministic inference-only text variants without altering gold data."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.utils.io import read_jsonl, write_jsonl

URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
USER = re.compile(r"(?<!\w)@[\w_]+")
HASHTAG = re.compile(r"(?<!\w)#([\w]+)")
REPEAT = re.compile(r"(.)\1{2,}", flags=re.DOTALL)
SPACE = re.compile(r"\s+")
TEENCODE = {
    "ko": "không", "k": "không", "khong": "không", "hok": "không", "hem": "không",
    "dc": "được", "đc": "được", "duoc": "được", "vs": "với", "cx": "cũng",
    "mk": "mình", "mik": "mình", "m": "mình", "bn": "bạn", "b": "bạn",
    "nx": "nữa", "r": "rồi", "j": "gì", "z": "vậy", "ntn": "như thế nào",
}
TEENCODE_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(token) for token in sorted(TEENCODE, key=len, reverse=True)) + r")(?!\w)", flags=re.IGNORECASE)
EMOJI_WORDS = {
    "😂": " cười ", "🤣": " cười ", "😆": " cười ", "😅": " cười gượng ",
    "🙂": " mỉm cười ", "😊": " vui ", "😡": " tức giận ", "🤬": " tức giận ",
    "😭": " buồn ", "😢": " buồn ", "🙄": " mỉa mai ", "😏": " mỉa mai ",
    "🤡": " mỉa mai ", "💀": " cười ",
}
LAUGH = re.compile(r"(?:=\)+|:\)+|x+d+|h[a-z]{2,}|k+h[a-z]*h[a-z]*)", flags=re.IGNORECASE)


def transform(text: str, variant: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    if variant == "identity":
        return text
    if variant in {"social", "social_lower"}:
        text = URL.sub(" URL ", text)
        text = USER.sub(" USER ", text)
        text = HASHTAG.sub(r" \1 ", text)
        text = REPEAT.sub(r"\1\1", text)
    if variant == "teencode":
        text = TEENCODE_RE.sub(lambda match: TEENCODE[match.group(0).lower()], text)
    if variant == "emoji_words":
        for emoji, word in EMOJI_WORDS.items():
            text = text.replace(emoji, word)
        text = LAUGH.sub(" cười ", text)
    if variant in {"lower", "social_lower"}:
        text = text.lower()
    return SPACE.sub(" ", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["identity", "lower", "social", "social_lower", "teencode", "emoji_words"], required=True)
    args = parser.parse_args()
    rows = []
    for row in read_jsonl(args.input):
        copy = dict(row)
        copy["text"] = transform(str(row["text"]), args.variant)
        rows.append(copy)
    write_jsonl(args.output, rows)
    print({"status": "ok", "variant": args.variant, "records": len(rows), "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
