from __future__ import annotations

import re
import unicodedata

SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, *, lowercase: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = SPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text
