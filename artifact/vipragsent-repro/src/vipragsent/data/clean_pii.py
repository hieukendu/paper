from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_\.]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")


def detect_pii(text: str) -> list[str]:
    found: list[str] = []
    if EMAIL_RE.search(text):
        found.append("email")
    if URL_RE.search(text):
        found.append("url")
    if HANDLE_RE.search(text):
        found.append("handle")
    if PHONE_RE.search(text):
        found.append("phone")
    return found


def clean_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = HANDLE_RE.sub("[USER]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text
