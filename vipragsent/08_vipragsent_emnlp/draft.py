import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# ============================================================
# CONFIG
# ============================================================

DATASET_ID = "phucdev/ViSoBERT"
TARGET_N = 12_000

# Hugging Face /rows API max là 100, không tăng lên 1000 được.
PAGE_SIZE = 100

# Request chậm lại để tránh 429.
SLEEP_SECONDS = 8

OUT_DIR = Path(r"D:\hf_cache\exports\visobert_12000_api")
OUT_JSONL = OUT_DIR / "fresh_visobert_12000_annotation_ready.jsonl"
OUT_CSV = OUT_DIR / "fresh_visobert_12000_preview.csv"
OUT_SUMMARY = OUT_DIR / "download_summary.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://datasets-server.huggingface.co"


# ============================================================
# HELPERS
# ============================================================

def make_headers() -> Dict[str, str]:
    headers = {}

    # Nếu đã set HF_TOKEN thì request có token, đỡ bị rate-limit hơn.
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def get_json(url: str, params: Dict[str, Any], retries: int = 20) -> Dict[str, Any]:
    last_err = None
    headers = make_headers()

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=90)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")

                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    # Backoff mạnh hơn: 2 phút, 3 phút, 4 phút...
                    wait = min(120 + attempt * 30, 300)

                print(f"[429] Too Many Requests at offset={params.get('offset')}. Waiting {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            wait = min(20 * attempt, 180)
            print(f"[WARN] Request failed attempt {attempt}/{retries}: {repr(e)}")
            print(f"       Waiting {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


def resolve_config_split(dataset_id: str):
    data = get_json(f"{BASE}/splits", {"dataset": dataset_id})
    splits = data.get("splits", [])

    if not splits:
        raise RuntimeError(f"No splits found for {dataset_id}. Response: {data}")

    print("Available splits/configs:")
    for s in splits:
        print(s)

    for s in splits:
        if s.get("split") == "train":
            return s["config"], s["split"]

    s = splits[0]
    return s["config"], s["split"]


def extract_text_from_api_row(api_row: Dict[str, Any]) -> Optional[str]:
    row = api_row.get("row", api_row)

    for key in ["text", "comment", "sentence", "content", "document"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def is_good_text(text: str) -> bool:
    if not text:
        return False

    text = text.strip()

    if len(text) < 10 or len(text) > 700:
        return False

    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return False

    letters = re.findall(r"[A-Za-zÀ-ỹà-ỹĐđ]", text)
    return len(letters) >= 5


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def read_existing_state():
    """
    Đọc file JSONL đã tải trước đó để resume.
    Return:
      saved_count, seen_texts, next_offset
    """
    if not OUT_JSONL.exists():
        return 0, set(), 0

    saved_count = 0
    seen = set()
    max_offset = -PAGE_SIZE

    print("Found existing file, reading for resume:", OUT_JSONL)

    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = item.get("text", "")
            if text:
                seen.add(normalize_for_dedup(text))

            saved_count += 1

            off = item.get("source_offset")
            if isinstance(off, int):
                max_offset = max(max_offset, off)

    next_offset = max_offset + PAGE_SIZE
    if next_offset < 0:
        next_offset = 0

    return saved_count, seen, next_offset


def export_preview_csv(max_rows: int = 500):
    with open(OUT_JSONL, "r", encoding="utf-8") as f_in, open(
        OUT_CSV, "w", encoding="utf-8-sig", newline=""
    ) as f_out:
        writer = None

        for i, line in enumerate(f_in):
            if i >= max_rows:
                break

            item = json.loads(line)

            if writer is None:
                writer = csv.DictWriter(f_out, fieldnames=list(item.keys()))
                writer.writeheader()

            writer.writerow(item)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Dataset:", DATASET_ID)
    print("Target:", TARGET_N)
    print("Page size:", PAGE_SIZE)
    print("Sleep seconds:", SLEEP_SECONDS)
    print("Output:", OUT_JSONL)

    token = os.environ.get("HF_TOKEN")
    print("HF_TOKEN:", "FOUND" if token else "NOT FOUND")

    config, split = resolve_config_split(DATASET_ID)
    print("Using config:", config)
    print("Using split:", split)

    saved, seen, offset = read_existing_state()

    print(f"Resume state: saved={saved:,}, next_offset={offset:,}")

    scanned = offset

    mode = "a" if OUT_JSONL.exists() and saved > 0 else "w"

    with open(OUT_JSONL, mode, encoding="utf-8") as f:
        while saved < TARGET_N:
            params = {
                "dataset": DATASET_ID,
                "config": config,
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            }

            data = get_json(f"{BASE}/rows", params)
            rows = data.get("rows", [])

            if not rows:
                print("No more rows returned. Stop.")
                break

            for api_row in rows:
                scanned += 1
                text = extract_text_from_api_row(api_row)

                if not text or not is_good_text(text):
                    continue

                norm = normalize_for_dedup(text)
                if norm in seen:
                    continue

                seen.add(norm)

                item = {
                    "id": f"fresh_visobert_{saved:05d}",
                    "text": text,
                    "source_dataset": DATASET_ID,
                    "source_method": "huggingface_dataset_viewer_api",
                    "source_config": config,
                    "source_split": split,
                    "source_offset": offset,
                    "platform": "mixed_facebook_tiktok_youtube",

                    # ViPragSent-style labels, annotate sau.
                    "implicit": None,
                    "sarcasm": None,
                    "irony": None,
                    "idiom": None,
                    "code_switch": None,
                    "mocking": None,
                    "polarity": None,
                    "emotion": None,
                    "rationale": None,
                }

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                saved += 1

                if saved % 500 == 0:
                    print(
                        f"Saved {saved:,}/{TARGET_N:,} | "
                        f"scanned≈{scanned:,} | offset={offset:,}",
                        flush=True
                    )

                if saved >= TARGET_N:
                    break

            offset += PAGE_SIZE

            # Chậm lại để tránh 429.
            time.sleep(SLEEP_SECONDS)

    export_preview_csv(max_rows=500)

    summary = {
        "dataset": DATASET_ID,
        "config": config,
        "split": split,
        "target": TARGET_N,
        "saved": saved,
        "scanned_approx": scanned,
        "last_offset": offset,
        "jsonl": str(OUT_JSONL),
        "csv_preview": str(OUT_CSV),
        "sleep_seconds": SLEEP_SECONDS,
        "page_size": PAGE_SIZE,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print("Saved:", saved)
    print("JSONL:", OUT_JSONL)
    print("CSV preview:", OUT_CSV)
    print("Summary:", OUT_SUMMARY)


if __name__ == "__main__":
    main()