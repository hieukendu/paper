from __future__ import annotations

import csv
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download

from vipragsent.data.clean_pii import clean_pii
from vipragsent.data.normalize_text import normalize_text
from vipragsent.data.schema import make_record
from vipragsent.utils.hashing import sha256_file
from vipragsent.utils.io import ensure_dir


PUBLIC_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "uit_vsfc": {
        "task": "polarity",
        "preserve_original_splits": True,
        "candidate_sources": ["UIT NLP datasets page", "kietnv/VietnameseDatasets mirrors"],
        "license_status": "manual_review_required",
    },
    "uit_vsmec": {
        "task": "emotion",
        "preserve_original_splits": True,
        "candidate_sources": ["UIT NLP datasets page", "tridm/UIT-VSMEC mirror"],
        "license_status": "manual_review_required",
    },
    "aivivn_2019": {
        "task": "polarity",
        "preserve_original_splits": True,
        "candidate_sources": ["Kaggle AIVIVN-2019", "GitHub mirrors"],
        "license_status": "manual_review_required",
    },
}

VSFC_URLS = {
    "train": {
        "sentences": "https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download",
        "sentiments": "https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download",
        "topics": "https://drive.google.com/uc?id=14MuDtwMnNOcr4z_8KdpxprjbwaQ7lJ_C&export=download",
    },
    "dev": {
        "sentences": "https://drive.google.com/uc?id=1sMJSR3oRfPc3fe1gK-V3W5F24tov_517&export=download",
        "sentiments": "https://drive.google.com/uc?id=1GiY1AOp41dLXIIkgES4422AuDwmbUseL&export=download",
        "topics": "https://drive.google.com/uc?id=1DwLgDEaFWQe8mOd7EpF-xqMEbDLfdT-W&export=download",
    },
    "test": {
        "sentences": "https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download",
        "sentiments": "https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download",
        "topics": "https://drive.google.com/uc?id=1_ArMpDguVsbUGl-xSMkTF_p5KpZrmpSB&export=download",
    },
}

VSFC_SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}
VSMEC_EMOTIONS = {
    0: "enjoyment",
    1: "sadness",
    2: "anger",
    3: "disgust",
    4: "fear",
    5: "surprise",
    6: "other",
}
AIVIVN_SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}
AIVIVN_BINARY_SENTIMENT = {0: "negative", 1: "positive"}


def public_download_plan(enable_network: bool = False) -> dict[str, Any]:
    status = "download_disabled"
    if enable_network:
        status = "network_enabled_but_license_confirmation_required"
    return {
        "status": status,
        "datasets": PUBLIC_DATASET_REGISTRY,
        "note": "This scaffold records candidate sources only; human license/split verification is required.",
    }


def load_env_file(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def download_file(url: str, destination: str | Path) -> Path:
    destination = Path(destination)
    ensure_dir(destination.parent)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "vipragsent-repro/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        handle.write(response.read())
    return destination


def download_uit_vsfc(raw_root: str | Path) -> dict[str, Any]:
    raw_root = ensure_dir(Path(raw_root) / "uit_vsfc")
    records: list[dict[str, Any]] = []
    files: list[str] = []
    for split, urls in VSFC_URLS.items():
        paths = {
            key: download_file(url, raw_root / f"{split}_{key}.txt")
            for key, url in urls.items()
        }
        files.extend(str(path) for path in paths.values())
        sentences = paths["sentences"].read_text(encoding="utf-8").splitlines()
        sentiments = paths["sentiments"].read_text(encoding="utf-8").splitlines()
        for idx, (sentence, sentiment_raw) in enumerate(zip(sentences, sentiments)):
            try:
                polarity = VSFC_SENTIMENT[int(sentiment_raw.strip())]
            except (ValueError, KeyError):
                polarity = None
            text = clean_pii(sentence.strip())
            records.append(
                make_record(
                    text,
                    dataset="uit_vsfc",
                    platform="student_feedback",
                    source_path=str(paths["sentences"]),
                    license_name="unknown_hf_card_manual_review_required",
                    split=split,
                    record_id=f"uit_vsfc_{split}_{idx:06d}",
                    text_normalized=normalize_text(text),
                    labels={"polarity": polarity},
                    label_status="reviewed",
                    created_by="src/vipragsent/data/ingest_public.py",
                    pii_cleaned=True,
                    checksum=sha256_file(paths["sentences"]),
                )
            )
    return {"dataset": "uit_vsfc", "records": records, "files": files, "status": "ok"}


def download_uit_vsmec(raw_root: str | Path, *, token: str | None = None) -> dict[str, Any]:
    raw_root = ensure_dir(Path(raw_root) / "uit_vsmec")
    split_files = {"train": "train.json", "dev": "valid.json", "test": "test.json"}
    records: list[dict[str, Any]] = []
    files: list[str] = []
    for split, filename in split_files.items():
        downloaded = Path(
            hf_hub_download(
                repo_id="tridm/UIT-VSMEC",
                repo_type="dataset",
                filename=filename,
                token=token,
                local_dir=raw_root,
                local_dir_use_symlinks=False,
            )
        )
        files.append(str(downloaded))
        for idx, row in enumerate(iter_json_records(downloaded)):
            text = clean_pii(str(first_present(row, ["sentence", "text", "comment", "content", "Sentence"]) or "").strip())
            if not text:
                continue
            emotion = normalise_emotion(first_present(row, ["emotion", "label", "Emotion", "sentiment"]))
            records.append(
                make_record(
                    text,
                    dataset="uit_vsmec",
                    platform="facebook_comments",
                    source_path=str(downloaded),
                    license_name="unknown_hf_card_manual_review_required",
                    split=split,
                    record_id=f"uit_vsmec_{split}_{idx:06d}",
                    text_normalized=normalize_text(text),
                    labels={"emotion": emotion},
                    label_status="reviewed",
                    created_by="src/vipragsent/data/ingest_public.py",
                    pii_cleaned=True,
                    checksum=sha256_file(downloaded),
                )
            )
    return {"dataset": "uit_vsmec", "records": records, "files": files, "status": "ok"}


def download_aivivn_2019(raw_root: str | Path) -> dict[str, Any]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_root = ensure_dir(Path(raw_root) / "aivivn_2019")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("mcocoz/aivivn-2019", path=str(raw_root), unzip=True, quiet=True)
    train_path = raw_root / "train.csv"
    test_path = raw_root / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train.csv and test.csv from Kaggle dataset mcocoz/aivivn-2019")

    records: list[dict[str, Any]] = []
    train_rows = list(iter_csv_records(train_path))
    rng = random.Random(20260520)
    dev_indices = set(rng.sample(range(len(train_rows)), max(1, int(len(train_rows) * 0.1)))) if train_rows else set()
    for idx, row in enumerate(train_rows):
        split = "dev" if idx in dev_indices else "train"
        record = aivivn_record(row, train_path, split=split, idx=idx)
        if record:
            records.append(record)
    for idx, row in enumerate(iter_csv_records(test_path)):
        record = aivivn_record(row, test_path, split="test", idx=idx)
        if record:
            records.append(record)
    return {
        "dataset": "aivivn_2019",
        "records": records,
        "files": [str(train_path), str(test_path)],
        "status": "ok",
        "notes": "Kaggle source has binary labels and train/test only; dev split was deterministically sampled from train at 10%. This differs from the 3-way split described in the workflow and needs human confirmation.",
    }


def aivivn_record(row: dict[str, Any], path: Path, *, split: str, idx: int) -> dict[str, Any] | None:
    text = first_present(row, ["comment", "review", "text", "content", "sentence", "Comment"])
    if text is None:
        return None
    text = clean_pii(str(text).strip())
    polarity = normalise_aivivn_polarity(first_present(row, ["label", "sentiment", "Label", "Sentiment", "class"]))
    return make_record(
        text,
        dataset="aivivn_2019",
        platform="ecommerce_reviews",
        source_path=str(path),
        license_name="kaggle_dataset_manual_review_required",
        split=split,
        record_id=f"aivivn_2019_{split}_{idx:06d}",
        text_normalized=normalize_text(text),
        labels={"polarity": polarity},
        label_status="reviewed",
        created_by="src/vipragsent/data/ingest_public.py",
        pii_cleaned=True,
        checksum=sha256_file(path),
    )


def iter_csv_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    if isinstance(payload, list):
        for value in payload:
            if isinstance(value, dict):
                yield value
    elif isinstance(payload, dict):
        for key in ("data", "records", "items", "rows"):
            if isinstance(payload.get(key), list):
                for value in payload[key]:
                    if isinstance(value, dict):
                        yield value
                return
        yield payload


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalise_polarity(value: Any) -> str | None:
    if value is None:
        return None


def normalise_aivivn_polarity(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        aliases = {
            "negative": "negative",
            "neg": "negative",
            "0": "negative",
            "positive": "positive",
            "pos": "positive",
            "1": "positive",
            "neutral": "neutral",
            "neu": "neutral",
            "2": "neutral",
        }
        return aliases.get(lowered)
    try:
        return AIVIVN_BINARY_SENTIMENT[int(value)]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        aliases = {
            "negative": "negative",
            "neg": "negative",
            "0": "negative",
            "neutral": "neutral",
            "neu": "neutral",
            "1": "neutral",
            "positive": "positive",
            "pos": "positive",
            "2": "positive",
        }
        return aliases.get(lowered)
    try:
        return AIVIVN_SENTIMENT[int(value)]
    except (ValueError, KeyError, TypeError):
        return None


def normalise_emotion(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        aliases = {
            "enjoyment": "enjoyment",
            "joy": "enjoyment",
            "happy": "enjoyment",
            "sadness": "sadness",
            "sad": "sadness",
            "anger": "anger",
            "angry": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "surprise": "surprise",
            "other": "other",
            "neutral": "other",
        }
        if lowered in aliases:
            return aliases[lowered]
        if lowered.isdigit():
            return VSMEC_EMOTIONS.get(int(lowered))
        return None
    try:
        return VSMEC_EMOTIONS[int(value)]
    except (ValueError, KeyError, TypeError):
        return None


def download_public_datasets(raw_root: str | Path, *, include_aivivn: bool = True, token: str | None = None) -> dict[str, Any]:
    outputs = []
    outputs.append(download_uit_vsfc(raw_root))
    outputs.append(download_uit_vsmec(raw_root, token=token))
    if include_aivivn:
        outputs.append(download_aivivn_2019(raw_root))
    return {
        "status": "ok",
        "datasets": [
            {key: value for key, value in output.items() if key != "records"}
            | {"record_count": len(output["records"])}
            for output in outputs
        ],
        "records": [record for output in outputs for record in output["records"]],
    }
