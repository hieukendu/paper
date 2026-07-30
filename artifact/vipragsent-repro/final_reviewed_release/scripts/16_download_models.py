from __future__ import annotations

"""Download only runtime files needed by the training scripts."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download

from vipragsent.utils.env import load_env_file


MODELS = {
    "phobert-base": {"repo_id": "vinai/phobert-base", "revision": None},
    "xlm-roberta-large": {"repo_id": "FacebookAI/xlm-roberta-large", "revision": None},
    "ViSoBERT": {"repo_id": "uitnlp/visobert", "revision": "196a62afad9cbe4f52a54aabad828b13f0eec59a"},
    "Sailor-7B-Chat": {"repo_id": "sail/Sailor-7B-Chat", "revision": "19066fae0a8a3ba029c190d8e3dacccf4d1234b8"},
    "Vistral-7B-Chat": {"repo_id": "Viet-Mistral/Vistral-7B-Chat", "revision": "d331b64e61b935cc43c2b3010ae9fb4fde599b45"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=os.getenv("HF_HOME", "/root/hf_cache"))
    parser.add_argument("--skip-7b", action="store_true")
    args = parser.parse_args()
    load_env_file(ROOT / ".env")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    root = Path(args.cache_root) / "models"
    for name, spec in MODELS.items():
        if args.skip_7b and name in {"Sailor-7B-Chat", "Vistral-7B-Chat"}:
            continue
        repo_id = spec["repo_id"]
        revision = spec["revision"]
        try:
            path = snapshot_download(
                repo_id,
                revision=revision,
                local_dir=root / name,
                token=token,
                ignore_patterns=["*.bin", "tf_model*", "flax_model*", "onnx/*"] if name != "phobert-base" else ["tf_model*", "flax_model*"],
            )
            print(f"{name}: ok -> {path} (revision={revision or 'default'})")
        except Exception as exc:
            print(f"{name}: failed -> {type(exc).__name__}: {exc}")
            if name in {"phobert-base", "xlm-roberta-large"}:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
