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
    "phobert-base": "vinai/phobert-base",
    "xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "Sailor-7B": "sail/Sailor-7B",
    "Vistral-7B-Chat": "Viet-Mistral/Vistral-7B-Chat",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=os.getenv("HF_HOME", "/root/hf_cache"))
    parser.add_argument("--skip-7b", action="store_true")
    args = parser.parse_args()
    load_env_file(ROOT / ".env")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    root = Path(args.cache_root) / "models"
    for name, repo_id in MODELS.items():
        if args.skip_7b and name in {"Sailor-7B", "Vistral-7B-Chat"}:
            continue
        try:
            path = snapshot_download(
                repo_id,
                local_dir=root / name,
                token=token,
                ignore_patterns=["*.bin", "tf_model*", "flax_model*", "onnx/*"] if name != "phobert-base" else ["tf_model*", "flax_model*"],
            )
            print(f"{name}: ok -> {path}")
        except Exception as exc:
            print(f"{name}: failed -> {type(exc).__name__}: {exc}")
            if name in {"phobert-base", "xlm-roberta-large"}:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
