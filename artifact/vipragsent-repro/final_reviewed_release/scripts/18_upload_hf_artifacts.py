from __future__ import annotations

"""Archive completed model artifacts to private Hugging Face model repositories.

The script deliberately excludes a running Vistral run unless --include-vistral
is requested.  It uploads only reusable weights and their manifests, never
datasets, environment files, Hugging Face caches, or base-model weights.
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def token_from_environment_or_file(env_path: Path) -> str:
    for name in TOKEN_NAMES:
        if value := os.getenv(name):
            return value
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in TOKEN_NAMES:
                return value.strip().strip("\"").strip("'")
    raise SystemExit("No Hugging Face token found. Set HF_TOKEN or add it to .env.")


def upload_text(api: HfApi, repo_id: str, path_in_repo: str, content: str) -> None:
    api.upload_file(
        path_or_fileobj=BytesIO(content.encode("utf-8")),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {path_in_repo}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sailor-repo")
    parser.add_argument("--checkpoint-repo")
    parser.add_argument("--vistral-repo")
    parser.add_argument("--include-vistral", action="store_true")
    parser.add_argument(
        "--only-vistral",
        action="store_true",
        help="Upload just the Vistral adapter and its completed metadata.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    upload_vistral = args.include_vistral or args.only_vistral
    if upload_vistral and not args.vistral_repo:
        raise SystemExit("Vistral upload requires --vistral-repo.")
    if not args.only_vistral and (not args.sailor_repo or not args.checkpoint_repo):
        raise SystemExit("--sailor-repo and --checkpoint-repo are required unless --only-vistral is used.")

    token = token_from_environment_or_file(ROOT / ".env")
    api = HfApi(token=token)
    repo_ids = [args.vistral_repo] if args.only_vistral else [args.sailor_repo, args.checkpoint_repo]
    if upload_vistral and not args.only_vistral:
        repo_ids.append(args.vistral_repo)
    for repo_id in repo_ids:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if not info.private:
            raise SystemExit(f"Refusing to upload experiment artifacts to public repo: {repo_id}")

    if not args.only_vistral:
        sailor_root = ROOT / "outputs" / "sailor_7b_sft" / "20260520"
        if not (sailor_root / "adapter" / "adapter_model.safetensors").exists():
            raise SystemExit(f"Sailor adapter is missing: {sailor_root}")
        upload_text(
            api,
            args.sailor_repo,
            "README.md",
            """# ViPragSent Sailor-7B QLoRA adapter

Private research artifact. Load the licensed Sailor-7B base model separately,
then attach the files in `adapter/` with PEFT. The accompanying manifest and
history record the seed and training configuration. This repository contains
no base-model weights, datasets, secrets, or raw social-media text.
""",
        )
        print(f"Uploading Sailor adapter to {args.sailor_repo}", flush=True)
        api.upload_large_folder(
            repo_id=args.sailor_repo,
            folder_path=sailor_root,
            repo_type="model",
            num_workers=args.workers,
            print_report=True,
            print_report_every=30,
        )

        upload_text(
            api,
            args.checkpoint_repo,
            "README.md",
            """# ViPragSent experiment checkpoints

Private archival repository for completed encoder checkpoints. Layout mirrors
the experiment systems and low-resource budgets under `outputs/`. Each run
contains `best.pt`, `history.json`, and `run_manifest.json`; load a checkpoint
with the matching code, model ID, tokenizer, and configuration from the
ViPragSent reproduction repository. This repository intentionally excludes
base-model weights, datasets, API credentials, Hugging Face caches, and the
still-running Vistral artifact.
""",
        )
        print(f"Uploading completed encoder checkpoints to {args.checkpoint_repo}", flush=True)
        api.upload_large_folder(
            repo_id=args.checkpoint_repo,
            folder_path=ROOT / "outputs",
            repo_type="model",
            allow_patterns=["**/best.pt", "**/history.json", "**/run_manifest.json"],
            ignore_patterns=["sailor_7b_sft/**", "vistral_7b_sft/**"],
            num_workers=args.workers,
            print_report=True,
            print_report_every=30,
        )

    if upload_vistral:
        vistral_root = ROOT / "outputs" / "vistral_7b_sft" / "20260520"
        if not (vistral_root / "adapter" / "adapter_model.safetensors").exists():
            raise SystemExit(f"Vistral adapter is missing: {vistral_root}")
        upload_text(
        api,
        args.vistral_repo,
        "README.md",
            """# ViPragSent Vistral-7B QLoRA adapter

Private research artifact. Load the licensed Vistral-7B base model separately,
then attach `adapter/` with PEFT. This repository contains no base-model
weights, datasets, secrets, or raw social-media text.
""",
        )
        print(f"Uploading Vistral adapter to {args.vistral_repo}", flush=True)
        api.upload_large_folder(
            repo_id=args.vistral_repo,
            folder_path=vistral_root,
            repo_type="model",
            num_workers=args.workers,
            print_report=True,
            print_report_every=30,
        )
    print("Hugging Face artifact upload complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
