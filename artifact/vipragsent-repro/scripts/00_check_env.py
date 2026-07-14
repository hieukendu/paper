from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.models.train_guard import detect_vram_gb
from vipragsent.utils.env import env_bool, load_env_file


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def env_present(name: str) -> bool:
    value = os.getenv(name, "").strip()
    if not value:
        return False
    return not (value.startswith("<") and value.endswith(">"))


def build_status() -> dict:
    load_env_file(ROOT / ".env")
    data_root = Path(os.getenv("DATA_ROOT", str(ROOT / "data")))
    local_visobert = Path(os.getenv("LOCAL_VISOBERT_EXPORT", r"D:\hf_cache\exports\visobert_12000_api"))
    return {
        "project_root": str(ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "pytest": package_version("pytest"),
        },
        "paths": {
            "data_root": {"path": str(data_root), "exists": data_root.exists()},
            "local_visobert_export": {"path": str(local_visobert), "exists": local_visobert.exists()},
            "env_file": {"path": str(ROOT / ".env"), "exists": (ROOT / ".env").exists()},
            "env_example": {"path": str(ROOT / ".env.example"), "exists": (ROOT / ".env.example").exists()},
        },
        "api": {
            "azure_openai_endpoint_present": env_present("AZURE_OPENAI_ENDPOINT"),
            "azure_openai_key_present": env_present("AZURE_OPENAI_API_KEY"),
            "azure_openai_label_deployment_present": env_present("AZURE_OPENAI_DEPLOYMENT_LABEL") or env_present("AZURE_OPENAI_DEPLOYMENT"),
            "azure_openai_rationale_deployment_present": env_present("AZURE_OPENAI_DEPLOYMENT_RATIONALE") or env_present("AZURE_OPENAI_DEPLOYMENT"),
            "api_system_prefix": os.getenv("VIPRAGSENT_API_SYSTEM_PREFIX", "gpt41_mini"),
            "hf_token_present": env_present("HF_TOKEN"),
            "kaggle_credentials_present": env_present("KAGGLE_USERNAME") and env_present("KAGGLE_KEY"),
        },
        "safety": {
            "run_mode": os.getenv("RUN_MODE", "setup_only"),
            "enable_fine_tuning": env_bool("ENABLE_FINE_TUNING", False),
            "enable_social_crawl": env_bool("ENABLE_SOCIAL_CRAWL", False),
            "local_training_allowed": env_bool("LOCAL_TRAINING_ALLOWED", False),
        },
        "hardware": {
            "detected_vram_gb": detect_vram_gb(),
            "local_gpu_name": os.getenv("LOCAL_GPU_NAME", "unknown"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check setup-only environment status.")
    parser.add_argument("--json", action="store_true", help="Print JSON status. This is the default.")
    parser.parse_args()
    print(json.dumps(build_status(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
