#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/vipragsent-repro}"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    assert torch.cuda.get_device_properties(0).total_memory >= 18 * 2**30
PY

python scripts/16_download_models.py --cache-root "$HF_HOME"
