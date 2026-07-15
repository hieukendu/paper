#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/vipragsent-repro}"
cd "$ROOT"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"

# One seed per 7B model is reported separately from the three-seed encoder
# protocol.  A failed gated/downloaded model should not erase encoder results.
python scripts/train_qlora_sft.py --model-id "$HF_HOME/models/Sailor-7B" --system sailor_7b_sft --seed 20260520
python scripts/train_qlora_sft.py --model-id "$HF_HOME/models/Vistral-7B-Chat" --system vistral_7b_sft --seed 20260520 --attn-implementation sdpa --prediction-batch-size 16
python scripts/15_summarize_training_runs.py
python scripts/run_experiments.py --vipragsent-test "$ROOT/data/processed/vipragsent_test.jsonl" --public-data "$ROOT/data/processed/all_unified.jsonl" --predictions-dir "$ROOT/results/predictions" --output-dir "$ROOT/results" --bootstrap-resamples 1000
python scripts/make_artifacts.py
