#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/vipragsent-repro}"
cd "$ROOT"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"

# The recorded 7B protocol retains the same three seeds as the learned encoder
# systems.  Each command writes seed-scoped artifacts, so a failed model/seed
# stops the launcher before any aggregate result is produced.
SEEDS=(20260520 20260521 20260522)
for seed in "${SEEDS[@]}"; do
  python scripts/train_qlora_sft.py --model-id "$HF_HOME/models/Sailor-7B" --model-revision b8b49a0f02073e58db2e42e5811955dfe87ca970 --system sailor_7b_sft --seed "$seed"
  python scripts/train_qlora_sft.py --model-id "$HF_HOME/models/Vistral-7B-Chat" --model-revision d331b64e61b935cc43c2b3010ae9fb4fde599b45 --system vistral_7b_sft --seed "$seed" --attn-implementation sdpa --prediction-batch-size 16
done
python scripts/15_summarize_training_runs.py
python scripts/run_experiments.py --vipragsent-test "$ROOT/data/processed/vipragsent_test.jsonl" --external-evaluation-data "$ROOT/data/processed/all_unified.jsonl" --predictions-dir "$ROOT/results/predictions" --output-dir "$ROOT/results" --bootstrap-resamples 1000
python scripts/make_artifacts.py
