#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/vipragsent-repro}"
cd "$ROOT"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"
SEED=20260520

python scripts/make_low_resource_splits.py --train data/processed/vipragsent_train.jsonl --budgets 64 128 256 512 1024
for budget in 64 128 256 512 1024; do
  train="data/processed/low_resource/sarcasm_${budget}.jsonl"
  python scripts/train_multitask_encoder.py \
    --model-id "$HF_HOME/models/phobert-base" --system phobert_finetune --train "$train" --seed "$SEED" \
    --batch-size 8 --grad-accum 4 --bf16 --no-rationale --no-emotion --no-polarity \
    --output-root "outputs/low_resource/${budget}" --prediction-root "results/predictions/low_resource_sarcasm/${budget}"
  python scripts/train_multitask_encoder.py \
    --model-id "$HF_HOME/models/phobert-base" --system vipragsent_full --train "$train" --seed "$SEED" \
    --batch-size 8 --grad-accum 4 --bf16 \
    --output-root "outputs/low_resource/${budget}" --prediction-root "results/predictions/low_resource_sarcasm/${budget}"
done

python scripts/15_summarize_training_runs.py
python scripts/run_experiments.py --vipragsent-test "$ROOT/data/processed/vipragsent_test.jsonl" --external-evaluation-data "$ROOT/data/processed/all_unified.jsonl" --predictions-dir "$ROOT/results/predictions" --output-dir "$ROOT/results" --bootstrap-resamples 1000
python scripts/make_artifacts.py
