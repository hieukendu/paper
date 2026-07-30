#!/usr/bin/env bash
set -euo pipefail

# Real-data encoder protocol for the 20 GB H100 allocation.  Every command
# writes a checkpoint, history/run manifest and prediction JSONL.  Results are
# subsequently computed from those JSONL files, not from manuscript values.
ROOT="${1:-/root/vipragsent-repro}"
cd "$ROOT"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"

test "$(wc -l < data/generated/rationales.jsonl)" -eq 8000 || {
  echo "Expected 8000 train-only rationales; refusing incomplete run." >&2
  exit 2
}
python scripts/14_prepare_experiment_data.py

predict_public () {
  local model="$1" system="$2" seed="$3" checkpoint="$4"; shift 4
  for dataset in uit_vsfc uit_vsmec aivivn_2019; do
    python scripts/train_multitask_encoder.py \
      --model-id "$model" --system "$system" --seed "$seed" --checkpoint "$checkpoint" \
      --predict-data "data/processed/public_eval/${dataset}_test.jsonl" \
      --prediction-output "results/predictions/ordinary_sentiment/${dataset}/${system}/${seed}.jsonl" \
      "$@"
  done
}

SEEDS=(20260520 20260521 20260522)
for seed in "${SEEDS[@]}"; do
  python scripts/train_multitask_encoder.py \
    --model-id "$HF_HOME/models/phobert-base" --system phobert_finetune --seed "$seed" \
    --batch-size 8 --grad-accum 4 --bf16 --no-rationale --no-emotion --no-polarity
  predict_public "$HF_HOME/models/phobert-base" phobert_finetune "$seed" "outputs/phobert_finetune/${seed}/best.pt" --no-rationale --no-emotion --no-polarity --batch-size 16 --bf16

  python scripts/train_multitask_encoder.py \
    --model-id "$HF_HOME/models/phobert-base" --system vipragsent_full --seed "$seed" \
    --batch-size 8 --grad-accum 4 --bf16
  predict_public "$HF_HOME/models/phobert-base" vipragsent_full "$seed" "outputs/vipragsent_full/${seed}/best.pt" --batch-size 16 --bf16

  python scripts/train_multitask_encoder.py \
    --model-id "$HF_HOME/models/xlm-roberta-large" --system xlmr_large --seed "$seed" \
    --batch-size 2 --grad-accum 16 --bf16
  predict_public "$HF_HOME/models/xlm-roberta-large" xlmr_large "$seed" "outputs/xlmr_large/${seed}/best.pt" --batch-size 4 --bf16
done

# Q2: one fixed seed for resource-aware ablations.  Each row is a separate
# trained model; the summary script records the single-seed scope.
python scripts/train_multitask_encoder.py --model-id "$HF_HOME/models/phobert-base" --system vipragsent_no_rationale --seed 20260520 --batch-size 8 --grad-accum 4 --bf16 --no-rationale
python scripts/train_multitask_encoder.py --model-id "$HF_HOME/models/phobert-base" --system vipragsent_no_emotion --seed 20260520 --batch-size 8 --grad-accum 4 --bf16 --no-emotion
python scripts/train_multitask_encoder.py --model-id "$HF_HOME/models/phobert-base" --system vipragsent_no_polarity --seed 20260520 --batch-size 8 --grad-accum 4 --bf16 --no-polarity
python scripts/train_multitask_encoder.py --model-id "$HF_HOME/models/phobert-base" --system vipragsent_no_uncertainty --seed 20260520 --batch-size 8 --grad-accum 4 --bf16 --no-uncertainty

python scripts/15_summarize_training_runs.py
python scripts/summarize_ablation_predictions.py
python scripts/run_experiments.py --vipragsent-test "$ROOT/data/processed/vipragsent_test.jsonl" --external-evaluation-data "$ROOT/data/processed/all_unified.jsonl" --predictions-dir "$ROOT/results/predictions" --output-dir "$ROOT/results" --bootstrap-resamples 1000
python scripts/make_artifacts.py
