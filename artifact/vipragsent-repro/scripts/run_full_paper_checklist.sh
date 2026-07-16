#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"
mkdir -p logs

run_if_missing () {
  local marker="$1"; shift
  if [[ -s "$marker" ]]; then
    echo "SKIP existing $marker"
  else
    echo "RUN $*"
    "$@"
  fi
}

# The registered protocol reports the expensive 7B baselines at one seed,
# separately from the three-seed encoder models. An explicit extension remains
# available for a future allocation, but is not silently mixed into this run.
if [[ "${EXTEND_7B_SEEDS:-0}" == "1" ]]; then
  for seed in 20260521 20260522; do
    run_if_missing "results/predictions/main_pragmatic/sailor_7b_sft/${seed}.jsonl" \
      python scripts/train_qlora_sft.py --model-id "$HF_HOME/models/Sailor-7B" --system sailor_7b_sft --seed "$seed"
    run_if_missing "results/predictions/main_pragmatic/vistral_7b_sft/${seed}.jsonl" \
      python scripts/train_qlora_sft.py --model-id "$HF_HOME/models/Vistral-7B-Chat" --system vistral_7b_sft --seed "$seed" --attn-implementation sdpa --prediction-batch-size 16
  done
fi

# Ordinary-task retention for every trained encoder ablation. These are
# prediction-only passes over external benchmarks, not ViPragSent sources.
for system in vipragsent_no_rationale vipragsent_no_emotion vipragsent_no_polarity vipragsent_no_uncertainty; do
  extra=()
  case "$system" in
    vipragsent_no_rationale) extra+=(--no-rationale) ;;
    vipragsent_no_emotion) extra+=(--no-emotion) ;;
    vipragsent_no_polarity) extra+=(--no-polarity) ;;
    vipragsent_no_uncertainty) extra+=(--no-uncertainty) ;;
  esac
  for dataset in uit_vsfc uit_vsmec aivivn_2019; do
    output="results/predictions/ordinary_sentiment/${dataset}/${system}/20260520.jsonl"
    run_if_missing "$output" python scripts/train_multitask_encoder.py \
      --model-id "$HF_HOME/models/phobert-base" --system "$system" --seed 20260520 \
      --checkpoint "outputs/${system}/20260520/best.pt" \
      --predict-data "data/processed/public_eval/${dataset}_test.jsonl" \
      --prediction-output "$output" --batch-size 16 --bf16 "${extra[@]}"
  done
done

# Ordinary retention is evaluated for encoder/ablation models, whose heads
# directly implement the registered polarity/emotion tasks. The generative 7B
# baselines are pragmatic JSON classifiers and are reported only on Q1.

python scripts/15_summarize_training_runs.py
python scripts/summarize_ablation_predictions.py
python scripts/19_compute_iaa.py
python scripts/20_paired_significance.py
python scripts/run_experiments.py \
  --vipragsent-test "$ROOT/data/processed/vipragsent_test.jsonl" \
  --public-data "$ROOT/data/processed/all_unified.jsonl" \
  --predictions-dir "$ROOT/results/predictions" --output-dir "$ROOT/results" \
  --bootstrap-resamples 1000
python scripts/make_artifacts.py
python scripts/21_paper_readiness.py
python scripts/17_collect_answer_bundle.py
touch logs/full_paper_checklist.complete
