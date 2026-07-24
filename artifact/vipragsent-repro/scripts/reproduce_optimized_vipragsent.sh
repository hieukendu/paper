#!/usr/bin/env bash
# Reproduce the best frozen-weight candidate.  This script never trains or
# updates weights; it only loads archived checkpoints in prediction-only mode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${VIPRAGSENT_CHECKPOINT_ARCHIVE:-$ROOT/../../huggingface/vipragsent-experiment-checkpoints}"
OUT="$ROOT/answers/optimized_vipragsent"
BASE="$ARCHIVE/pretrained/phobert-base"
XLMR_BASE="$ARCHIVE/pretrained/xlm-roberta-large"
VISTRAL_BASE="$ROOT/../../huggingface/vipragsent-vistral-7b-qlora/pretrained/Vistral-7B-Chat"

mkdir -p "$OUT/predictions/dev" "$OUT/predictions/test" "$OUT/configs" "$OUT/logs"
for split in dev test; do
  for seed in 20260520 20260521 20260522; do
    python "$ROOT/scripts/train_multitask_encoder.py" \
      --model-id "$BASE" \
      --local-base-pytorch-bin "$BASE/pytorch_model.bin" \
      --system frozen_reproduction --seed "$seed" \
      --checkpoint "$ARCHIVE/vipragsent_full/$seed/best.pt" \
      --predict-data "$ROOT/data/processed/vipragsent_${split}.jsonl" \
      --prediction-output "$OUT/predictions/$split/reproduction_${seed}.jsonl" \
      --batch-size 64 --bf16
  done
done
python "$ROOT/scripts/frozen_threshold_ensemble.py" fit \
  --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --predictions "$OUT/predictions/dev" \
  --output "$OUT/configs/threshold_ensemble.json"
python "$ROOT/scripts/frozen_code_switch_lexicon.py" fit \
  --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --dev-gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --dev-predictions "$OUT/predictions/dev" \
  --output "$OUT/configs/code_switch_lexicon.json"
python "$ROOT/scripts/frozen_code_switch_lexicon.py" apply \
  --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/test" \
  --base-config "$OUT/configs/threshold_ensemble.json" \
  --rule "$OUT/configs/code_switch_lexicon.json" \
  --output "$OUT/predictions/vipragsent_frozen_lexical_ensemble_test.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" score \
  --gold "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/vipragsent_frozen_lexical_ensemble_test.jsonl" \
  --output "$OUT/frozen_lexical_test_metrics.csv"
mkdir -p "$OUT/predictions/candidates"
python "$ROOT/scripts/frozen_threshold_ensemble.py" apply \
  --config "$OUT/configs/threshold_ensemble.json" --predictions "$OUT/predictions/dev" \
  --output "$OUT/predictions/candidates/threshold_dev.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" apply \
  --config "$OUT/configs/threshold_ensemble.json" --predictions "$OUT/predictions/test" \
  --output "$OUT/predictions/vipragsent_frozen_threshold_ensemble_test.jsonl"
python "$ROOT/scripts/frozen_code_switch_lexicon.py" apply \
  --train "$ROOT/data/processed/vipragsent_train.jsonl" --data "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --predictions "$OUT/predictions/dev" --base-config "$OUT/configs/threshold_ensemble.json" \
  --rule "$OUT/configs/code_switch_lexicon.json" --output "$OUT/predictions/candidates/lexical_dev.jsonl"
python "$ROOT/scripts/frozen_source_thresholds.py" fit \
  --gold "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/dev" \
  --output "$OUT/configs/source_thresholds.json"
python "$ROOT/scripts/frozen_source_thresholds.py" apply \
  --config "$OUT/configs/source_thresholds.json" --data "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --predictions "$OUT/predictions/dev" --output "$OUT/predictions/candidates/source_thresholds_dev.jsonl"
python "$ROOT/scripts/frozen_source_thresholds.py" apply \
  --config "$OUT/configs/source_thresholds.json" --data "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/test" --output "$OUT/predictions/vipragsent_frozen_source_thresholds_test.jsonl"
for EMBED in phobert visobert; do
  if [ "$EMBED" = phobert ]; then
    EMBED_BASE="$BASE"; EMBED_CHECKPOINT="--checkpoint $ARCHIVE/vipragsent_full/20260520/best.pt"
    EMBED_CONFIG="$OUT/configs/embedding_knn.json"
    EMBED_DEV="$OUT/predictions/candidates/knn_dev.jsonl"
    EMBED_TEST="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl"
  else
    EMBED_BASE="$ARCHIVE/pretrained/visobert"; EMBED_CHECKPOINT=""
    EMBED_CONFIG="$OUT/configs/visobert_embedding_knn.json"
    EMBED_DEV="$OUT/predictions/candidates/visobert_knn_dev.jsonl"
    EMBED_TEST="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl"
  fi
  python "$ROOT/scripts/frozen_embedding_knn.py" fit --base "$EMBED_BASE" --base-bin "$EMBED_BASE/pytorch_model.bin" $EMBED_CHECKPOINT \
    --train "$ROOT/data/processed/vipragsent_train.jsonl" --dev-gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
    --dev-predictions "$OUT/predictions/dev" --output "$EMBED_CONFIG"
  python "$ROOT/scripts/frozen_embedding_knn.py" apply --base "$EMBED_BASE" --base-bin "$EMBED_BASE/pytorch_model.bin" $EMBED_CHECKPOINT \
    --train "$ROOT/data/processed/vipragsent_train.jsonl" --data "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/dev" \
    --base-config "$OUT/configs/threshold_ensemble.json" --config "$EMBED_CONFIG" \
    --output "$EMBED_DEV"
  python "$ROOT/scripts/frozen_embedding_knn.py" apply --base "$EMBED_BASE" --base-bin "$EMBED_BASE/pytorch_model.bin" $EMBED_CHECKPOINT \
    --train "$ROOT/data/processed/vipragsent_train.jsonl" --data "$ROOT/data/processed/vipragsent_test.jsonl" --predictions "$OUT/predictions/test" \
    --base-config "$OUT/configs/threshold_ensemble.json" --config "$EMBED_CONFIG" \
    --output "$EMBED_TEST"
done
python "$ROOT/scripts/frozen_label_selector.py" fit --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate threshold="$OUT/predictions/candidates/threshold_dev.jsonl" --candidate lexical="$OUT/predictions/candidates/lexical_dev.jsonl" \
  --candidate phobert_knn="$OUT/predictions/candidates/knn_dev.jsonl" --candidate visobert_knn="$OUT/predictions/candidates/visobert_knn_dev.jsonl" \
  --output "$OUT/configs/four_way_hybrid_selector.json"
python "$ROOT/scripts/frozen_label_selector.py" apply --config "$OUT/configs/four_way_hybrid_selector.json" \
  --candidate threshold="$OUT/predictions/vipragsent_frozen_threshold_ensemble_test.jsonl" --candidate lexical="$OUT/predictions/vipragsent_frozen_lexical_ensemble_test.jsonl" \
  --candidate phobert_knn="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl" --candidate visobert_knn="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl" \
  --output "$OUT/predictions/vipragsent_frozen_four_way_hybrid_test.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" score --gold "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/vipragsent_frozen_four_way_hybrid_test.jsonl" --output "$OUT/frozen_four_way_hybrid_test_metrics.csv"
python "$ROOT/scripts/frozen_weighted_knn.py" fit --base "$BASE" --base-bin "$BASE/pytorch_model.bin" \
  --checkpoint "$ARCHIVE/vipragsent_full/20260520/best.pt" --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --dev-gold "$ROOT/data/processed/vipragsent_dev.jsonl" --dev-predictions "$OUT/predictions/dev" \
  --output "$OUT/configs/weighted_knn.json"
python "$ROOT/scripts/frozen_weighted_knn.py" apply --base "$BASE" --base-bin "$BASE/pytorch_model.bin" \
  --checkpoint "$ARCHIVE/vipragsent_full/20260520/best.pt" --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/dev" \
  --base-config "$OUT/configs/threshold_ensemble.json" --config "$OUT/configs/weighted_knn.json" \
  --output "$OUT/predictions/candidates/weighted_knn_dev.jsonl"
python "$ROOT/scripts/frozen_weighted_knn.py" apply --base "$BASE" --base-bin "$BASE/pytorch_model.bin" \
  --checkpoint "$ARCHIVE/vipragsent_full/20260520/best.pt" --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_test.jsonl" --predictions "$OUT/predictions/test" \
  --base-config "$OUT/configs/threshold_ensemble.json" --config "$OUT/configs/weighted_knn.json" \
  --output "$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" score --gold "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl" --output "$OUT/frozen_weighted_knn_test_metrics.csv"
python "$ROOT/scripts/frozen_label_selector.py" fit --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate threshold="$OUT/predictions/candidates/threshold_dev.jsonl" --candidate lexical="$OUT/predictions/candidates/lexical_dev.jsonl" \
  --candidate phobert_knn="$OUT/predictions/candidates/knn_dev.jsonl" --candidate visobert_knn="$OUT/predictions/candidates/visobert_knn_dev.jsonl" \
  --candidate source_thresholds="$OUT/predictions/candidates/source_thresholds_dev.jsonl" --candidate weighted_knn="$OUT/predictions/candidates/weighted_knn_dev.jsonl" \
  --output "$OUT/configs/six_way_hybrid_selector.json"
python "$ROOT/scripts/frozen_label_selector.py" apply --config "$OUT/configs/six_way_hybrid_selector.json" \
  --candidate threshold="$OUT/predictions/vipragsent_frozen_threshold_ensemble_test.jsonl" --candidate lexical="$OUT/predictions/vipragsent_frozen_lexical_ensemble_test.jsonl" \
  --candidate phobert_knn="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl" --candidate visobert_knn="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl" \
  --candidate source_thresholds="$OUT/predictions/vipragsent_frozen_source_thresholds_test.jsonl" --candidate weighted_knn="$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl" \
  --output "$OUT/predictions/vipragsent_frozen_six_way_hybrid_test.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" score --gold "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/vipragsent_frozen_six_way_hybrid_test.jsonl" --output "$OUT/frozen_six_way_hybrid_test_metrics.csv"
python "$ROOT/scripts/frozen_embedding_knn.py" fit --base "$XLMR_BASE" --base-bin "$XLMR_BASE/pytorch_model.bin" \
  --train "$ROOT/data/processed/vipragsent_train.jsonl" --dev-gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --dev-predictions "$OUT/predictions/dev" --batch-size 32 --output "$OUT/configs/xlmr_embedding_knn.json"
python "$ROOT/scripts/frozen_embedding_knn.py" apply --base "$XLMR_BASE" --base-bin "$XLMR_BASE/pytorch_model.bin" \
  --train "$ROOT/data/processed/vipragsent_train.jsonl" --data "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/dev" \
  --base-config "$OUT/configs/threshold_ensemble.json" --config "$OUT/configs/xlmr_embedding_knn.json" --batch-size 32 \
  --output "$OUT/predictions/candidates/xlmr_knn_dev.jsonl"
python "$ROOT/scripts/frozen_embedding_knn.py" apply --base "$XLMR_BASE" --base-bin "$XLMR_BASE/pytorch_model.bin" \
  --train "$ROOT/data/processed/vipragsent_train.jsonl" --data "$ROOT/data/processed/vipragsent_test.jsonl" --predictions "$OUT/predictions/test" \
  --base-config "$OUT/configs/threshold_ensemble.json" --config "$OUT/configs/xlmr_embedding_knn.json" --batch-size 32 \
  --output "$OUT/predictions/vipragsent_frozen_xlmr_knn_test.jsonl"
python "$ROOT/scripts/frozen_causal_embedding_knn.py" fit --base "$VISTRAL_BASE" --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --dev-gold "$ROOT/data/processed/vipragsent_dev.jsonl" --dev-predictions "$OUT/predictions/dev" --batch-size 32 --pooling mean \
  --output "$OUT/configs/vistral_base_embedding_knn.json"
python "$ROOT/scripts/frozen_causal_embedding_knn.py" apply --base "$VISTRAL_BASE" --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/dev" --base-config "$OUT/configs/threshold_ensemble.json" \
  --config "$OUT/configs/vistral_base_embedding_knn.json" --batch-size 32 --output "$OUT/predictions/candidates/vistral_base_knn_dev.jsonl"
python "$ROOT/scripts/frozen_causal_embedding_knn.py" apply --base "$VISTRAL_BASE" --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_test.jsonl" --predictions "$OUT/predictions/test" --base-config "$OUT/configs/threshold_ensemble.json" \
  --config "$OUT/configs/vistral_base_embedding_knn.json" --batch-size 32 --output "$OUT/predictions/vipragsent_frozen_vistral_base_knn_test.jsonl"
python "$ROOT/scripts/frozen_score_blend.py" fit --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate phobert="$OUT/predictions/candidates/knn_dev.jsonl:_knn_blend" \
  --candidate visobert="$OUT/predictions/candidates/visobert_knn_dev.jsonl:_knn_blend" \
  --candidate weighted="$OUT/predictions/candidates/weighted_knn_dev.jsonl:_weighted_knn" \
  --candidate xlmr="$OUT/predictions/candidates/xlmr_knn_dev.jsonl:_knn_blend" \
  --candidate vistral_base="$OUT/predictions/candidates/vistral_base_knn_dev.jsonl:_causal_knn_blend" \
  --output "$OUT/configs/score_blend_extended.json"
python "$ROOT/scripts/frozen_score_blend.py" apply --config "$OUT/configs/score_blend_extended.json" \
  --candidate phobert="$OUT/predictions/candidates/knn_dev.jsonl:_knn_blend" \
  --candidate visobert="$OUT/predictions/candidates/visobert_knn_dev.jsonl:_knn_blend" \
  --candidate weighted="$OUT/predictions/candidates/weighted_knn_dev.jsonl:_weighted_knn" \
  --candidate xlmr="$OUT/predictions/candidates/xlmr_knn_dev.jsonl:_knn_blend" \
  --candidate vistral_base="$OUT/predictions/candidates/vistral_base_knn_dev.jsonl:_causal_knn_blend" \
  --output "$OUT/predictions/candidates/score_blend_extended_dev.jsonl"
python "$ROOT/scripts/frozen_score_blend.py" apply --config "$OUT/configs/score_blend_extended.json" \
  --candidate phobert="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl:_knn_blend" \
  --candidate visobert="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl:_knn_blend" \
  --candidate weighted="$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl:_weighted_knn" \
  --candidate xlmr="$OUT/predictions/vipragsent_frozen_xlmr_knn_test.jsonl:_knn_blend" \
  --candidate vistral_base="$OUT/predictions/vipragsent_frozen_vistral_base_knn_test.jsonl:_causal_knn_blend" \
  --output "$OUT/predictions/vipragsent_frozen_score_blend_extended_test.jsonl"
python "$ROOT/scripts/frozen_code_switch_rules.py" fit --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --dev-gold "$ROOT/data/processed/vipragsent_dev.jsonl" --dev-predictions "$OUT/predictions/dev" \
  --output "$OUT/configs/code_switch_rules.json"
python "$ROOT/scripts/frozen_code_switch_rules.py" apply --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/dev" --base-config "$OUT/configs/threshold_ensemble.json" \
  --rule "$OUT/configs/code_switch_rules.json" --output "$OUT/predictions/candidates/lexical_rules_dev.jsonl"
python "$ROOT/scripts/frozen_code_switch_rules.py" apply --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_test.jsonl" --predictions "$OUT/predictions/test" --base-config "$OUT/configs/threshold_ensemble.json" \
  --rule "$OUT/configs/code_switch_rules.json" --output "$OUT/predictions/vipragsent_frozen_lexical_rules_test.jsonl"
python "$ROOT/scripts/frozen_label_selector.py" fit --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate threshold="$OUT/predictions/candidates/threshold_dev.jsonl" --candidate lexical="$OUT/predictions/candidates/lexical_dev.jsonl" \
  --candidate lexical_rules="$OUT/predictions/candidates/lexical_rules_dev.jsonl" \
  --candidate phobert_knn="$OUT/predictions/candidates/knn_dev.jsonl" --candidate visobert_knn="$OUT/predictions/candidates/visobert_knn_dev.jsonl" \
  --candidate source_thresholds="$OUT/predictions/candidates/source_thresholds_dev.jsonl" --candidate weighted_knn="$OUT/predictions/candidates/weighted_knn_dev.jsonl" \
  --candidate score_blend="$OUT/predictions/candidates/score_blend_extended_dev.jsonl" \
  --output "$OUT/configs/eight_way_extended_hybrid_selector.json"
python "$ROOT/scripts/frozen_label_selector.py" apply --config "$OUT/configs/eight_way_extended_hybrid_selector.json" \
  --candidate threshold="$OUT/predictions/vipragsent_frozen_threshold_ensemble_test.jsonl" --candidate lexical="$OUT/predictions/vipragsent_frozen_lexical_ensemble_test.jsonl" \
  --candidate lexical_rules="$OUT/predictions/vipragsent_frozen_lexical_rules_test.jsonl" \
  --candidate phobert_knn="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl" --candidate visobert_knn="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl" \
  --candidate source_thresholds="$OUT/predictions/vipragsent_frozen_source_thresholds_test.jsonl" --candidate weighted_knn="$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl" \
  --candidate score_blend="$OUT/predictions/vipragsent_frozen_score_blend_extended_test.jsonl" \
  --output "$OUT/predictions/vipragsent_frozen_eight_way_extended_hybrid_test.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" score --gold "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/vipragsent_frozen_eight_way_extended_hybrid_test.jsonl" --output "$OUT/frozen_eight_way_extended_hybrid_test_metrics.csv"
python "$ROOT/scripts/frozen_label_dependency.py" fit --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --dev-gold "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/candidates/score_blend_extended_dev.jsonl" \
  --probability-suffix _score_blend --output "$OUT/configs/label_dependency.json"
python "$ROOT/scripts/frozen_label_dependency.py" apply --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_dev.jsonl" --predictions "$OUT/predictions/candidates/score_blend_extended_dev.jsonl" \
  --config "$OUT/configs/label_dependency.json" --output "$OUT/predictions/candidates/label_dependency_dev.jsonl"
python "$ROOT/scripts/frozen_label_dependency.py" apply --train "$ROOT/data/processed/vipragsent_train.jsonl" \
  --data "$ROOT/data/processed/vipragsent_test.jsonl" --predictions "$OUT/predictions/vipragsent_frozen_score_blend_extended_test.jsonl" \
  --config "$OUT/configs/label_dependency.json" --output "$OUT/predictions/vipragsent_frozen_label_dependency_test.jsonl"
python "$ROOT/scripts/frozen_label_selector.py" fit --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate threshold="$OUT/predictions/candidates/threshold_dev.jsonl" --candidate lexical="$OUT/predictions/candidates/lexical_dev.jsonl" \
  --candidate lexical_rules="$OUT/predictions/candidates/lexical_rules_dev.jsonl" \
  --candidate phobert_knn="$OUT/predictions/candidates/knn_dev.jsonl" --candidate visobert_knn="$OUT/predictions/candidates/visobert_knn_dev.jsonl" \
  --candidate source_thresholds="$OUT/predictions/candidates/source_thresholds_dev.jsonl" --candidate weighted_knn="$OUT/predictions/candidates/weighted_knn_dev.jsonl" \
  --candidate score_blend="$OUT/predictions/candidates/score_blend_extended_dev.jsonl" --candidate label_dependency="$OUT/predictions/candidates/label_dependency_dev.jsonl" \
  --output "$OUT/configs/nine_way_dependency_hybrid_selector.json"
python "$ROOT/scripts/frozen_label_selector.py" apply --config "$OUT/configs/nine_way_dependency_hybrid_selector.json" \
  --candidate threshold="$OUT/predictions/vipragsent_frozen_threshold_ensemble_test.jsonl" --candidate lexical="$OUT/predictions/vipragsent_frozen_lexical_ensemble_test.jsonl" \
  --candidate lexical_rules="$OUT/predictions/vipragsent_frozen_lexical_rules_test.jsonl" \
  --candidate phobert_knn="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl" --candidate visobert_knn="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl" \
  --candidate source_thresholds="$OUT/predictions/vipragsent_frozen_source_thresholds_test.jsonl" --candidate weighted_knn="$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl" \
  --candidate score_blend="$OUT/predictions/vipragsent_frozen_score_blend_extended_test.jsonl" --candidate label_dependency="$OUT/predictions/vipragsent_frozen_label_dependency_test.jsonl" \
  --output "$OUT/predictions/vipragsent_frozen_nine_way_dependency_hybrid_test.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" score --gold "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/vipragsent_frozen_nine_way_dependency_hybrid_test.jsonl" --output "$OUT/frozen_nine_way_dependency_hybrid_test_metrics.csv"
python "$ROOT/scripts/frozen_source_score_blend.py" fit --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate phobert="$OUT/predictions/candidates/knn_dev.jsonl:_knn_blend" \
  --candidate visobert="$OUT/predictions/candidates/visobert_knn_dev.jsonl:_knn_blend" \
  --candidate weighted="$OUT/predictions/candidates/weighted_knn_dev.jsonl:_weighted_knn" \
  --candidate xlmr="$OUT/predictions/candidates/xlmr_knn_dev.jsonl:_knn_blend" \
  --candidate vistral_base="$OUT/predictions/candidates/vistral_base_knn_dev.jsonl:_causal_knn_blend" \
  --output "$OUT/configs/source_score_blend.json"
python "$ROOT/scripts/frozen_source_score_blend.py" apply --config "$OUT/configs/source_score_blend.json" --data "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate phobert="$OUT/predictions/candidates/knn_dev.jsonl:_knn_blend" \
  --candidate visobert="$OUT/predictions/candidates/visobert_knn_dev.jsonl:_knn_blend" \
  --candidate weighted="$OUT/predictions/candidates/weighted_knn_dev.jsonl:_weighted_knn" \
  --candidate xlmr="$OUT/predictions/candidates/xlmr_knn_dev.jsonl:_knn_blend" \
  --candidate vistral_base="$OUT/predictions/candidates/vistral_base_knn_dev.jsonl:_causal_knn_blend" \
  --output "$OUT/predictions/candidates/source_score_blend_dev.jsonl"
python "$ROOT/scripts/frozen_source_score_blend.py" apply --config "$OUT/configs/source_score_blend.json" --data "$ROOT/data/processed/vipragsent_test.jsonl" \
  --candidate phobert="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl:_knn_blend" \
  --candidate visobert="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl:_knn_blend" \
  --candidate weighted="$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl:_weighted_knn" \
  --candidate xlmr="$OUT/predictions/vipragsent_frozen_xlmr_knn_test.jsonl:_knn_blend" \
  --candidate vistral_base="$OUT/predictions/vipragsent_frozen_vistral_base_knn_test.jsonl:_causal_knn_blend" \
  --output "$OUT/predictions/vipragsent_frozen_source_score_blend_test.jsonl"
python "$ROOT/scripts/frozen_label_selector.py" fit --gold "$ROOT/data/processed/vipragsent_dev.jsonl" \
  --candidate threshold="$OUT/predictions/candidates/threshold_dev.jsonl" --candidate lexical="$OUT/predictions/candidates/lexical_dev.jsonl" \
  --candidate lexical_rules="$OUT/predictions/candidates/lexical_rules_dev.jsonl" \
  --candidate phobert_knn="$OUT/predictions/candidates/knn_dev.jsonl" --candidate visobert_knn="$OUT/predictions/candidates/visobert_knn_dev.jsonl" \
  --candidate source_thresholds="$OUT/predictions/candidates/source_thresholds_dev.jsonl" --candidate weighted_knn="$OUT/predictions/candidates/weighted_knn_dev.jsonl" \
  --candidate score_blend="$OUT/predictions/candidates/score_blend_extended_dev.jsonl" --candidate label_dependency="$OUT/predictions/candidates/label_dependency_dev.jsonl" \
  --candidate source_score_blend="$OUT/predictions/candidates/source_score_blend_dev.jsonl" \
  --output "$OUT/configs/ten_way_source_dependency_hybrid_selector.json"
python "$ROOT/scripts/frozen_label_selector.py" apply --config "$OUT/configs/ten_way_source_dependency_hybrid_selector.json" \
  --candidate threshold="$OUT/predictions/vipragsent_frozen_threshold_ensemble_test.jsonl" --candidate lexical="$OUT/predictions/vipragsent_frozen_lexical_ensemble_test.jsonl" \
  --candidate lexical_rules="$OUT/predictions/vipragsent_frozen_lexical_rules_test.jsonl" \
  --candidate phobert_knn="$OUT/predictions/vipragsent_frozen_knn_ensemble_test.jsonl" --candidate visobert_knn="$OUT/predictions/vipragsent_frozen_visobert_knn_test.jsonl" \
  --candidate source_thresholds="$OUT/predictions/vipragsent_frozen_source_thresholds_test.jsonl" --candidate weighted_knn="$OUT/predictions/vipragsent_frozen_weighted_knn_test.jsonl" \
  --candidate score_blend="$OUT/predictions/vipragsent_frozen_score_blend_extended_test.jsonl" --candidate label_dependency="$OUT/predictions/vipragsent_frozen_label_dependency_test.jsonl" \
  --candidate source_score_blend="$OUT/predictions/vipragsent_frozen_source_score_blend_test.jsonl" \
  --output "$OUT/predictions/vipragsent_frozen_ten_way_source_dependency_hybrid_test.jsonl"
python "$ROOT/scripts/frozen_threshold_ensemble.py" score --gold "$ROOT/data/processed/vipragsent_test.jsonl" \
  --predictions "$OUT/predictions/vipragsent_frozen_ten_way_source_dependency_hybrid_test.jsonl" --output "$OUT/frozen_ten_way_source_dependency_hybrid_test_metrics.csv"
python "$ROOT/scripts/generate_frozen_status_report.py"
