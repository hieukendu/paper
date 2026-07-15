#!/usr/bin/env bash
set -euo pipefail

# Continue the full protocol only after the encoder/Q2 process supplied by the
# launcher has completed every expected run.  This lets a long H100 schedule
# survive disconnects without converting failed partial outputs into results.
ROOT="${1:-/root/vipragsent-repro}"
ENCODER_PID="${2:?Usage: run_h100_remaining_suite.sh ROOT ENCODER_PID}"
EXPECTED_ENCODER_MANIFESTS=13

cd "$ROOT"
mkdir -p logs

while true; do
  state="$(ps -o stat= -p "$ENCODER_PID" 2>/dev/null | tr -d ' ' || true)"
  case "$state" in
    ""|Z*) break ;;
    *) sleep 60 ;;
  esac
done

manifest_count="$(find outputs -name run_manifest.json -type f | wc -l | tr -d ' ')"
if [[ "$manifest_count" -lt "$EXPECTED_ENCODER_MANIFESTS" ]]; then
  echo "Encoder/Q2 did not complete: expected at least $EXPECTED_ENCODER_MANIFESTS manifests, found $manifest_count." >&2
  exit 1
fi

bash scripts/run_h100_low_resource.sh "$ROOT"
bash scripts/run_h100_qlora.sh "$ROOT"
python scripts/15_summarize_training_runs.py
python scripts/run_experiments.py \
  --vipragsent-test "$ROOT/data/processed/vipragsent_test.jsonl" \
  --public-data "$ROOT/data/processed/all_unified.jsonl" \
  --predictions-dir "$ROOT/results/predictions" \
  --output-dir "$ROOT/results" \
  --bootstrap-resamples 1000
python scripts/make_artifacts.py
python scripts/17_collect_answer_bundle.py
touch logs/full_suite.complete
