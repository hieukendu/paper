# ViPragSent H100 20GB Runbook

This runbook executes the real-data experiment without claiming the draft paper's mock numbers.

## Current decisions

- Canonical gold: 12,000 records adjudicated by `quynh_nhu`.
- Split: 8,000 train / 2,000 dev / 2,000 test, multilabel/platform stratified.
- Rationale: GPT-4.1-mini on train only. Manual 5% audit was explicitly waived; this deviation must be reported.
- Default seeds: 3 because the school allocation ends 2026-07-15. Use all 5 paper seeds only if time is extended.
- H100 20GB: bf16 encoder fine-tuning; 7B models only with NF4 QLoRA.

## Local preparation

```bash
wc -l data/generated/rationales.jsonl   # must become 8000
python scripts/13_finalize_rationale_dataset.py
python scripts/07_validate_processed.py
pytest -q
```

## Deploy

Place the school private key at `~/.ssh/id_rsa` with mode `600`, then:

```bash
scripts/deploy_to_school_server.sh
```

The deployment installs dependencies and downloads PhoBERT, XLM-R-large, Sailor-7B and Vistral-7B directly on the school server. Gated model failures are non-fatal and must be recorded.

## Priority order on the GPU server

```bash
cd /root/vipragsent-repro
scripts/run_h100_encoders.sh /root/vipragsent-repro
scripts/run_h100_low_resource.sh /root/vipragsent-repro
```

Priority is: PhoBERT baseline, ViPragSent full, XLM-R, Q2 ablations, Q3 low-resource, then gated 7B QLoRA systems. Run API prompted baselines independently; they do not use GPU.

## Final outputs

- Checkpoints and curves: `outputs/{system}/{seed}/`
- Prediction contract: `results/predictions/**/{seed}.jsonl`
- Q1-Q4 JSON: `results/*.json`
- Tables: `tables/*.md`
- Figures: `figures/*.svg`
- Traceable claims: `results/claim_ledger.csv`

Do not copy numeric values from `main.pdf`. Only report values regenerated from prediction JSONL files.
