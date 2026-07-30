# Repository audit

- Evidence source: generated `results/*.json`, prediction JSONL, evaluator code, and adjudicated split files.
- Gold protocol: 8,000 train / 2,000 dev / 2,000 test records.
- Baselines are immutable imported artifacts; their registry and prediction paths are in `answers/optimized_vipragsent/baseline_registry.csv`.
- Original ViPragSent uses three frozen PhoBERT multitask checkpoints. Reproduction of seed 20260520 exactly matched the recorded pragmatic metrics.
- No dev prediction artifacts existed initially; frozen inference regenerated three dev and three test predictions from the archived checkpoints.
