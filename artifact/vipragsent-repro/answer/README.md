# ViPragSent experiment hand-off

Created: 2026-07-18T07:31:22.828574+00:00
Status: `results_generated`

All metrics in `results/`, tables, and figures are generated from prediction JSONL and trainer manifests in this run; no values are copied from `main.pdf`.

Important protocol notes:

- Gold split: 8,000 train / 2,000 dev / 2,000 test adjudicated records.
- Rationale coverage: 8,000 train records; manual faithfulness audit was waived and is documented in data_provenance/.
- Q3 is an exploratory single-seed comparison at 64, 128, 256, 512, and 1,024 sarcasm-positive examples.
- The external ordinary-task diagnostic is an encoder-only comparison; UIT-VSMEC is reported as seven-way emotion macro-F1.
- Calibration is reported only for systems with pragmatic-polarity confidence scores.
- Source code, datasets, prediction JSONL, and pinned external model archives are documented in `reproducibility/artifact_registry.json`.
- `reproducibility/verification_manifest.json` records SHA-256 hashes for the copied artifacts; it is an integrity check, not an experiment rerun.
- `reproducibility/id_set_manifest.json` records non-disclosing ID-set and ordered-ID hashes for the private train/dev/test partitions; it verifies split membership without releasing identifiers or text.
- Run manifests are retained for audit; weights are retrieved from the registered Hugging Face archives rather than copied into this bundle.

Copied files:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- `results/main_pragmatic.json`
- `results/ordinary_sentiment.json`
- `results/multitask_ablation.json`
- `results/low_resource_sarcasm.json`
- `results/calibration.json`
- `results/error_confusion.json`
- `results/learning_curves.json`
- `results/cost_breakdown.json`
- `results/artifact_index.json`
- `results/claim_ledger.csv`
- `results/annotation_agreement.json`
- `results/significance.json`
- `results/paper_readiness.json`
- `tables/annotation_agreement.md`
- `tables/calibration.md`
- `tables/cost_breakdown.md`
- `tables/error_confusion.md`
- `tables/learning_curves.md`
- `tables/low_resource_sarcasm.md`
- `tables/main_pragmatic.md`
- `tables/multitask_ablation.md`
- `tables/ordinary_sentiment.md`
- `tables/pending_summary.md`
- `tables/significance.md`
- `figures/fig1_pipeline.svg`
- `figures/fig2_per_phenomenon.svg`
- `figures/fig4_low_resource_sarcasm.svg`
- `figures/fig5_confusion.svg`
- `figures/fig6_learning_curves.svg`
- `figures/fig7_calibration.svg`
- `figures/README.md`
- `run_manifests/low_resource\1024\phobert_finetune\20260520\run_manifest.json`
- `run_manifests/low_resource\1024\vipragsent_full\20260520\run_manifest.json`
- `run_manifests/low_resource\128\phobert_finetune\20260520\run_manifest.json`
- `run_manifests/low_resource\128\vipragsent_full\20260520\run_manifest.json`
- `run_manifests/low_resource\256\phobert_finetune\20260520\run_manifest.json`
- `run_manifests/low_resource\256\vipragsent_full\20260520\run_manifest.json`
- `run_manifests/low_resource\512\phobert_finetune\20260520\run_manifest.json`
- `run_manifests/low_resource\512\vipragsent_full\20260520\run_manifest.json`
- `run_manifests/low_resource\64\phobert_finetune\20260520\run_manifest.json`
- `run_manifests/low_resource\64\vipragsent_full\20260520\run_manifest.json`
- `run_manifests/phobert_finetune\20260520\run_manifest.json`
- `run_manifests/phobert_finetune\20260521\run_manifest.json`
- `run_manifests/phobert_finetune\20260522\run_manifest.json`
- `run_manifests/sailor_7b_sft\20260520\run_manifest.json`
- `run_manifests/sailor_7b_sft\20260521\run_manifest.json`
- `run_manifests/sailor_7b_sft\20260522\run_manifest.json`
- `run_manifests/vipragsent_full\20260520\run_manifest.json`
- `run_manifests/vipragsent_full\20260521\run_manifest.json`
- `run_manifests/vipragsent_full\20260522\run_manifest.json`
- `run_manifests/vipragsent_no_emotion\20260520\run_manifest.json`
- `run_manifests/vipragsent_no_polarity\20260520\run_manifest.json`
- `run_manifests/vipragsent_no_rationale\20260520\run_manifest.json`
- `run_manifests/vipragsent_no_uncertainty\20260520\run_manifest.json`
- `run_manifests/vistral_7b_sft\20260520\run_manifest.json`
- `run_manifests/vistral_7b_sft\20260521\run_manifest.json`
- `run_manifests/vistral_7b_sft\20260522\run_manifest.json`
- `run_manifests/xlmr_large\20260520\run_manifest.json`
- `run_manifests/xlmr_large\20260521\run_manifest.json`
- `run_manifests/xlmr_large\20260522\run_manifest.json`
- `data_provenance/gold_build_report.json`
- `data_provenance/annotation_import_report.json`
- `data_provenance/rationale_audit_waiver.json`
- `data_provenance/datasets.json`
- `data_provenance/source_registry.json`
- `data_provenance/checksums.json`
- `data_provenance/manifest.json`
- `reproducibility/artifact_registry.json`
- `reproducibility/verification_manifest.json`
- `reproducibility/id_set_manifest.json`
- `reproducibility/local_prediction_split_manifest.json`
- `run_manifests/README.md`
