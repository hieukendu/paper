# ViPragSent Experiment Evidence Bundle

This package contains the evaluated ViPragSent experiments, their machine-readable evidence, and the scripts that regenerate result tables and figures. It is not a source of inherited manuscript numbers: the authoritative evidence is generated from adjudicated gold labels, prediction JSONL, and trainer manifests in `results/`.

## Scope of the completed study

- **Data:** 12,000 adjudicated ViPragSent records, split into 8,000 train / 2,000 development / 2,000 test examples.
- **Main pragmatic evaluation:** PhoBERT, XLM-R, ViSoBERT, Sailor-7B-Chat, Vistral-7B-Chat, GPT-4.1-mini zero-/8-shot, and the ViPragSent multitask model. Learned encoders and 7B systems have three recorded seeds; API baselines are single-run prompted baselines. ViSoBERT is pinned to revision `196a62afad9cbe4f52a54aabad828b13f0eec59a`; the two 7B Chat pins are documented in `configs/artifact_registry.json` as user-attested 2026-07-14 download-time provenance.
- **External ordinary-task diagnostic:** PhoBERT, XLM-R, ViPragSent, four ablations, and ViSoBERT have retained evaluation artifacts on UIT-VSFC, UIT-VSMEC, and AIVIVN with their recorded system/seed scopes. These datasets are external evaluation benchmarks, not ViPragSent sources.
- **Ablation:** full model, the no-multitask PhoBERT reference, and removal of emotion, ordinary-sentiment, rationale, and uncertainty auxiliaries all have recorded three-seed pragmatic results. Inference-time rationale generation and hard-label distillation are outside this study's scope.
- **Supplementary analyses:** P1 is a descriptive source-stratified sensitivity analysis (1,666 `visobert_local` and 334 VIVID-labelled test records); P2 is a three-seed low-resource sarcasm study (PhoBERT versus ViPragSent at five positive budgets). Calibration, confusion analysis, learning curves, measured GPU wall-clock time, paired-bootstrap diagnostics, and inter-annotator agreement remain separately scoped analyses.

The completed results do not establish that ViPragSent outperforms every baseline. Any manuscript must state the observed comparison faithfully and must not reuse values from the separate draft manuscript package.

## Data governance

The final ViPragSent gold corpus contains 10,000 retained local ViSoBERT-export records and 2,000 adjudicated records identified in their stored metadata as `VIVID_seed_and_irony_generation` candidates. The latter replace earlier ViSoBERT rows and are the rows used by the reported experiments. Raw or processed text is private, non-commercial research material and must not be redistributed until the applicable source permission or license is documented and reviewed. See `configs/data_governance.yaml`, `answer/LICENSE`, and `answer/THIRD_PARTY_NOTICES.md`.

The external benchmark datasets are evaluation-only. Their individual terms and provenance are recorded in `answer/data_provenance/`, and the current license-review status is recorded in `docs/external_dataset_access_and_license_status.md`.

The canonical annotation workbooks are `Duy_Duc_synced-1.xlsx`, `Nhat_Khang_synced-1.xlsx`, and `Quynh_Nhu_synced-1.xlsx`. The Quynh Nhu workbook is verified against the fixed adjudicated gold labels used by all recorded experiments; refreshing reviewer agreement must not alter `data/processed/vipragsent_*.jsonl` or rerun experiments.

## Evidence layout

- `results/*.json`: generated metrics, calibration, significance, agreement, provenance, and readiness metadata.
- `results/predictions/`: imported model/API predictions used to compute the metrics.
- `tables/` and `figures/`: generated Markdown tables and SVG figures.
- `answer/`: portable hand-off bundle regenerated from the authoritative artifacts. Its `reproducibility/artifact_registry.json` links the source repository and private model archives without embedding credentials.
- `cards/`: private checkpoint and adapter archive cards. Checkpoints and adapters are intentionally ignored by Git; their locations and access policy are recorded in `configs/artifact_registry.json`, while run manifests are retained in `answer/run_manifests/`.

## Regenerate the evidence bundle

Run from this directory after the required prediction files and trainer manifests are available:

```powershell
python scripts/12_import_annotation_workbooks.py
python scripts/summarize_ablation_predictions.py
python scripts/19_compute_iaa.py
python scripts/20_paired_significance.py
python scripts/run_p0_p1_p2_experiments.py --bootstrap-resamples 200
python scripts/run_experiments.py --vipragsent-test data/processed/vipragsent_test.jsonl --external-evaluation-data data/processed/all_unified.jsonl --predictions-dir results/predictions --output-dir results --bootstrap-resamples 1000
python scripts/make_artifacts.py
python scripts/22_generate_manuscript_experiment_tables.py
python scripts/21_paper_readiness.py
python scripts/17_collect_answer_bundle.py
python -m pytest -q
```

`scripts/run_full_paper_checklist.sh` is the corresponding server-side checklist. It does not silently retrain a model when the recorded prediction files already exist.

## Reporting rules

- Use only values in generated `results/*.json` and tables derived from them.
- Report calibration only for systems with stored confidence scores; missing confidence is `N/A`, not zero calibration error.
- Treat P1 as a descriptive source-stratified sensitivity analysis, not causal evidence of a source-domain effect or source superiority.
- Treat P2 as exploratory three-seed evidence: its mixed monotonicity across systems and uncertainty do not establish a data-efficiency advantage.
- Report the observed inter-annotator agreement and the adjudication protocol; do not describe generated rationales as manually faithfulness-verified.
