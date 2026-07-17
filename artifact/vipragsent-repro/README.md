# ViPragSent Experiment Evidence Bundle

This package contains the evaluated ViPragSent experiments, their machine-readable evidence, and the scripts that regenerate result tables and figures. It is not a source of inherited manuscript numbers: the authoritative evidence is generated from adjudicated gold labels, prediction JSONL, and trainer manifests in `results/`.

## Scope of the completed study

- **Data:** 12,000 adjudicated ViPragSent records, split into 8,000 train / 2,000 development / 2,000 test examples.
- **Main pragmatic evaluation:** PhoBERT, XLM-R, Sailor-7B, Vistral-7B, GPT-4.1-mini zero-/8-shot, and the ViPragSent multitask model. Encoder and 7B systems have three recorded seeds; API baselines are single-run prompted baselines.
- **Ordinary-task retention:** PhoBERT, XLM-R, and ViPragSent on UIT-VSFC, UIT-VSMEC, and AIVIVN. These datasets are external evaluation benchmarks, not ViPragSent sources.
- **Ablation:** full model, removal of emotion, ordinary-sentiment, rationale, and uncertainty auxiliaries, plus the no-multitask PhoBERT reference. Inference-time rationale generation and hard-label distillation are outside this study's scope.
- **Supplementary analyses:** low-resource sarcasm (PhoBERT versus ViPragSent, one registered seed), calibration for systems with pragmatic-polarity confidence scores, confusion analysis, learning curves, measured GPU wall-clock time, paired bootstrap significance, and inter-annotator agreement.

The completed results do not establish that ViPragSent outperforms every baseline. Any manuscript must state the observed comparison faithfully and must not reuse values from the separate draft manuscript package.

## Data governance

ViPragSent text comes solely from the local ViSoBERT export. Raw or processed social-media text is private, non-commercial research material and must not be redistributed until source permission is archived. See `configs/data_governance.yaml`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`.

The public benchmark datasets are evaluation-only. Their individual terms and provenance are recorded in `data/manifest/` and the generated `answer/data_provenance/` bundle.

The canonical annotation workbooks are `Duy_Duc_synced-1.xlsx`, `Nhat_Khang_synced-1.xlsx`, and `Quynh_Nhu_synced-1.xlsx`. The Quynh Nhu workbook is verified against the fixed adjudicated gold labels used by all recorded experiments; refreshing reviewer agreement must not alter `data/processed/vipragsent_*.jsonl` or rerun experiments.

## Evidence layout

- `results/*.json`: generated metrics, calibration, significance, agreement, provenance, and readiness metadata.
- `results/predictions/`: imported model/API predictions used to compute the metrics.
- `tables/` and `figures/`: generated Markdown tables and SVG figures.
- `answer/`: portable hand-off bundle regenerated from the authoritative artifacts.
- `cards/`: private checkpoint and adapter archive cards. Checkpoints and adapters are intentionally ignored by Git; their manifests are retained in `answer/run_manifests/`.

## Regenerate the evidence bundle

Run from this directory after the required prediction files and trainer manifests are available:

```powershell
python scripts/12_import_annotation_workbooks.py
python scripts/summarize_ablation_predictions.py
python scripts/19_compute_iaa.py
python scripts/20_paired_significance.py
python scripts/run_experiments.py --vipragsent-test data/processed/vipragsent_test.jsonl --public-data data/processed/all_unified.jsonl --predictions-dir results/predictions --output-dir results --bootstrap-resamples 1000
python scripts/make_artifacts.py
python scripts/21_paper_readiness.py
python scripts/17_collect_answer_bundle.py
python -m pytest -q
```

`scripts/run_full_paper_checklist.sh` is the corresponding server-side checklist. It does not silently retrain a model when the recorded prediction files already exist.

## Reporting rules

- Use only values in generated `results/*.json` and tables derived from them.
- Report calibration only for systems with stored confidence scores; missing confidence is `N/A`, not zero calibration error.
- Treat the low-resource study as exploratory because it has one registered seed per budget.
- Report the observed inter-annotator agreement and the adjudication protocol; do not describe generated rationales as manually faithfulness-verified.
