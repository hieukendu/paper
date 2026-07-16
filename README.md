# ViPragSent

ViPragSent is a Vietnamese social-media benchmark and reproducibility package for pragmatic sentiment phenomena: implicit sentiment, sarcasm, irony, idiomatic or figurative language, code-switching, and mocking. The project also records intended polarity and emotion as auxiliary tasks.

This repository is the evidence source for the accompanying paper. Reported values must be generated from the JSON predictions and run manifests in this repository; values from an earlier `main.pdf` are not reused as experimental evidence.

## Repository map

- [`artifact/vipragsent-repro/`](artifact/vipragsent-repro/): reproducible data, training, evaluation, and artifact pipeline.
- [`artifact/vipragsent-repro/answer/`](artifact/vipragsent-repro/answer/): compact paper hand-off containing generated results, tables, figures, run manifests, and provenance reports.
- [`artifact/vipragsent-repro/results/`](artifact/vipragsent-repro/results/): machine-readable metrics and prediction files.
- [`artifact/vipragsent-repro/configs/`](artifact/vipragsent-repro/configs/): experiment and data-governance configuration.
- [`artifact/vipragsent-repro/cards/`](artifact/vipragsent-repro/cards/): versioned copies of the Hugging Face model cards.

## Dataset card

### Dataset source and composition

ViPragSent contains 12,000 Vietnamese social-media comments derived solely from the local ViSoBERT export. UIT-VSFC, UIT-VSMEC, and AIVIVN are external ordinary-task evaluation benchmarks; they are not incorporated into ViPragSent.

The adjudicated split is 8,000 train, 2,000 development, and 2,000 test records. Each record follows the unified JSONL schema documented in the reproduction artifact and includes six binary pragmatic labels, polarity, emotion, provenance, and annotation status.

### Annotation

Two reviewers independently annotated 10,000 comparable records, followed by adjudication. The repository reports per-label percent agreement, nominal Cohen's kappa, and nominal Krippendorff's alpha in [`answer/results/annotation_agreement.json`](artifact/vipragsent-repro/answer/results/annotation_agreement.json). The remaining 2,000 records were adjudicator-only replacements and are identified in the annotation provenance report.

### Intended use

The dataset is intended for research on Vietnamese pragmatic sentiment, multi-task classification, calibration, and low-resource learning. It should not be used for decisions about individuals, content moderation without human review, surveillance, or other high-impact automated decisions.

### Privacy, limitations, and license

The pipeline includes PII cleaning and does not perform new social-media crawling. Social text can still contain sensitive or offensive content, annotation is subjective, and rare labels remain imbalanced. Generated training rationales were not manually audited for faithfulness and must not be described as human-verified explanations.

The redistribution terms of the source ViSoBERT export remain subject to manual confirmation. Until that confirmation is recorded, raw ViPragSent text must be treated as research-only and must not be publicly redistributed. External evaluation datasets retain their own terms and citations.

## Models and checkpoints

Private Hugging Face repositories archive the reusable weights:

- [`Thundergod2007/vipragsent-experiment-checkpoints`](https://huggingface.co/Thundergod2007/vipragsent-experiment-checkpoints): PhoBERT/XLM-R encoder checkpoints, ablations, histories, and manifests.
- [`Thundergod2007/vipragsent-sailor-7b-qlora`](https://huggingface.co/Thundergod2007/vipragsent-sailor-7b-qlora): Sailor-7B QLoRA adapter.
- [`Thundergod2007/vipragsent-vistral-7b-qlora`](https://huggingface.co/Thundergod2007/vipragsent-vistral-7b-qlora): Vistral-7B QLoRA adapter.

Base-model weights are not redistributed. Load each artifact with the exact base model, tokenizer, code revision, and configuration named by its run manifest. The three-seed 7B extension is still running; final aggregate results will be regenerated after all runs complete.

## Reproduction

```bash
cd artifact/vipragsent-repro
python scripts/00_check_env.py
pytest -q
bash scripts/run_full_paper_checklist.sh "$PWD"
```

The full runner is resumable: completed predictions are reused, missing registered runs are executed, metrics are recomputed, and the final hand-off is written to `answer/`. Set `EXTEND_7B_SEEDS=1` only when intentionally running the three-seed 7B extension.

Secrets in `.env`, local model caches, and the `outputs/` checkpoint directory are ignored by Git. Checkpoints are stored separately on Hugging Face.

## Current evidence status

Completed evidence includes three-seed encoder runs, low-resource experiments, implemented multi-task ablations with ordinary-task retention, calibration, confusion analysis, IAA, bootstrap confidence intervals, paired significance tests, provenance manifests, and archived checkpoints. Remaining non-compute actions are confirmation of the ViSoBERT redistribution terms and preparation of the final manuscript from the generated artifacts.
