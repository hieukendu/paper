# ViPragSent Reproduction Scaffold

Setup-only scaffold for the ViPragSent paper: "Implicit Sentiment, Sarcasm, and Figurative Language in Vietnamese Social Media".

This repository prepares the data, annotation, validation, experiment-config and artifact pipeline. It does not fine-tune PhoBERT, XLM-R, Vistral, Sailor, or any other model in this phase.

## Implemented Now

- Repo tree, configs, `.env.example`, `.gitignore`, and human-action checklist.
- Unified JSONL schema for six pragmatic labels, polarity, emotion, provenance and annotation status.
- Local ViSoBERT ingest helpers with PII cleaning, normalization, checksum and manifest scaffolding.
- Public dataset downloader/registry placeholders that require explicit human/license confirmation.
- Annotation batch builder, silver-label scaffold, disagreement/adjudication merge helpers and rationale generation scaffold.
- Metrics, calibration, bootstrap, confusion and pending artifact schemas.
- Guard-railed `scripts/train.py` that refuses to run unless fine-tuning is explicitly enabled and confirmed.
- Tests for schema, PII cleaning, label mapping, split overlap, metrics and artifact schema.

## Deferred

- Real fine-tuning and Q1-Q4 experiment runs.
- Gold labels before human review/adjudication.
- Release of research-only raw text.
- Direct social-media crawling.
- Any numeric claim reported as a final result.

## Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `.env` manually. Never commit `.env`.

For LLM silver labels, rationale generation or no-fine-tuning API baselines, this scaffold can use Azure OpenAI only when you explicitly pass the relevant confirmation flag. Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, and the Azure deployment names `AZURE_OPENAI_DEPLOYMENT_LABEL` / `AZURE_OPENAI_DEPLOYMENT_RATIONALE`. These are deployment names in Azure, not plain model IDs. The default configured deployment label is `gpt-4.1-mini`; the default path still avoids API calls.

## Data Setup

The first source is the local ViSoBERT export:

```text
D:\hf_cache\exports\visobert_12000_api
```

Run safe checks first:

```powershell
python scripts/00_check_env.py
python scripts/01_prepare_sources.py --dry-run
```

Public datasets are configured as fallbacks only. UIT-VSFC, UIT-VSMEC and AIVIVN license/access/split policies must be confirmed by a human before using them.

## JSONL Schema And Labels

Every record uses one unified schema:

- Six binary pragmatic labels: `implicit_sentiment`, `sarcasm`, `irony`, `idiom_figurative`, `code_switching`, `mocking`.
- Intended polarity: `positive`, `neutral`, `negative`.
- Emotion: `enjoyment`, `sadness`, `anger`, `disgust`, `fear`, `surprise`, `other`.
- Missing labels stay `null` and are represented by `available_labels`.
- Only adjudicated records should be treated as final gold data.

## Agent-Assisted Annotation Workflow

```text
unlabeled_pool.jsonl
  -> scripts/03_make_annotation_batches.py
  -> data/annotation/batches/batch_001_input.jsonl
  -> scripts/04_agent_prelabelling.py
  -> data/annotation/agent_silver/*.jsonl
  -> human reviewer_01 + reviewer_02
  -> scripts/05_merge_human_annotations.py
  -> data/annotation/disagreements/*.jsonl
  -> adjudicator review
  -> data/annotation/adjudicated/*.jsonl
  -> data/processed/vipragsent_train/dev/test.jsonl
```

Agent labels are silver labels only. Human review and adjudication are required before final processed JSONL is considered gold.

## Rationale Generation And Audit

Rationales are generated from adjudicated train labels only. They are auxiliary training targets later, not inference outputs. Audit at least 5% and drop unfaithful rationales.

## Experiment Scaffolds Q1-Q4

- Q1: main pragmatic detection and ordinary sentiment retention.
- Q2: multi-task ablations.
- Q3: low-resource sarcasm slices.
- Q4: calibration, confusion and learning-curve artifacts.

The current scripts prepare configs, command plans and pending schemas only.

## Artifact And Claim Ledger Generation

```powershell
python scripts/09_make_artifacts_dryrun.py
```

This creates pending/mock-safe artifacts that are clearly marked `pending`. Do not cite them as real experimental results.

## Safe Checks

```powershell
python scripts/00_check_env.py
pytest
python scripts/07_validate_processed.py --toy
```

## No-Fine-Tuning Experiment Runner

The repo now includes a runnable no-fine-tuning experiment harness. It evaluates
gold JSONL against imported prediction JSONL files, computes the paper metrics,
and generates tables/figures/claim ledger. It does not train models.

Prediction files should follow this layout:

```text
results/predictions/main_pragmatic/{system}/{seed}.jsonl
results/predictions/ordinary_sentiment/{dataset}/{system}/{seed}.jsonl
results/predictions/low_resource_sarcasm/{budget}/{system}/{seed}.jsonl
```

Each prediction JSONL record needs at least:

```json
{"id":"record_id","system":"vipragsent_full","seed":20260520,"predictions":{"implicit_sentiment":0,"sarcasm":0,"irony":0,"idiom_figurative":0,"code_switching":0,"mocking":0,"polarity":"neutral","emotion":"other"}}
```

Run the no-fine-tune suite:

```powershell
python scripts/run_experiments.py
python scripts/make_artifacts.py
```

After human annotation is complete, build gold splits and run all no-fine-tuning
experiments/artifacts:

```powershell
python scripts/run_after_annotation.py
```

To include GPT-4.1-mini API baselines:

```powershell
python scripts/run_after_annotation.py --run-api-baselines --confirm-api-call
```

Human annotation files are expected here:

```text
data/annotation/reviewer_01/*.jsonl
data/annotation/reviewer_02/*.jsonl
data/annotation/adjudicated/*.jsonl
```

Reviewer/adjudicator records may either contain a `labels` object or flat label
fields. Minimal reviewer record:

```json
{"id":"fresh_visobert_00001","annotator_id":"ann_01","labels":{"implicit_sentiment":0,"sarcasm":0,"irony":0,"idiom_figurative":0,"code_switching":0,"mocking":0,"polarity":"neutral","emotion":"other"}}
```

If the two reviewers agree, `run_after_annotation.py` promotes the record to
final adjudicated gold automatically. If they disagree, the final labels must be
provided in `data/annotation/adjudicated/*.jsonl`; unresolved disagreements are
written to `data/annotation/disagreements/all_disagreements.jsonl`.

For a smoke test before adjudicated gold exists:

```powershell
python scripts/run_experiments.py --include-heuristic --allow-silver-gold --limit 200 --bootstrap-resamples 100
python scripts/make_artifacts.py
```

Systems that require fine-tuning/SFT are marked `blocked` until their prediction
JSONL files are imported from the machine that runs training.

The GPT-4.1-mini baseline prediction paths are:

```text
results/predictions/main_pragmatic/gpt41_mini_zero_shot/20260520.jsonl
results/predictions/main_pragmatic/gpt41_mini_8_shot/20260520.jsonl
```

## Enabling Fine-Tuning Later

Fine-tuning remains disabled until human actions are complete. Later cloud runs require:

- Approved/adjudicated data.
- `.env` with `ENABLE_FINE_TUNING=true`.
- `RUN_MODE` not set to `setup_only`.
- `scripts/train.py --confirm-run-training`.
- A suitable GPU or an explicit toy-run override.
