# Engineering Plan: ViPragSent (08_vipragsent_emnlp)

This plan turns the auto-generated ViPragSent manuscript package into a real,
defensible EMNLP/ARR submission. Every file reference is relative to the project
root (`08_vipragsent_emnlp/`).

---

## 1. Paper summary & current state

**ViPragSent** ("Vietnamese Pragmatic Sentiment") targets pragmatic phenomena in
Vietnamese social media that surface-polarity benchmarks under-cover: implicit
sentiment, sarcasm, irony, idiom/figurative language, code-switching, and
mocking. The method (see [`sections/03_method.tex`](sections/03_method.tex))
extends ~8.4k adjudicated Facebook/TikTok/YouTube comments with six binary
pragmatic labels plus 3-way intended polarity and UIT-VSMEC-7 emotion, then
trains shared backbones (PhoBERT-base, XLM-R-large, Vistral-7B, Sailor-7B) under
a multi-task loss with homoscedastic-uncertainty weighting plus a training-only
chain-of-thought rationale head distilled from GPT-4o-mini. Headline claims:
+5.5 macro-pragmatic F1 over a PhoBERT fine-tune (+7.4 sarcasm, +5.9 implicit),
+3.3 over GPT-4o-mini 8-shot, retained ordinary sentiment (+2.4 UIT-VSMEC), and
ECE halved (0.094 → 0.048).

**Critical state warning: every quantitative result, figure, table, and the
qualitative panel in this package is SYNTHETIC.** All numbers are produced by a
fixed-seed (`20260520`) numerical simulator at
[`scripts/make_artifacts.py`](scripts/make_artifacts.py) — hardcoded base scores
(`BASE_PHEN`, `ORD_BASE`), additive deltas (`SYS_DELTA`, `ORD_DELTA`), a hand-written
ablation list (`ABLATIONS`), a logistic curve for the low-resource slice, a
hardcoded confusion matrix, and Gaussian-sampled "seeds." No dataset has been
collected or annotated, no model trained, no API called. The configs
([`configs/*.yaml`](configs/)) are placeholder stubs (`primary_backbone: TBD`,
`datasets: pending_manifest`, `seeds: [1, 2, 3]` — inconsistent with the paper's
claimed 5 seeds). Tables in `sections/*.tex` are hardcoded inline and can drift
from `results/*.json`. **All numbers and figures must be replaced with real
experimental output before any submission.** [`notes/completion_note.md`](notes/completion_note.md)
confirms this: "Replace every `Synthetic example result` value with real
experiments."

---

## 2. Complete TABLE inventory

Five LaTeX tables. None are auto-generated; all are hardcoded inline and must be
regenerated from real `results/*.json`.

| # | `\label` | Source (file:line) | Metrics / columns reported | Real experiment + results file needed |
|---|----------|--------------------|----------------------------|----------------------------------------|
| T1 | `tab:datasets` | [`sections/04_experimental_setup.tex:19-36`](sections/04_experimental_setup.tex) | Dataset, Task, Domain, Labels, Train/dev/test split, License. Rows: UIT-VSFC (11426/1583/3166), UIT-VSMEC (5548/686/693), AIVIVN-2019 (16087/1788/5454), ViPragSent (8412/2104/2106) | Real data collection + annotation. Counts/licenses from a real `data/manifest/datasets.json` (currently synthetic). Must verify split counts match actually downloaded/annotated data. |
| T2 | `tab:main_pragmatic` | [`sections/05_results.tex:28-58`](sections/05_results.tex) | Per-phenomenon binary macro-F1 (Implicit, Sarcasm, Irony, Idiom, Code-sw., Mocking, Macro-prag) ±95% CI, for 11 systems (PhoBERT-ST, PhoBERT FT, XLM-R, Sailor-7B SFT, Vistral-7B SFT, GPT-4o-mini zero/8-shot, ViPragSent −noaux/−CoT-only/−exp-only, ViPragSent) | Train all 11 systems on real ViPragSent train set; eval on real 2,106-comment test set; 5 seeds; paired-bootstrap CIs. Produces [`results/main_pragmatic.json`](results/main_pragmatic.json). |
| T3 | `tab:ordinary` | [`sections/05_results.tex:81-104`](sections/05_results.tex) | Macro-F1 on UIT-VSFC, UIT-VSMEC, AIVIVN for 8 systems (no CIs shown) | Eval each trained system on the three public test sets. Produces [`results/ordinary_sentiment.json`](results/ordinary_sentiment.json). Add CIs to match T2 rigor. |
| T4 | `tab:ablation` | [`sections/05_results.tex:118-147`](sections/05_results.tex) | Prag. F1, Ord. F1, ECE (×10³), relative Cost, for 8 ablation configs (full, −emotion, −ord-sent, −expl/CoT, −multitask, −uncertainty-weighting, expl-augmented-inference, hard-label-distill) | Controlled single-backbone ablation sweep (dev set). Produces [`results/multitask_ablation.json`](results/multitask_ablation.json). Note: caption says "dev set" and "different backbone from Table 2, hence higher absolute values" — keep that distinction. |
| T5 | `tab:cost` | [`sections/05_results.tex:240-262`](sections/05_results.tex) | GPU-h, Compute $, API $, Total $ per system (8 rows). Annotation $1860 dominates | Real compute/API accounting logged during T2/T4 runs. Produces [`results/cost_breakdown.json`](results/cost_breakdown.json). |

---

## 3. Complete FIGURE inventory

Eight figures. Figures 1 and 8 are hand-drawn diagrams (no result data);
Figures 2–7 plot simulated arrays. All produced by drawing functions in
[`scripts/make_artifacts.py`](scripts/make_artifacts.py).

| # | `\label` | Producing function (make_artifacts.py) | What it plots | Real data source needed |
|---|----------|----------------------------------------|---------------|--------------------------|
| F1 | `fig:pipeline` ([`05_results.tex:23`](sections/05_results.tex)) | `draw_pipeline()` (L400-431) | Static architecture box diagram (text→backbone→encoder→heads→loss→prediction) | None — schematic. Keep, but verify it matches final method. No numeric data. |
| F2 | `fig:perphen` ([`05_results.tex:78`](sections/05_results.tex)) | `draw_per_phenomenon()` (L436-461) | Grouped bars: per-phenomenon macro-F1 for 5 systems (phobert, xlmr, vistral, gpt4o_fs, vipragsent), ±95% CI | Reads `main_results` (→ [`results/main_pragmatic.json`](results/main_pragmatic.json)). Same source as T2. |
| F3 | `fig:mtl` ([`05_results.tex:157`](sections/05_results.tex)) | `draw_mtl_vs_st()` (L466-496) | Paired bars + arrows: single-task PhoBERT vs ViPragSent per phenomenon | Reads `main_results` (`phobert_st` and `vipragsent` rows of [`results/main_pragmatic.json`](results/main_pragmatic.json)). Note: claim_ledger maps `fig:mtl` to `multitask_ablation.json` but the code actually pulls from `main_pragmatic` — **fix this ledger mismatch**. |
| F4 | `fig:sarcasm_lr` ([`05_results.tex:179`](sections/05_results.tex)) | `draw_low_resource()` (L501-523) | Sarcasm F1 vs #labelled sarcasm examples (log2 x: 64,128,256,512,1024,2048) for 5 systems | Real low-resource ablation: retrain on subsampled sarcasm positives. Produces [`results/low_resource_sarcasm.json`](results/low_resource_sarcasm.json). |
| F5 | `fig:confusion` ([`05_results.tex:204`](sections/05_results.tex)) | `draw_confusion()` (L528-551) | Row-normalised 6×6 confusion matrix on pragmatic polarity (positive, negative, neutral, ironic-pos, sarcastic-neg, mocking) | Real confusion counts from ViPragSent predictions on test set. Produces [`results/error_confusion.json`](results/error_confusion.json). |
| F6 | `fig:expcurve` ([`05_results.tex:213`](sections/05_results.tex)) | `draw_explanation_curves()` (L556-577) | Test pragmatic F1 vs epoch (1-10) for 3 training regimes (direct, CoT-only, full) | Real per-epoch eval logs (hardcoded arrays `base/cot/full` in code). Needs a new `results/learning_curves.json` (currently NOT in claim_ledger / no result file; ledger wrongly points `fig:expcurve` to `main_pragmatic.json`). |
| F7 | `fig:calibration` ([`05_results.tex:223`](sections/05_results.tex)) | `draw_calibration()` (L582-604), helper `make_reliability()` (L257-276) | Reliability diagrams (3 panels: PhoBERT, Vistral, ViPragSent) with ECE annotations | Real per-bin confidence/accuracy from polarity-head probabilities. Produces [`results/calibration.json`](results/calibration.json). |
| F8 | `fig:qualitative` ([`06_analysis.tex:14`](sections/06_analysis.tex)) | `draw_qualitative()` (L609-642) | Two example cards with system predictions (text hardcoded in function) | Real adjudicated test examples + real model predictions. Source [`data/generated/qualitative.jsonl`](data/generated/qualitative.jsonl) (currently 2 synthetic rows). |

---

## 4. Data → artifact dependency map

```
REAL EXPERIMENTS (to run)                RESULTS FILES                  TABLES / FIGURES
-------------------------                -------------                  ----------------
E0  Data collection + annotation  ─────► data/manifest/datasets.json ─► T1 tab:datasets
    (8.4k/2.1k/2.1k pragmatic)            data/.../*.jsonl (annotations)  (+ annotation α stats in
                                                                          04/appendix B prose)

E1  Train 11 systems, eval test   ─────► results/main_pragmatic.json ──► T2 tab:main_pragmatic
    (PhoBERT-ST/FT, XLM-R, Sailor,                                        F2 fig:perphen
     Vistral, GPT-4o-mini zs/8s,                                          F3 fig:mtl (phobert_st vs vipragsent)
     ViPragSent + 3 inference variants)

E2  Eval same systems on 3 public ─────► results/ordinary_sentiment.json► T3 tab:ordinary
    sentiment test sets

E3  Single-backbone ablation      ─────► results/multitask_ablation.json► T4 tab:ablation
    sweep (8 configs, dev set)

E4  Low-resource sarcasm sweep    ─────► results/low_resource_sarcasm.json►F4 fig:sarcasm_lr
    (subsample 64..2048 positives)

E5  Confusion / error analysis    ─────► results/error_confusion.json ──► F5 fig:confusion

E6  Per-epoch eval logging        ─────► results/learning_curves.json* ─► F6 fig:expcurve
    (during E1 ViPragSent runs)           (*NEW file — not yet in ledger)

E7  Calibration (polarity head)   ─────► results/calibration.json ──────► F7 fig:calibration

E8  Compute/API accounting        ─────► results/cost_breakdown.json ───► T5 tab:cost
    (logged across E1-E7)

E9  Qualitative case selection    ─────► data/generated/qualitative.jsonl►F8 fig:qualitative

F1 fig:pipeline = schematic, no data dependency.
```

Note: `results/calibration.json` is the source for F7 only; `result_schema.json`
([`results/result_schema.json`](results/result_schema.json)) is a generic
per-run template (status=pending) and is currently NOT used by any table/figure —
adopt it as the canonical per-run record format for raw seed dumps.

---

## 5. Experiments to run

### Datasets to obtain (per [`configs/datasets.yaml`](configs/datasets.yaml) — currently `pending_manifest`)

The config is a stub; it must be filled with concrete entries. Required corpora:

1. **UIT-VSFC** (`vannguyen2018uitvsfc`) — 3-way student-feedback sentiment.
   Research-only license (request access). Splits 11426/1583/3166.
2. **UIT-VSMEC** (`ho2020uitvsmec`) — 7-way Facebook emotion. Research-only.
   Splits 5548/686/693.
3. **AIVIVN-2019** (`aivivn2019`) — 3-way e-commerce review sentiment. Public
   competition release. Splits 16087/1788/5454.
4. **ViPragSent (new, to be built)** — ~12,622 comments from Facebook (46%),
   TikTok (31%), YouTube (23%), Jan–Nov 2025; 3,209 re-used from UIT-VSFC/VSMEC
   test pools + 9,413 fresh. Final annotated split 8412/2104/2106. CC-BY-NC-4.0.
   Note source_note.md scoped this at 10k–15k comments; the manuscript commits to
   12,622. The pilot success bar (`notes/source_note.md`): start with 2,000
   examples, sarcasm + implicit only, succeed if multi-task transfer adds ≥4
   macro-F1 over single-task.

Pre-processing per [`sections/04_experimental_setup.tex`](sections/04_experimental_setup.tex):
VnCoreNLP word segmentation (`nguyen2017vntokenizer`), ViSoLex lexical
normalisation (`vlexnorm2024`), ViHSD/ViHOS toxic pre-filtering.

### Annotation protocol (E0)
- 2 L1 Vietnamese annotators per comment + 1 adjudicator (linguistics PhD).
- 9-field tag: 6 binary pragmatic flags (implicit, sarcasm, irony, idiom,
  code_switch, mocking) + 3-way intended polarity + UIT-VSMEC-7 emotion +
  optional 1–2 sentence rationale. Rubric in [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex).
- Targets: Krippendorff's α ≥ 0.71 overall (paper claims per-field
  0.72/0.74/0.66/0.69/0.84/0.61), Cohen's κ ≥ 0.69 on the sarcasm-flip decision.
  Re-annotate batches with pre-adjudication α < 0.55.

### Models / backbones (per [`configs/models.yaml`](configs/models.yaml) — currently `TBD`)
Fill in: **PhoBERT-base** (`phobert2020`, default deployment backbone),
**XLM-R-large** (`conneau2020xlmr`), **Vistral-7B** (`vistral2024`, headline
backbone, QLoRA), **Sailor-7B** (`sailor2024`, QLoRA). GPT-4o-mini
(`openai2024gpt4o`) via API for rationale generation + prompted baseline.

### Baselines to implement (11 systems, real names from the paper)
1. PhoBERT (single-task) — one head per phenomenon.
2. PhoBERT fine-tune — single 6-way multi-label head (the primary reference point).
3. XLM-R-large.
4. Sailor-7B SFT (QLoRA r=16, α=32).
5. Vistral-7B SFT (QLoRA r=16, α=32).
6. GPT-4o-mini zero-shot.
7. GPT-4o-mini 8-shot (fixed in-context demos from train set).
8. ViPragSent − no auxiliary loss.
9. ViPragSent − CoT-only (read labels off rationale).
10. ViPragSent − explanation-only.
11. **ViPragSent (full)** — multi-task heads + uncertainty weighting + rationale
    head dropped at inference.

### Training protocol (per [`configs/training.yaml`](configs/training.yaml) — currently `seeds:[1,2,3]`, FIX to 5)
- PhoBERT/XLM-R: AdamW, lr 2e-5, batch 32, 10 epochs, early stop on dev
  macro-pragmatic F1, seq len 128, fp16, grad clip 1.0.
- Vistral/Sailor: QLoRA r=16 α=32 dropout 0.05 on q,k,v,o; lr 1e-4; batch 16;
  3 epochs; NF4 quantisation.
- Rationale decoder: 2 layers, 128-d, 4 heads, dropout 0.1, teacher forcing,
  loss weight β=0.3. Uncertainty log-variances init 0, learned jointly.
- GPT-4o-mini: greedy decoding with rationale/classification prompts (Appendix B).

### Evaluation (per [`configs/eval.yaml`](configs/eval.yaml))
- Primary: binary macro-F1 per phenomenon + macro-pragmatic F1.
- Ordinary retention: macro-F1 on UIT-VSFC, UIT-VSMEC, AIVIVN.
- Calibration: ECE, 10 equal-width bins, on pragmatic polarity head; report Brier.
- **5 seeds** (random init, data shuffle, dropout) — must reconcile training.yaml
  (currently 3) with the paper (5).
- Significance: paired bootstrap (`koehn2004bootstrap`), 1000 resamples; 95% CIs.
- IAA: Krippendorff's α, Cohen's κ.

---

## 6. Replacement procedure (step-by-step)

### (a) Run experiments and dump metrics to `results/*.json`
Run E0–E9 (Section 4). Write a real training/eval harness (new
`scripts/train.py`, `scripts/eval.py` — do NOT exist yet) that, per the
"Artifact contract" in [`sections/03_method.tex:99-106`](sections/03_method.tex),
logs per-run: input ID + platform, gold + predicted labels, per-head logits,
rationale, seed, backbone ID, script hash. Dump raw per-seed records following
[`results/result_schema.json`](results/result_schema.json) (fill `metrics.primary`,
`confidence_interval`, `cost`, `seed`, `status:"complete"`), then aggregate into
the five existing result JSONs **preserving their current schemas** so figures
keep working:
- `main_pragmatic.json`: `{metric, seeds, budget, systems:{sys:{phen:{mean,ci95,seeds[]}, macro_prag}}}`
- `ordinary_sentiment.json`: `{metric, datasets, systems:{sys:{ds:{mean,ci95}}}}`
- `multitask_ablation.json`: `[{name, macro_prag_f1, ord_sent_f1, ece, rel_cost}]`
- `low_resource_sarcasm.json`: `{sys:[{n,mean,ci95}]}`
- `error_confusion.json`: `{labels, counts[6][6]}`
- `calibration.json`: `{sys:{ece, brier, bins:[{low,high,n,conf,acc}]}}`
- `cost_breakdown.json`: `{sys:{compute_gpu_h, compute_usd, api_usd, annotation_usd, total_usd}}`
- **NEW** `learning_curves.json` for F6: `{regime:[f1_per_epoch]}` (add to ledger).

### (b) Refactor `make_artifacts.py` to READ results instead of simulating
In [`scripts/make_artifacts.py`](scripts/make_artifacts.py), **delete the
simulator and replace with file readers**. Specific constants/functions to remove
or replace:
- Remove `BASE_PHEN` (L97-104), `SYS_DELTA` (L107-130), `ORD_BASE` (L163),
  `ORD_DELTA` (L165-177), `sample_score()` (L132-136), and the loops at
  L139-158 and L179-196 → replace with `json.load(open(RES/"main_pragmatic.json"))`
  and `.../ordinary_sentiment.json`.
- Remove `ABLATIONS` list (L201-210) → load `multitask_ablation.json`.
- Remove the logistic-curve loop L219-235 → load `low_resource_sarcasm.json`.
- Remove hardcoded `CONFUSION` (L242-249) → load `error_confusion.json`.
- Remove `make_reliability()` (L257-276) and hardcoded `CALIB` ECE/Brier (L278-289)
  → load `calibration.json`.
- Remove hardcoded `COSTS` (L296-321) → load `cost_breakdown.json`.
- Remove the `manifest` dict (L328-348) → real `data/manifest/datasets.json`
  becomes the source of truth (script should NOT overwrite it with synthetic data).
- Remove `QUAL` (L353-370) → load real `data/generated/qualitative.jsonl`.
- The hardcoded per-epoch arrays `base/cot/full` in `draw_explanation_curves()`
  (L559-561) → load new `learning_curves.json`.
- Keep the `draw_*` plotting functions (L400-642) — they are reusable once fed
  real arrays. Keep the figure-style block (L33-58). The RNG seed (L27) becomes
  irrelevant once simulation is gone; remove it.

### (c) Regenerate figures
After (b), run `python scripts/make_artifacts.py` to re-render
[`figures/fig2..fig8.pdf`](figures/) from real data. F1 (`fig1_pipeline.pdf`) is a
schematic — re-verify it matches the final architecture. Confirm F3 reads from
the same source as the ledger says (fix ledger or code — see §7).

### (d) Replace hardcoded inline numbers in each table
The five tables are hand-typed in LaTeX and WILL drift from JSON. Update each,
then **switch to auto-generation**:
- **T1 `tab:datasets`** ([`04_experimental_setup.tex:26-29`](sections/04_experimental_setup.tex)):
  real split counts + verified licenses.
- **T2 `tab:main_pragmatic`** ([`05_results.tex:35-46`](sections/05_results.tex)):
  every `\tiny{$\pm$0.5}` is a synthetic placeholder CI — replace all 11 rows with
  real means + real per-cell CIs.
- **T3 `tab:ordinary`** ([`05_results.tex:88-96`](sections/05_results.tex)):
  8 rows of 3 numbers.
- **T4 `tab:ablation`** ([`05_results.tex:125-132`](sections/05_results.tex)):
  8 rows; note the `\dag` ECE-scaled-by-10³ footnote.
- **T5 `tab:cost`** ([`05_results.tex:247-254`](sections/05_results.tex)):
  8 rows of GPU-h/$/API/total.
- **RECOMMENDATION (prevent drift):** add a `scripts/make_tables.py` that emits
  `.tex` fragments (e.g. `tables/main_pragmatic.tex`) from the JSON, and
  `\input{}` them in the sections instead of inline numbers. This makes
  paper == results by construction and is the single highest-leverage rigor fix.

### (e) Update `claim_ledger.csv` and re-verify every numeric claim in prose
- Fix [`results/claim_ledger.csv`](results/claim_ledger.csv) mismatches:
  `fig:mtl` is mapped to `multitask_ablation.json` but code reads
  `main_pragmatic.json`; `fig:expcurve` is mapped to `main_pragmatic.json` but
  has no real source (add `learning_curves.json`). Add a row per prose claim.
- Re-verify EVERY inline number in prose against the JSON, including:
  abstract (`main.tex:44-57`: +5.5, +7.4, +5.9, +2.4/+0.6/+0.5, +7.2/+3.2,
  −1.5/−6.4, ECE 0.094→0.048); intro findings ([`01_introduction.tex:70-89`](sections/01_introduction.tex):
  91.5/46.7/58.5, the per-phenomenon deltas, −2.7); results prose
  ([`05_results.tex`](sections/05_results.tex): +3.3 over 8-shot, the
  −3.4/−1.2 GPT regressions, 35.6 vs 48.2 @64, +12.6→+7.2 gap, 0.93 fire rate,
  2–5% vs 11–18% confusion); analysis ([`06_analysis.tex`](sections/06_analysis.tex):
  61% vs 19% miss rate, 74% margin, +2.7/+3.0/+1.5 decomposition, 61.5→59.9
  backbone swap, 5.1ms/34.7ms latency); IAA numbers in
  [`appendix/b_annotation_and_prompts.tex:57-64`](appendix/b_annotation_and_prompts.tex)
  and compute totals in [`appendix/a_reproducibility.tex:36-41`](appendix/a_reproducibility.tex)
  (~470 A100-h, $11.3 API).

### (f) Recompile
`latexmk -pdf main.tex` (or the project's existing build). Confirm all
`\ref{tab:*}`/`\ref{fig:*}` resolve and no synthetic markers remain.

---

## 7. Consistency & rigor checks

1. **paper == results == figures**: after auto-generating tables (§6d) and
   figures (§6c), diff every table cell and figure value against its JSON. The
   abstract macro-pragmatic claim "+5.5" must equal `vipragsent.macro_prag −
   phobert.macro_prag` in `main_pragmatic.json`.
2. **Internal inconsistencies to resolve NOW** (present even in the synthetic
   build): the abstract/intro say the headline ViPragSent is the **Vistral**
   backbone (T2 row "ViPragSent (ours, Vistral)") but the intro lists PhoBERT as
   first-named backbone — make backbone-of-record explicit everywhere. T2 absolute
   macro-prag (61.5) ≠ T4 absolute (66.1) by design (different backbone + dev vs
   test) — keep the caption disclaimer but verify it's true once real.
   `configs/training.yaml` says 3 seeds, paper says 5 — fix config to 5.
3. **CIs / seeds / significance**: every T2 cell currently shows an identical
   `±0.5` — replace with genuine per-cell 95% bootstrap CIs over 5 seeds. Run
   paired bootstrap (1000 resamples) for the headline gaps and report p-values or
   CI-of-difference, not just point deltas.
4. **Dataset licenses / checksums / manifest**: [`data/manifest/datasets.json`](data/manifest/datasets.json)
   has NO checksums and synthetic counts; its README promises "source URL,
   license, checksum, count." Add real SHA256 per file, real source URLs, and
   verify UIT-VSFC/UIT-VSMEC research-only terms permit re-annotation + release of
   derived labels (the limitations section already restricts to releasing only
   the new annotations for the 3,209 re-used comments — honor this).
5. **Remove every synthetic marker**: grep the tree for
   `Synthetic example result - replace before submission` (in
   [`notes/visual_plan.md`](notes/visual_plan.md)) and the simulator docstring
   "Numbers come from a fixed-seed reproducible simulator"
   ([`scripts/make_artifacts.py:1-5`](scripts/make_artifacts.py)). Update
   [`notes/completion_note.md`](notes/completion_note.md) blockers once cleared.
6. **Ethics/IRB**: confirm the scraping consent + take-down clause + IRB review
   described in [`sections/08_ethics.tex`](sections/08_ethics.tex) and
   [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex)
   actually happened before release.

---

## 8. Resource estimate

From the proposal feasibility note ([`notes/source_note.md`](notes/source_note.md):
"Medium: 80–140 GPU-hours, 24–48 GB VRAM, plus annotation") and the (synthetic)
manuscript budget ([`appendix/a_reproducibility.tex`](appendix/a_reproducibility.tex):
~470 A100-h):

- **Compute**: Proposal says 80–140 GPU-h; manuscript claims ~470 A100-h across
  all systems × 5 seeds (95h PhoBERT/XLM-R + 240h Vistral/Sailor SFT + 80h
  ablations + 55h inference). Realistic envelope: **~250–470 A100-hours** depending
  on whether the 7B SFT sweeps run at full 5 seeds. VRAM 24–48 GB (QLoRA NF4 fits
  7B on one 48GB card). **Medium** overall, dominated by the two 7B QLoRA models.
- **API**: GPT-4o-mini for ~8.4k rationale generations + zero/8-shot baselines on
  the 2,106 test set ≈ **$11–15 USD** (manuscript: $11.3).
- **Annotation (the binding constraint)**: ~12.6k comments × 2 annotators + 1
  adjudicator at $15/h, 14-week pass ≈ **$1,860 USD** per full corpus pass
  (manuscript figure). Adjudicator throughput ~1,200 comments/week. This is the
  real bottleneck and the longest lead-time item — start E0 first.

---

## 9. Risks & mitigations (ordered by severity)

1. **All results are fabricated (blocker).** Nothing is reproducible; submitting
   as-is is research misconduct. *Mitigation*: execute §6 end-to-end before any
   claim is made; do not cite any current number externally.
2. **Annotation cost + IAA on subjective labels.** Mocking α target is only 0.61;
   sarcasm is 9% / mocking 4% of data — rare classes drive headline claims.
   *Mitigation*: run the 2,000-example sarcasm+implicit pilot first
   (source_note success bar ≥+4 F1); budget adjudication time; report α
   honestly; if mocking α stays low, demote it to a secondary result.
3. **Headline gains may not replicate.** The simulator bakes in +5.5/+7.4; real
   multi-task transfer on rare classes may be smaller or noisier. *Mitigation*:
   pre-register the pilot success criterion; use paired-bootstrap significance,
   not point deltas; be prepared to reframe if gains are <4 F1.
4. **Dataset license / release legality.** UIT-VSFC/UIT-VSMEC are research-only;
   scraped FB/TikTok/YouTube text raises ToS + PII concerns. *Mitigation*: release
   only derived annotations for re-used comments; PII scrubbing + take-down clause
   per ethics section; legal/IRB sign-off before release.
5. **GPT-4o-mini rationale faithfulness + reproducibility.** Rationales are
   model-generated; the model and prompt may change. *Mitigation*: pin model
   version + decoding params + prompt hash (already in the artifact contract);
   keep the 5% faithfulness audit; the ~8.7% unfaithful-drop claim must be a real
   measured number.
6. **Schema / ledger drift.** Tables are hand-typed; claim_ledger has at least two
   wrong mappings (fig:mtl, fig:expcurve). *Mitigation*: auto-generate tables
   (§6d) and add a CI check that every `\label` appears in the ledger with a
   valid trace file.

---

## 10. Execution checklist

- [ ] Fill `configs/datasets.yaml`, `configs/models.yaml`, `configs/training.yaml`
      (set seeds=5), `configs/eval.yaml`, `configs/artifacts.yaml` with real values.
- [ ] E0: collect + PII-scrub + annotate the ViPragSent corpus (8412/2104/2106);
      compute and record real Krippendorff α / Cohen κ; secure IRB + licenses.
- [ ] Build real `scripts/train.py` + `scripts/eval.py` honoring the artifact
      contract (per-run logs, script hash, 5 seeds, bootstrap CIs).
- [ ] E1: train + eval all 11 systems → `results/main_pragmatic.json`.
- [ ] E2: ordinary-sentiment eval → `results/ordinary_sentiment.json`.
- [ ] E3: ablation sweep (8 configs, dev) → `results/multitask_ablation.json`.
- [ ] E4: low-resource sarcasm sweep → `results/low_resource_sarcasm.json`.
- [ ] E5: confusion analysis → `results/error_confusion.json`.
- [ ] E6: per-epoch logging → NEW `results/learning_curves.json`.
- [ ] E7: calibration (ECE/Brier/bins) → `results/calibration.json`.
- [ ] E8: compute/API accounting → `results/cost_breakdown.json`.
- [ ] E9: select real qualitative cases → `data/generated/qualitative.jsonl`.
- [ ] Fill `data/manifest/datasets.json` with real counts, URLs, licenses, SHA256.
- [ ] Refactor `scripts/make_artifacts.py` to READ all JSONs (remove `BASE_PHEN`,
      `SYS_DELTA`, `ORD_*`, `sample_score`, `ABLATIONS`, logistic loop, `CONFUSION`,
      `make_reliability`, `CALIB`, `COSTS`, `manifest`, `QUAL`, RNG seed).
- [ ] Regenerate `figures/fig2..fig8.pdf`; re-verify fig1 schematic.
- [ ] Add `scripts/make_tables.py`; `\input` generated `.tex` for T1–T5.
- [ ] Update all 5 tables and re-verify every prose number in main.tex + all
      sections + both appendices against the JSONs.
- [ ] Fix and complete `results/claim_ledger.csv` (correct fig:mtl, fig:expcurve;
      one row per numeric claim).
- [ ] Remove every "Synthetic ... replace before submission" marker and the
      simulator docstring; update `notes/completion_note.md`.
- [ ] Run consistency gate (paper == results == figures == ledger); recompile
      with `latexmk -pdf main.tex`; confirm no dangling refs.
