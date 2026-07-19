# Detailed EACL Paper Outline — ViPragSent

**Mode:** `ars-outline` (outline only; no manuscript prose)  
**Evidence policy:** all experiment results and result-derived figures/tables listed below are **VERIFIED** from the double-checked `answer/` bundle. “Unverified” labels apply only to citation completion, submission metadata, or future extensions.

## Paper identity

| Field | Current value |
| --- | --- |
| Working title | **ViPragSent: Comparative Evaluation of Vietnamese Pragmatic Sentiment with a Social-Media Source Component.** Anonymous; it does not imply a performance win. |
| Research question | On an adjudicated Vietnamese-language test set covering six pragmatic phenomena and including a documented social-media source component, how do Vietnamese, multilingual, and Vietnamese-social-media encoders, 7B instruction-tuned models, prompted API baselines, and a multi-task encoder compare; and what does the three-seed ablation record about the corresponding configurations? |
| Paper type | Long empirical NLP conference paper |
| Core framing | Evaluation/resource and trade-off analysis, not proposed-model superiority |
| Numerical authority | `../answer/` |
| Method authority | Repository code, `../README.md`, `../configs/`, and retained run manifests |

## Layout status

No page or word-count constraint is active until a target venue and its current call are selected. The outline specifies section purpose and evidence boundaries, not a submission-length target.

## Detailed section outline

## Abstract (one paragraph; write last)

**Purpose:** State the task, data scope, comparison groups, one central measured result, and the limited trade-off finding.

**Paragraph content checklist:**

1. Task and setting: adjudicated Vietnamese-language records with a documented social-media source component; six pragmatic phenomena.
2. Evaluation design: encoders, 7B SFT models, prompted API baselines, and multi-task encoder.
3. Verified main result: Vistral-7B SFT is the strongest recorded system; do not imply ViPragSent is best.
4. Verified analytical result: recorded three-seed multi-task trade-off association, qualified as non-causal and non-universal.
5. Traceability/access boundary: result artifacts and model archive metadata are available; raw text is governed and not represented as publicly released.

**Evidence:** C1–C5 in `PAPER_PLAN.md`.  
**Do not include:** causal claims, SOTA wording, or low-resource conclusions.

## 1. Introduction

**Purpose:** Motivate pragmatic evaluation in Vietnamese social-media text, state the evidence-bounded gap, and give the reader a precise evaluative RQ.

### 1.1 Problem setting and motivation

**Paragraph 1 — social-media language setting.**

- State that Vietnamese social-media text motivates domain-sensitive NLP evaluation.
- Required citation: `nguyen-etal-2023-visobert`.
- Do not assert an unverified claim that no previous Vietnamese social-media resources exist.

**Paragraph 2 — pragmatic evaluation gap.**

- Introduce the paper’s operational scope: implicit sentiment, sarcasm, irony, idiom/figurative language, code-switching, and mocking.
- Frame the gap narrowly: the paper provides a controlled evaluation for this recorded label set, rather than claiming to solve pragmatics generally.
- Evidence: `answer/data_provenance/gold_build_report.json`, `configs/labels.yaml` (**VERIFIED**).

### 1.2 Research question and study boundary

**Paragraph 3 — RQ and comparison design.**

- State the approved RQ verbatim or with no semantic expansion.
- Name the four system groups only: encoders, 7B SFT models, prompted API baselines, multi-task encoder.
- Evidence: `answer/results/main_pragmatic.json`; `answer/results/p0_visobert_baseline.json`; `configs/experiments_p0_p1_p2.yaml` (**VERIFIED**). `configs/experiments_q1.yaml` is retained only as a historical base configuration.

### 1.3 Contributions

**Paragraph 4 — numbered, bounded contributions.**

1. Adjudicated evaluation resource and task definition (C1).
2. Multi-family baseline comparison with recorded seeds/runs (C2).
3. Transparent comparative result, including the fact that Vistral-7B SFT is strongest among evaluated systems (C3).
4. An ablation-based trade-off analysis, not a performance-improvement claim (C4).
5. Result-to-artifact traceability (C5).

**Required evidence:** `answer/results/main_pragmatic.json`, `answer/results/p0_multi_seed_ablation.json`, `answer/results/p0_visobert_baseline.json`, `answer/reproducibility/verification_manifest.json` (**VERIFIED**).

## 2. Related Work

**Purpose:** Position the study in three connected literatures and end with a specific, non-grandiose scope statement.

### 2.1 Vietnamese and Vietnamese-social-media pretrained models

**Paragraph 1.**

- Identify PhoBERT, XLM-R, and ViSoBERT as model/context references.
- Required citations: `nguyen-tuan-nguyen-2020-phobert`, `conneau-etal-2020-unsupervised`, `nguyen-etal-2023-visobert`.
- Explain only their relevance to baseline selection and social-media context.

### 2.2 Vietnamese sentiment and emotion evaluation resources

**Paragraph 2.**

- Introduce UIT-VSFC, UIT-VSMEC, and AIVIVN only as external evaluation benchmarks, not sources for ViPragSent.
- Required citations before drafting: verified BibTeX records for UIT-VSFC, UIT-VSMEC, and the appropriate AIVIVN source.
- Evidence for their role in this study: `README.md`, `configs/datasets.yaml`, `answer/results/p0_multi_seed_ablation.json`, `answer/results/p0_visobert_baseline.json` (**VERIFIED**).

### 2.3 Auxiliary-task and multi-task learning for sentiment/pragmatics

**Paragraph 3.**

- Position auxiliary-task learning as a design family whose effects must be measured, rather than assumed beneficial.
- Required citation before drafting: a verified primary multi-task sentiment/pragmatics paper (candidate: Moore and Barnes, 2021).
- Close by stating the paper’s contribution: a recorded comparative evaluation and trade-off analysis in the defined Vietnamese setting.

**Qualification:** citations resolve in the current build. Both Chat bases have user-attested 2026-07-14 download-time revision pins; the archived manifests nevertheless lack independently recorded remote revisions and a frozen historical environment.

## 3. Task and Data

**Purpose:** Make the evaluated task reproducible and make access/governance boundaries visible before results are interpreted.

### 3.1 Task formulation and labels

**Paragraph 1.**

- Define the six pragmatic labels and distinguish them from polarity and emotion auxiliaries.
- Evidence: `configs/labels.yaml`; `answer/data_provenance/gold_build_report.json` (**VERIFIED**).
- Insert **Table 1: Task and data inventory**.

**Table 1 contents:** label family, label names, record counts/split counts, training/evaluation role. Do not include raw social-media examples unless permission is confirmed.

### 3.2 Data provenance, split, and governance

**Paragraph 2.**

- State the 12,000-record total and 8,000/2,000/2,000 split.
- State that source text is private, non-commercial research material and external public datasets are evaluation-only.
- Evidence: `README.md`; `configs/data_governance.yaml`; `answer/data_provenance/source_registry.json` (**VERIFIED**).

### 3.3 Annotation and agreement

**Paragraph 3.**

- Describe adjudication at the protocol level; report the agreement framework without overstating independence.
- State that the third rater is the fixed adjudicator and agreement should be described as post-adjudication agreement.
- Evidence: `answer/results/annotation_agreement.json` (**VERIFIED**).
- Place full agreement table in **Appendix Table A1**; mention only the appropriate compact summary in the main text.

## 4. Systems and Evaluation

**Purpose:** Specify what was compared and make seed, metric, and traceability qualifications explicit.

### 4.1 Evaluated systems

**Paragraph 1.**

- Group systems by: (i) PhoBERT/XLM-R encoders, (ii) Sailor-7B/Vistral-7B QLoRA SFT, (iii) GPT-4.1-mini zero-/8-shot, (iv) ViPragSent multi-task encoder.
- Evidence: `answer/results/main_pragmatic.json`; `answer/results/p0_visobert_baseline.json`; `configs/experiments_p0_p1_p2.yaml` (**VERIFIED**). `configs/experiments_q1.yaml` is historical only.
- Required citations: PhoBERT and XLM-R; add verified primary citations for Sailor, Vistral, GPT-4.1-mini before drafting.

### 4.2 Training/evaluation protocol

**Paragraph 2.**

- Report three recorded seeds for encoders and 7B systems; single-run qualification for prompted API baselines.
- Report macro pragmatic F1, bootstrap confidence intervals, and paired bootstrap comparisons as defined by the repository.
- Evidence: `configs/training_h100_20gb.yaml`; `configs/evaluation.yaml`; `answer/results/significance.json` (**VERIFIED**).

### 4.3 Reproducibility and traceability

**Paragraph 3.**

- State the evidence chain: predictions → result JSON → claim ledger → run manifests → pinned external archives.
- Evidence: `answer/results/claim_ledger.csv`; `answer/reproducibility/verification_manifest.json`; `answer/reproducibility/artifact_registry.json` (**VERIFIED**).
- Qualifier: this establishes artifact-level traceability; it must not be portrayed as an independent training rerun.

## 5. Results

**Purpose:** Answer the comparative RQ in evidence order; this section reports observed results and avoids mechanistic explanations not tested by the study.

### 5.1 Main pragmatic comparison

**Paragraph 1 — overall comparison.**

- Insert **Table 2: Main pragmatic detection results** from `answer/tables/main_pragmatic.md`.
- State that Vistral-7B SFT has the strongest recorded macro pragmatic F1 (82.83) among evaluated systems.
- State ViPragSent’s recorded macro pragmatic F1 (73.75) exactly as the source table reports.
- Evidence: `answer/results/main_pragmatic.json` (**VERIFIED**).

**Paragraph 2 — phenomenon-level variation.**

- Insert **Figure 1: Per-phenomenon F1 comparison** from `answer/figures/fig2_per_phenomenon.svg`.
- Describe only visible measured differences; do not attribute causes to model architecture unless separately tested.
- Evidence: same as Table 2 (**VERIFIED**).

### 5.2 Multi-task ablation and external diagnostics

**Paragraph 3 — ablation result.**

- Insert **Table 3: P0 three-seed ablation** in Results.
- Report that the recorded full multi-task row has lower pragmatic F1 than the no-multitask PhoBERT reference. Place the distinct external diagnostics for the four recorded ablations and ViSoBERT in **Appendix Table A4**.
- Evidence: `answer/results/p0_multi_seed_ablation.json`; `answer/results/p0_visobert_baseline.json` (**VERIFIED**).
- Mandatory qualifier: recorded three-seed trade-off association, not causal proof or a universal effect.

**Paragraph 4 — significance boundaries.**

- Cite paired-bootstrap results only for their named comparisons.
- Do not generalize those comparisons to all configurations.
- Evidence: `answer/results/significance.json`; `answer/tables/significance.md` in the verified artifact bundle (**VERIFIED**). The current manuscript does not include a significance appendix table.

### 5.3 Result summary for the RQ

**Paragraph 5.**

- Answer the comparative part of the RQ in one short synthesis: the best observed system is a 7B SFT baseline; the proposed multi-task model is analyzed for trade-offs, not claimed as the best system.

## 6. Analysis

**Purpose:** Analyze error and confidence evidence, then isolate exploratory results rather than folding them into the central conclusion.

### 6.1 Confusion analysis

**Paragraph 1.**

- Use **Figure 2: Pragmatic-polarity confusion** only if a future venue requires it; otherwise retain it only in the verified artifact bundle. The current appendix does not include it.
- Name the evaluated system and exact matrix scope in the caption.
- Evidence: `answer/results/error_confusion.json`; `answer/figures/fig5_confusion.svg` (**VERIFIED**).

### 6.2 Calibration

**Paragraph 2.**

- Report calibration only for systems with stored pragmatic-polarity confidence scores.
- Evidence: `answer/results/calibration.json`; `answer/tables/calibration.md` (**VERIFIED**).
- Default placement: report the stored values in the main text; retain bin-level reliability detail in the verified artifact bundle unless a future venue requires an appendix figure.

### 6.3 Source-stratified sensitivity and exploratory low-resource sarcasm

**Paragraph 3.**

- Report P1 as a descriptive 1,666/334 source-stratified sensitivity analysis; do not draw a causal source conclusion.
- Report P2 budgets and mixed monotonicity across the two systems only as exploratory three-seed observations.
- Evidence: `answer/results/p1_source_stratified_sensitivity.json`; `answer/results/p2_multi_seed_low_resource.json`; `answer/tables/p0_p1_p2_experiments.md` (**VERIFIED**).
- Default placement: summarize P1 in Analysis and retain **Appendix Table A3** for P1/P2 detail.

## 7. Conclusion

**Purpose:** Answer the RQ without restating every number or adding an unsupported practical recommendation.

**Paragraph content checklist:**

1. Re-state the empirical scope and comparative design.
2. State the central verified observation: 7B SFT systems are stronger than the proposed multi-task model in the recorded main macro-F1 comparison.
3. State the bounded contribution: a traceable Vietnamese social-media pragmatic evaluation and explicit measured trade-off.
4. Direct future work to safe data access, independent retraining reproduction, rationale-faithfulness evaluation, and qualitative analysis.

## Limitations

**Purpose:** Own the limits without introducing new results.

**Paragraph 1 — data and access.** Private source text, no raw-text public release claim, and potential limits on external reproduction.

**Paragraph 2 — evaluation and model scope.** One dataset/task formulation; no claim of general pragmatics coverage; P1 is observational and its VIVID stratum is smaller.

**Paragraph 3 — analysis scope.** Three-seed ablation and low-resource studies remain non-causal/exploratory; API single-run baselines; no manual faithfulness verification for generated rationales.

## Ethical Considerations

**Purpose:** Explain source restrictions, privacy/sensitive-text handling, annotation/adjudication, access, and foreseeable misuse.

**Evidence:** `configs/data_governance.yaml`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `answer/data_provenance/` (**VERIFIED**).

## Appendix plan

| Appendix item | Source | Status |
| --- | --- | --- |
| A1. Inter-annotator agreement by label | `answer/tables/annotation_agreement.md` | INCLUDED; VERIFIED |
| A2. Label prevalence | `answer/data_provenance/gold_build_report.json` | INCLUDED; VERIFIED |
| A3. P1 source sensitivity and P2 low-resource sarcasm results | `answer/tables/p0_p1_p2_experiments.md` | INCLUDED; P1 descriptive and P2 exploratory |
| A4. Three-seed external diagnostics | `answer/results/p0_multi_seed_ablation.json`; `answer/results/p0_visobert_baseline.json` | INCLUDED; VERIFIED, evaluation-only |
| Deferred artifact detail | `answer/tables/significance.md`; `answer/tables/calibration.md`; `answer/figures/fig6_learning_curves.svg`; `answer/figures/fig7_calibration.svg`; `answer/reproducibility/verification_manifest.json` | VERIFIED and retained in the artifact bundle; not included in the current manuscript appendix |

## Claim–evidence map

| Claim | Section | Evidence | Status | Interpretation rule |
| --- | --- | --- | --- | --- |
| C1: task/data resource | 1, 3 | `gold_build_report.json`, agreement result, label config | VERIFIED | No public raw-data-release implication. |
| C2: comparative evaluation | 1, 4, 5 | main result, manifests, protocol configs | VERIFIED | Preserve seed/run distinctions. |
| C3: Vistral strongest observed system | 1, 5, 7 | `main_pragmatic.json` | VERIFIED | “Strongest evaluated/recorded,” not universal SOTA. |
| C4: multi-task trade-off | 1, 5 | P0 ablation and ordinary-task results | VERIFIED | Three-seed recorded association only. |
| C5: artifact traceability | 1, 4, Appendix | ledger, hashes, registry | VERIFIED | Not an independent rerun. |
| C6: low-resource exploratory results | 6, Appendix | P2 low-resource result | VERIFIED | Three-seed, mixed monotonicity across systems; no data-efficiency superiority claim. |
| C7: source sensitivity | 6, Appendix | P1 source-stratified result | VERIFIED | Descriptive only; no causal source effect. |

## Required citations

| Citation / source | Planned section | BibTeX status |
| --- | --- | --- |
| Nguyen et al. (2023), ViSoBERT | 1, 2 | VERIFIED in `references.bib` |
| Nguyen and Tuan Nguyen (2020), PhoBERT | 2, 4 | VERIFIED in `references.bib` |
| Conneau et al. (2020), XLM-R | 2, 4 | VERIFIED in `references.bib` |
| UIT-VSFC primary paper | 2, 5 | VERIFIED: `nguyen-etal-2018-uit-vsfc` in `references.bib` |
| UIT-VSMEC primary paper | 2, 5 | VERIFIED: `ho-etal-2020-uit-vsmec` in `references.bib` |
| AIVIVN scholarly dataset description | 2, 5 | VERIFIED: `nguyen-etal-2020-efficient` supports the published setting; organizer authorship, canonical split, and license remain unresolved for the local mirror |
| Sailor-7B-Chat source/model card | 4 | PARTIAL: `dou-etal-2024-sailor` verifies the family; the Chat ID and its 2026-07-14 user-attested download-time pin are recorded, but the archived manifest does not independently log the remote revision or frozen environment |
| Vistral-7B-Chat source/model card | 4 | PARTIAL: official model card `nguyen-etal-2023-vistral`; its 2026-07-14 user-attested download-time pin is recorded, but the archived manifest does not independently log the remote revision or frozen environment |
| GPT-4.1-mini model documentation/version source | 4 | VERIFIED: `openai-2025-gpt-4-1-mini` matches the recorded snapshot |
| Multi-task sentiment/pragmatics primary source | 2 | VERIFIED: `moore-barnes-2021-multi` in `references.bib` |

## Items remaining unverified

These items do **not** invalidate the verified experiment inventory.

1. Final title, author list, affiliations, funding, conflict-of-interest statement, and target future ARR/EACL cycle.
2. Organizer-authored AIVIVN provenance/license evidence for any claim beyond the verified scholarly description and local hash-identified mirror; the 7B Chat pins are recorded from documented download-time provenance, while a frozen historical software/hardware environment remains a partial traceability limitation.
3. Permission/access mechanism for any public or reviewer-facing dataset release beyond the current governance terms.
4. Safe qualitative examples, a rationale-faithfulness audit, and an independent full retraining reproduction, if the authors decide to strengthen the paper. These remain future work.
