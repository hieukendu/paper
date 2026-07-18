# Detailed EACL Paper Outline — ViPragSent

**Mode:** `ars-outline` (outline only; no manuscript prose)  
**Evidence policy:** all experiment results and result-derived figures/tables listed below are **VERIFIED** from the double-checked `answer/` bundle. “Unverified” labels apply only to citation completion, submission metadata, or future extensions.

## Paper identity

| Field | Current value |
| --- | --- |
| Working title | **ViPragSent: A Comparative Evaluation of Vietnamese Social-Media Pragmatics and Multi-Task Trade-offs.** Anonymous; it does not imply a performance win. |
| Research question | On an adjudicated Vietnamese-language test set covering six pragmatic phenomena and including a documented social-media source component, how do standard Vietnamese/multilingual encoders, 7B instruction-tuned models, prompted API baselines, and a multi-task encoder compare; and what trade-off does the recorded ablation show between pragmatic detection and an external ordinary-task diagnostic? |
| Paper type | Long empirical NLP conference paper |
| Core framing | Evaluation/resource and trade-off analysis, not proposed-model superiority |
| Numerical authority | `../answer/` |
| Method authority | Repository code, `../README.md`, `../configs/`, and retained run manifests |

## Page and layout budget

| Component | Target words | Main-page allocation | Content role |
| --- | ---: | ---: | --- |
| Abstract | 170–200 | — | Drafted; one compact factual summary. |
| 1. Introduction | 550–650 | 0.9 | Problem, gap, RQ, verified contributions. |
| 2. Related Work | 500–600 | 0.8 | Three connected literatures. |
| 3. Task and Data | 650–750 | 1.1 | Dataset, labels, annotation, governance. |
| 4. Systems and Evaluation | 650–750 | 1.1 | Systems, protocol, metrics, reproducibility. |
| 5. Results | 950–1,100 | 1.8 | Comparative results and ablation trade-off. |
| 6. Analysis | 550–650 | 0.9 | Error, calibration, low-resource scope. |
| 7. Conclusion | 150–200 | 0.3 | Direct RQ answer and bounded takeaway. |
| Figures, tables, captions, layout reserve | — | 1.1 | Prevent overfilling the 8-page limit. |
| **Main content total** | **~4,000–4,650** | **~8 planning pages** | Non-binding until a target venue and its live template are selected. |
| Limitations | 250–350 | outside limit | Required dedicated section before references. |
| Ethical Considerations | 200–300 | outside limit | Data governance, privacy, annotation, misuse. |
| References and appendices | — | outside limit | Standard ACL placement. |

## Detailed section outline

## Abstract (one paragraph; write last)

**Purpose:** State the task, data scope, comparison groups, one central measured result, and the limited trade-off finding.

**Paragraph content checklist:**

1. Task and setting: adjudicated Vietnamese-language records with a documented social-media source component; six pragmatic phenomena.
2. Evaluation design: encoders, 7B SFT models, prompted API baselines, and multi-task encoder.
3. Verified main result: Vistral-7B SFT is the strongest recorded system; do not imply ViPragSent is best.
4. Verified analytical result: recorded multi-task trade-off, qualified as a single-seed ablation observation.
5. Traceability/access boundary: result artifacts and model archive metadata are available; raw text is governed and not represented as publicly released.

**Evidence:** C1–C5 in `PAPER_PLAN.md`.  
**Do not include:** causal claims, SOTA wording, or low-resource conclusions.

## 1. Introduction (~550–650 words; ~0.9 page)

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
- Evidence: `answer/results/main_pragmatic.json`; `configs/experiments_q1.yaml` (**VERIFIED**).

### 1.3 Contributions

**Paragraph 4 — numbered, bounded contributions.**

1. Adjudicated evaluation resource and task definition (C1).
2. Multi-family baseline comparison with recorded seeds/runs (C2).
3. Transparent comparative result, including the fact that Vistral-7B SFT is strongest among evaluated systems (C3).
4. An ablation-based trade-off analysis, not a performance-improvement claim (C4).
5. Result-to-artifact traceability (C5).

**Required evidence:** `answer/results/main_pragmatic.json`, `answer/results/multitask_ablation.json`, `answer/reproducibility/verification_manifest.json` (**VERIFIED**).

## 2. Related Work (~500–600 words; ~0.8 page)

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
- Evidence for their role in this study: `README.md`, `configs/datasets.yaml`, `answer/tables/ordinary_sentiment.md` (**VERIFIED**).

### 2.3 Auxiliary-task and multi-task learning for sentiment/pragmatics

**Paragraph 3.**

- Position auxiliary-task learning as a design family whose effects must be measured, rather than assumed beneficial.
- Required citation before drafting: a verified primary multi-task sentiment/pragmatics paper (candidate: Moore and Barnes, 2021).
- Close by stating the paper’s contribution: a recorded comparative evaluation and trade-off analysis in the defined Vietnamese setting.

**Unverified requirement:** bibliography completion for Section 2.2, Section 2.3, and the base-model papers for Sailor-7B, Vistral-7B, and GPT-4.1-mini.

## 3. Task and Data (~650–750 words; ~1.1 pages)

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

## 4. Systems and Evaluation (~650–750 words; ~1.1 pages)

**Purpose:** Specify what was compared and make seed, metric, and traceability qualifications explicit.

### 4.1 Evaluated systems

**Paragraph 1.**

- Group systems by: (i) PhoBERT/XLM-R encoders, (ii) Sailor-7B/Vistral-7B QLoRA SFT, (iii) GPT-4.1-mini zero-/8-shot, (iv) ViPragSent multi-task encoder.
- Evidence: `answer/results/main_pragmatic.json`; `configs/experiments_q1.yaml` (**VERIFIED**).
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

## 5. Results (~950–1,100 words; ~1.8 pages)

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

### 5.2 Multi-task ablation and external ordinary-task diagnostic

**Paragraph 3 — ablation result.**

- Insert **Table 3: Ablation and external ordinary-task diagnostic**.
- Report that the recorded full multi-task row has lower pragmatic F1 than the no-multitask PhoBERT reference, while the table reports higher values in the external ordinary-task diagnostic for the full row.
- Evidence: `answer/results/multitask_ablation.json`; `answer/results/ordinary_sentiment.json` (**VERIFIED**).
- Mandatory qualifier: recorded single-seed ablation; observed trade-off, not causal proof.

**Paragraph 4 — significance boundaries.**

- Cite paired-bootstrap results only for their named comparisons.
- Do not generalize those comparisons to all configurations.
- Evidence: `answer/results/significance.json`; **Appendix Table A2** (**VERIFIED**).

### 5.3 Result summary for the RQ

**Paragraph 5.**

- Answer the comparative part of the RQ in one short synthesis: the best observed system is a 7B SFT baseline; the proposed multi-task model is analyzed for trade-offs, not claimed as the best system.

## 6. Analysis (~550–650 words; ~0.9 page)

**Purpose:** Analyze error and confidence evidence, then isolate exploratory results rather than folding them into the central conclusion.

### 6.1 Confusion analysis

**Paragraph 1.**

- Use **Figure 2: Pragmatic-polarity confusion** only if layout allows; otherwise move to Appendix.
- Name the evaluated system and exact matrix scope in the caption.
- Evidence: `answer/results/error_confusion.json`; `answer/figures/fig5_confusion.svg` (**VERIFIED**).

### 6.2 Calibration

**Paragraph 2.**

- Report calibration only for systems with stored pragmatic-polarity confidence scores.
- Evidence: `answer/results/calibration.json`; `answer/tables/calibration.md` (**VERIFIED**).
- Default placement: **Appendix Figure A2** and short main-text pointer.

### 6.3 Exploratory low-resource sarcasm

**Paragraph 3.**

- Report budgets and non-monotonic outcomes only as exploratory observations.
- Evidence: `answer/results/low_resource_sarcasm.json`; `answer/tables/low_resource_sarcasm.md` (**VERIFIED**).
- Default placement: **Appendix Table A3 / Figure A1**, with one-sentence main-text mention only if space permits.
- Mandatory qualifier: one registered seed per budget.

## 7. Conclusion (~150–200 words; ~0.3 page)

**Purpose:** Answer the RQ without restating every number or adding an unsupported practical recommendation.

**Paragraph content checklist:**

1. Re-state the empirical scope and comparative design.
2. State the central verified observation: 7B SFT systems are stronger than the proposed multi-task model in the recorded main macro-F1 comparison.
3. State the bounded contribution: a traceable Vietnamese social-media pragmatic evaluation and explicit measured trade-off.
4. Direct future work to multi-seed ablations, safe data access, ViSoBERT baseline evaluation, and qualitative analysis.

## Limitations (required; 250–350 words; outside content limit)

**Purpose:** Own the limits without introducing new results.

**Paragraph 1 — data and access.** Private source text, no raw-text public release claim, and potential limits on external reproduction.

**Paragraph 2 — evaluation and model scope.** One dataset/task formulation; no claim of general pragmatics coverage; no evaluated ViSoBERT baseline in the current inventory.

**Paragraph 3 — analysis scope.** Single-seed ablation and low-resource studies; API single-run baselines; no manual faithfulness verification for generated rationales.

## Ethical Considerations (200–300 words; outside content limit)

**Purpose:** Explain source restrictions, privacy/sensitive-text handling, annotation/adjudication, access, and foreseeable misuse.

**Evidence:** `configs/data_governance.yaml`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `answer/data_provenance/` (**VERIFIED**).

## Appendix plan

| Appendix item | Source | Status |
| --- | --- | --- |
| A1. Inter-annotator agreement by label | `answer/tables/annotation_agreement.md` | VERIFIED |
| A2. Paired bootstrap comparisons | `answer/tables/significance.md` | VERIFIED |
| A3. Low-resource sarcasm results | `answer/tables/low_resource_sarcasm.md` | VERIFIED, exploratory |
| A4. Calibration | `answer/tables/calibration.md`; `answer/figures/fig7_calibration.svg` | VERIFIED, eligible systems only |
| A5. Learning curves and training details | `answer/tables/learning_curves.md`; `answer/figures/fig6_learning_curves.svg`; `answer/run_manifests/` | VERIFIED |
| A6. Reproducibility manifest | `answer/reproducibility/verification_manifest.json` | VERIFIED |

## Claim–evidence map

| Claim | Section | Evidence | Status | Interpretation rule |
| --- | --- | --- | --- | --- |
| C1: task/data resource | 1, 3 | `gold_build_report.json`, agreement result, label config | VERIFIED | No public raw-data-release implication. |
| C2: comparative evaluation | 1, 4, 5 | main result, manifests, protocol configs | VERIFIED | Preserve seed/run distinctions. |
| C3: Vistral strongest observed system | 1, 5, 7 | `main_pragmatic.json` | VERIFIED | “Strongest evaluated/recorded,” not universal SOTA. |
| C4: multi-task trade-off | 1, 5 | ablation and ordinary-task results | VERIFIED | Single-seed observed trade-off only. |
| C5: artifact traceability | 1, 4, Appendix | ledger, hashes, registry | VERIFIED | Not an independent rerun. |
| C6: low-resource exploratory results | 6, Appendix | low-resource result | VERIFIED | No data-efficiency superiority claim. |

## Required citations

| Citation / source | Planned section | BibTeX status |
| --- | --- | --- |
| Nguyen et al. (2023), ViSoBERT | 1, 2 | VERIFIED in `references.bib` |
| Nguyen and Tuan Nguyen (2020), PhoBERT | 2, 4 | VERIFIED in `references.bib` |
| Conneau et al. (2020), XLM-R | 2, 4 | VERIFIED in `references.bib` |
| UIT-VSFC primary paper | 2, 5 | VERIFIED: `nguyen-etal-2018-uit-vsfc` in `references.bib` |
| UIT-VSMEC primary paper | 2, 5 | VERIFIED: `ho-etal-2020-uit-vsmec` in `references.bib` |
| AIVIVN scholarly dataset description | 2, 5 | VERIFIED: `nguyen-etal-2020-efficient` supports the published setting; organizer authorship, canonical split, and license remain unresolved for the local mirror |
| Sailor-7B source/model card | 4 | PARTIAL: `dou-etal-2024-sailor` verifies the family and configured ID; the historical base-model revision is absent |
| Vistral-7B source/model card | 4 | PARTIAL: official model card `nguyen-etal-2023-vistral`; the historical base-model revision is absent |
| GPT-4.1-mini model documentation/version source | 4 | VERIFIED: `openai-2025-gpt-4-1-mini` matches the recorded snapshot |
| Multi-task sentiment/pragmatics primary source | 2 | VERIFIED: `moore-barnes-2021-multi` in `references.bib` |

## Items remaining unverified

These items do **not** invalidate the verified experiment inventory.

1. Final title, author list, affiliations, funding, conflict-of-interest statement, and target future ARR/EACL cycle.
2. Organizer-authored AIVIVN provenance/license evidence for any claim beyond the verified scholarly description and local hash-identified mirror; historical base-model revisions for Sailor-7B and Vistral-7B remain partial traceability limitations.
3. Permission/access mechanism for any public or reviewer-facing dataset release beyond the current governance terms.
4. A multi-seed ablation extension, a ViSoBERT baseline, and safe qualitative examples, if the authors decide to strengthen the paper. These are optional future experiments, not missing numerical evidence for the recorded experiments.
