# ViPragSent drafting state

**Workflow:** `academic-paper`  
**Invocation mode:** phase-by-phase / direct-mode  
**Current operation:** New-experiment integration revision and independent audits complete. No page constraint is active until a target venue is selected; VIVID authorization/license review remains unresolved.
**Gate result:** PASS for wording/metadata corrections, PDF rebuild, and anonymous front-matter verification. Source authorization remains a pre-submission evidence requirement.

## 0.1 Latest drafting update

- On 2026-07-19, P0, ViSoBERT, P1, P2, and the new external predictions passed artifact-level validation: 90/90 new prediction files had exact gold-ID coverage; 93 core and 45 external per-seed reports recomputed; and 46 stored aggregates matched. This is not an independent training rerun.
- The traceability bundle now records the six P0/P1/P2 result objects, 151 claim-ledger rows (including all six ViSoBERT label aggregates), ViSoBERT revision `196a62afad9cbe4f52a54aabad828b13f0eec59a`, P1's 1,666/334 source composition, and P2's three-seed 1:3 budget design.

- Section 3 was drafted from repository/configuration and `answer/` provenance evidence only.
- `SECTION_03_TRACEABILITY.md` maps every Section 3 claim and Table 1 field to its evidence source; numerical prose sentences carry preceding LaTeX `TRACE` comments.
- The detailed agreement table is included after `\appendix` from `latex/appendix/table_annotation_agreement.tex`; no agreement detail appears in the main Section 3 text.
- The manuscript compiled successfully with the bundled Tectonic executable on 2026-07-18; only non-blocking layout/toolchain warnings remain.
- Section 4 cites the verified Sailor and Vistral family records and records user-attested 2026-07-14 download-time pins for both Chat bases. The archived manifests retain only local paths, so the immutable pins improve retrieval traceability but do not substitute for a frozen historical software/hardware environment. The current QLoRA launcher, configuration, README, results, and archived manifests consistently specify three recorded seeds.
- The malformed local `latex/acl_natbib.bst` was replaced with the official ACL repository version at commit `d5adc823ff0f80f98c80405ca0ab66c68e684409` after structural and SHA-256 checks. BibTeX now resolves the Section 4 citations.
- Section 5 is drafted from `answer/` numerical artifacts only. It distinguishes the three-seed/single-run main comparison, the three-seed P0 ablation, and the separately aggregated external ordinary-task diagnostic.
- Table 2 reproduces every main-pragmatic point estimate at the source precision, while the stored intervals remain available in the verified result artifact; Table 3 preserves the full stored ablation precision and the external ordinary-task diagnostic panel.
- Figure 1 is a format-converted copy of the verified `answer/figures/fig2_per_phenomenon.svg` artifact; its SVG source remains alongside the PDF used by LaTeX.
- Section 5 uses `answer/` numerical artifacts only; the arithmetic/table check found all displayed main and P0 values consistent with their sources, and the new external diagnostics are reported in Appendix Table 7.
- Section 6 labels all non-result commentary as observed evidence, plausible interpretation, or untested hypothesis.
- The analysis uses only the stored ViPragSent pragmatic-polarity confusion artifact, confidence-eligible calibration artifacts, the descriptive P1 source-stratified results, and exploratory three-seed P2 low-resource sarcasm artifacts. The confusion-matrix graphic remains in the verified artifact bundle rather than the current appendix because its six class labels are not sufficiently legible in the main-text column.
- The AIVIVN citation gate was re-scoped without weakening provenance controls: Nguyen et al. (2020), DOI `10.3233/FAIA200579`, is VERIFIED as a scholarly description of the published AIVIVN-2019 binary setting, while organizer authorship, local-mirror identity, canonical split, and license remain separately qualified.
- `SECTION_02_TRACEABILITY.md` maps each paragraph-level claim to its verified citation or repository provenance record; no unverified organizer or license claim is used in the manuscript prose.
- The complete anonymous manuscript compiled successfully with bundled Tectonic after the Section 2 and bibliography update. BibTeX resolved `nguyen-etal-2020-efficient`; the compile log contains no undefined-citation, undefined-reference, or LaTeX-error diagnostic. Remaining warnings are pre-existing underfull-box/layout notices and the local `lineno.sty` UTF-8 warning.
- Section 1 uses the current evidence-bounded RQ, includes the evaluated ViSoBERT baseline, and states C1--C7 with adjacent LaTeX TRACE comments.
- `SECTION_01_TRACEABILITY.md` maps the Introduction's motivation, narrow task scope, and every contribution to the verified literature, repository, and result artifacts. The Introduction-to-Results check confirms the displayed 82.83 and 73.75 values are the rounded source values for Vistral-7B SFT and ViPragSent, respectively; no unqualified superiority or causal claim was found.
- The title, abstract, Conclusion, Limitations, and Ethical Considerations are drafted. The title and abstract are anonymous and must be verified in the rebuilt PDF.
- The author-provided, versioned annotation protocol is recorded at `docs/annotation_guidelines_v1.md`. Section 3 uses its six interpretive criteria, makes clear that contextual judgment is not a deterministic cue rule, and preserves the fixed-adjudicator gold-label boundary.
- Provenance metadata records the final gold composition: 10,000 retained local ViSoBERT-export records plus 2,000 author-created, context-augmented derivatives based on VIVID idiom/proverb materials. The VIVID upstream dataset revision is pinned at `471def6618e8dde0f2007f1ac5375e50addb2491`; source authorization and license documentation remain pending project action.
- The 2026-07-18 external-license review did not establish a primary license or research-use authorization for the exact UIT-VSFC, UIT-VSMEC, or AIVIVN artifacts. The project therefore retains the private-data boundary; applicable authorization or reviewed terms must be archived before submission.
- The paired-bootstrap tail proportions are described as stored, uncorrected finite-resample diagnostics. The manuscript reports intervals descriptively and makes no p-value-based inferential conclusion.
- The current PDF was rebuilt with bundled Tectonic on 2026-07-18. Its first page contains the anonymous title and current abstract, with no unresolved citation/reference or fatal TeX diagnostic. Remaining output is limited to existing underfull-box layout warnings and the local `lineno.sty` UTF-8 warning.
- Private split-integrity metadata now records per-split ID-set, ordered-ID-sequence, and raw-file SHA-256 hashes, with zero duplicate IDs and zero pairwise split overlap.  Section 3 distinguishes the 5,261 initial reviewer disagreements across the eight recorded fields from the zero unresolved disagreements after fixed adjudication.

## 1. Locked research question

> On an adjudicated Vietnamese-language test set covering six pragmatic phenomena and including a documented social-media source component, how do Vietnamese, multilingual, and Vietnamese-social-media encoders, 7B instruction-tuned models, prompted API baselines, and a multi-task encoder compare; and what does the three-seed ablation record about the corresponding configurations?

This wording is locked. Do not semantically expand it without explicit user approval.

## 2. Locked paper framing

- Empirical Vietnamese NLP evaluation/resource paper with trade-off analysis.
- Not a proposed-model-superiority paper.
- Anonymous generic ACL/ARR-style long-paper draft; future venue and cycle are TBD.
- `main(5).pdf` is excluded from every drafting decision.
- `answer/` is the sole authority for numerical results.
- Repository code/configuration/README/run manifests are the authority for method and protocol.
- Verified primary sources and `references.bib` are the authority for external claims.

## 3. Permitted contribution pool

| ID | Locked contribution | Evidence status | Required qualifier |
| --- | --- | --- | --- |
| C1 | Defines and evaluates six pragmatic phenomena in 12,000 adjudicated Vietnamese-language records: 10,000 retained local social-media export records and 2,000 author-created, context-augmented derivatives based on VIVID idiom/proverb materials, with polarity and emotion annotations. | VERIFIED, QUALIFIED | Do not call all records social-media or claim public raw-text release. |
| C2 | Provides a comparative evaluation across encoders (including ViSoBERT), 7B SFT models, and prompted API baselines. | VERIFIED | API baselines are single-run; learned encoders/7B systems and ViSoBERT have recorded three-seed runs. |
| C3 | Identifies Vistral-7B SFT as the strongest evaluated system (82.83 macro F1), while ViPragSent records 73.75. | VERIFIED | Strongest evaluated system, not universal SOTA; ViPragSent is not the winner. |
| C4 | Records a three-seed ablation association: lower pragmatic F1 than the no-multitask PhoBERT reference, with separately scoped external diagnostics. | VERIFIED | No causal, robust, or universal claim. |
| C5 | Traces reported comparisons to predictions, manifests, result hashes, and pinned external model archives. | VERIFIED | Artifact-level traceability, not an independent training rerun. |
| C6 | Records exploratory, three-seed low-resource sarcasm results with mixed monotonicity across the two systems. | VERIFIED | No data-efficiency superiority claim. |
| C7 | Records descriptive source-stratified sensitivity results for the evaluated 1,666/334 source-labelled test strata. | VERIFIED, QUALIFIED | No causal source effect, source superiority, or general transfer claim. |

No contribution outside C1–C7 may be drafted unless an evidence-map entry is added and the user explicitly approves it.

## 4. Forbidden claims

- ViPragSent outperforms all baselines, is state of the art, or is the strongest evaluated model.
- Raw/processed source text is publicly released or openly reproducible.
- Generated rationales are manually faithfulness-verified.
- The low-resource study proves consistent data efficiency.
- The external ordinary-task diagnostic establishes broad retention or positive transfer.
- Auxiliary tasks causally produce the observed ablation differences.
- Any claim that treats absent future experiments as completed.

## 5. Locked section order and budgets

| Order | Section | Status | Evidence / drafting condition |
| ---: | --- | --- | --- |
| 0 | Abstract | DRAFTED_COMPILED | Uses the locked RQ and evidence-bounded claims. |
| 1 | Introduction | DRAFTED_COMPILED | Current RQ and C1--C7; verified ViSoBERT context citation. |
| 2 | Related Work | DRAFTED_COMPILED | Verified scholarly AIVIVN citation; canonical organizer provenance remains qualified. |
| 3 | Task and Data | DRAFTED_COMPILED | Uses answer provenance and preserves the governance boundary. |
| 4 | Systems and Evaluation | DRAFTED_COMPILED | Protocol is VERIFIED; only verified citation keys are used. |
| 5 | Results | DRAFTED_COMPILED | Uses `answer/` only and reports qualified diagnostics. |
| 6 | Analysis | DRAFTED_COMPILED | Confusion/calibration are qualified; P1 descriptive and P2 exploratory. |
| 7 | Conclusion | DRAFTED_COMPILED | Answers the bounded RQ without superiority framing. |
| 8 | Limitations | DRAFTED_COMPILED | Preserves data, seed, provenance, and access limitations. |
| 9 | Ethical Considerations | DRAFTED_COMPILED | Covers governance, privacy, annotation, and misuse. |
| 10 | References | DRAFTED_COMPILED | Verified scholarly AIVIVN record resolves; organizer attribution remains qualified. |
| 11 | Appendices | DRAFTED_COMPILED | Includes agreement, label prevalence, P1/P2, and external-diagnostic tables. |


## 6. Planned main artifacts

| Item | Placement | Authoritative source | Status |
| --- | --- | --- | --- |
| Table 1: task and data inventory | Section 3 | `answer/data_provenance/gold_build_report.json`, `configs/labels.yaml`, `configs/data_governance.yaml` | DRAFTED; source facts VERIFIED |
| Table 2: main pragmatic detection | Section 5 | `answer/tables/main_pragmatic.md` | DRAFTED; source values VERIFIED |
| Table 3: P0 three-seed ablation | Section 5 | `answer/tables/p0_p1_p2_experiments.md` | DRAFTED; preserves seed and configuration qualifiers |
| Figure 1: per-phenomenon comparison | Section 5 | `answer/figures/fig2_per_phenomenon.svg` | DRAFTED; source graphic VERIFIED |
| Table A1: inter-annotator agreement | Appendix | `answer/tables/annotation_agreement.md` | DRAFTED; source values VERIFIED |
| Table A2: label prevalence | Appendix | `answer/data_provenance/gold_build_report.json` | DRAFTED; source values VERIFIED |
| Pragmatic-polarity confusion graphic | Retained artifact bundle | `answer/figures/fig5_confusion.svg` | VERIFIED; not included in the current appendix because label legibility requires a larger display |
| Table A3: P1 source sensitivity and P2 low-resource results | Appendix | `answer/tables/p0_p1_p2_experiments.md` | DRAFTED; P1 descriptive, P2 exploratory three-seed |
| Table A4: three-seed external diagnostics | Appendix | `answer/results/p0_multi_seed_ablation.json`, `answer/results/p0_visobert_baseline.json` | DRAFTED; evaluation-only diagnostics |

## 7. Unresolved citations

| ID | Needed for | Status / action |
| --- | --- | --- |
| CIT-01 | UIT-VSFC | VERIFIED: `nguyen-etal-2018-uit-vsfc` is available. |
| CIT-02 | UIT-VSMEC | VERIFIED: `ho-etal-2020-uit-vsmec` is available. |
| CIT-03 | AIVIVN scholarly description | VERIFIED: `nguyen-etal-2020-efficient` is a publisher-hosted proceedings record. Canonical organizer attribution and the local mirror's license remain non-blocking provenance limitations. |
| CIT-04 | Sailor-7B-Chat | PARTIAL: primary family source and a user-attested 2026-07-14 download-time revision pin are recorded; the archived run manifest does not independently log that remote revision or a frozen environment. |
| CIT-05 | Vistral-7B-Chat | PARTIAL: official model card and a user-attested 2026-07-14 download-time revision pin are recorded; the archived run manifest does not independently log that remote revision or a frozen environment. |
| CIT-06 | GPT-4.1-mini | VERIFIED: `openai-2025-gpt-4-1-mini` matches the recorded snapshot. |
| CIT-07 | Auxiliary-task literature | VERIFIED: `moore-barnes-2021-multi` is available. |

Existing verified BibTeX keys include `nguyen-etal-2023-visobert`, `nguyen-tuan-nguyen-2020-phobert`, `conneau-etal-2020-unsupervised`, `nguyen-etal-2018-uit-vsfc`, `ho-etal-2020-uit-vsmec`, `nguyen-etal-2020-efficient`, `openai-2025-gpt-4-1-mini`, and `moore-barnes-2021-multi`.

## 8. Unresolved experiments and scope boundaries

The following are **not** required to represent the currently VERIFIED inventory and must remain limitations/future work unless new verified artifacts are added:

- Manual rationale-faithfulness evaluation.
- Qualitative examples.
- New data-access arrangement or public raw-text release.

## 9. Anonymous-submission metadata

| Field | State |
| --- | --- |
| Target venue/cycle | TBD; do not label as EACL 2026. |
| Title | `ViPragSent: Comparative Evaluation of Vietnamese Pragmatic Sentiment with a Social-Media Source Component` (anonymous; wording does not imply a model win). |
| Authors and affiliations | Intentionally absent. |
| Acknowledgements and funding | Intentionally absent. |
| Repository/data URLs | No identifying URLs in anonymous draft or supplement. |
| ACL template files | Present in `manuscript/latex/`; `acl_natbib.bst` was restored from the official ACL repository on 2026-07-18. |
| Submission checklist / AI-use disclosure | TBD after the target venue/cycle is selected. |

## 10. Immutable planning inputs for this drafting stage

Do not edit during drafting without an explicit user instruction:

- `PAPER_PLAN.md`
- `DETAILED_OUTLINE.md`
- `EVIDENCE_MAP.md`
- `references.bib` (unless performing an explicitly requested citation-verification update)
- `latex/README.md`
