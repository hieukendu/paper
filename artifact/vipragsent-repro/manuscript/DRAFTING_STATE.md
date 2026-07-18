# ViPragSent drafting state

**Workflow:** `academic-paper`  
**Invocation mode:** phase-by-phase / direct-mode  
**Current operation:** Post-audit revision complete  
**Gate result:** PASS for wording/metadata corrections, PDF rebuild, and anonymous front-matter verification. Source authorization remains a pre-submission evidence requirement.

## 0.1 Latest drafting update

- Section 3 was drafted from repository/configuration and `answer/` provenance evidence only.
- `SECTION_03_TRACEABILITY.md` maps every Section 3 claim and Table 1 field to its evidence source; numerical prose sentences carry preceding LaTeX `TRACE` comments.
- The detailed agreement table is included after `\appendix` from `latex/appendix/table_annotation_agreement.tex`; no agreement detail appears in the main Section 3 text.
- The manuscript compiled successfully with the bundled Tectonic executable on 2026-07-18; only non-blocking layout/toolchain warnings remain.
- Section 4 cites the verified Sailor and Vistral family records while retaining the limitation that historical base-model revisions are unavailable. The current QLoRA launcher, configuration, README, results, and archived manifests consistently specify three recorded seeds.
- The malformed local `latex/acl_natbib.bst` was replaced with the official ACL repository version at commit `d5adc823ff0f80f98c80405ca0ab66c68e684409` after structural and SHA-256 checks. BibTeX now resolves the Section 4 citations.
- Section 5 is drafted from `answer/` numerical artifacts only. It distinguishes the three-seed/single-run main comparison, the single-seed ablation, and the separately aggregated external ordinary-task diagnostic.
- Table 2 reproduces every main-pragmatic point estimate at the source precision, while the stored intervals remain available in the verified result artifact; Table 3 preserves the full stored ablation precision and the external ordinary-task diagnostic panel.
- Figure 1 is a format-converted copy of the verified `answer/figures/fig2_per_phenomenon.svg` artifact; its SVG source remains alongside the PDF used by LaTeX.
- Section 5 contains 1,081 prose words and compiled successfully on 2026-07-18. The arithmetic/table check found all 49 displayed Table 2 means, all six Panel A rows, and all nine Panel B cells consistent with their `answer/` sources; the copied Figure 1 SVG has the same SHA-256 as its source artifact.
- Section 6 is drafted at 612 prose words and compiled successfully on 2026-07-18. It labels all non-result commentary as observed evidence, plausible interpretation, or untested hypothesis.
- The analysis uses only the stored ViPragSent pragmatic-polarity confusion artifact, confidence-eligible calibration artifacts, and exploratory one-seed low-resource sarcasm artifacts. The copied confusion-matrix graphic is reserved for appendix placement because the six class labels are not sufficiently legible in the main-text column.
- The AIVIVN citation gate was re-scoped without weakening provenance controls: Nguyen et al. (2020), DOI `10.3233/FAIA200579`, is VERIFIED as a scholarly description of the published AIVIVN-2019 binary setting, while organizer authorship, local-mirror identity, canonical split, and license remain separately qualified.
- Section 2 is drafted at 500 prose words. `SECTION_02_TRACEABILITY.md` maps each paragraph-level claim to its verified citation or repository provenance record; no unverified organizer or license claim is used in the manuscript prose.
- The complete anonymous manuscript compiled successfully with bundled Tectonic after the Section 2 and bibliography update. BibTeX resolved `nguyen-etal-2020-efficient`; the compile log contains no undefined-citation, undefined-reference, or LaTeX-error diagnostic. Remaining warnings are pre-existing underfull-box/layout notices and the local `lineno.sty` UTF-8 warning.
- Section 1 is drafted at 567 prose words. It uses the current evidence-bounded RQ, names only the four approved system groups, and states C1--C5 with adjacent LaTeX TRACE comments.
- `SECTION_01_TRACEABILITY.md` maps the Introduction's motivation, narrow task scope, and every contribution to the verified literature, repository, and result artifacts. The Introduction-to-Results check confirms the displayed 82.83 and 73.75 values are the rounded source values for Vistral-7B SFT and ViPragSent, respectively; no unqualified superiority or causal claim was found.
- The title, abstract, Conclusion, Limitations, and Ethical Considerations are drafted. The title and abstract are anonymous and must be verified in the rebuilt PDF.
- The author-provided, versioned annotation protocol is recorded at `docs/annotation_guidelines_v1.md`. Section 3 uses its six interpretive criteria, makes clear that contextual judgment is not a deterministic cue rule, and preserves the fixed-adjudicator gold-label boundary.
- Provenance metadata records the final gold composition: 10,000 retained local ViSoBERT-export records plus 2,000 author-created, context-augmented derivatives based on VIVID idiom/proverb materials. The VIVID upstream dataset revision is pinned at `471def6618e8dde0f2007f1ac5375e50addb2491`; source authorization and license documentation remain pending project action.
- The 2026-07-18 external-license review did not establish a primary license or research-use authorization for the exact UIT-VSFC, UIT-VSMEC, or AIVIVN artifacts. The project therefore retains the private-data boundary; applicable authorization or reviewed terms must be archived before submission.
- The paired-bootstrap tail proportions are described as stored, uncorrected finite-resample diagnostics. The manuscript reports intervals descriptively and makes no p-value-based inferential conclusion.
- The eight-page allocation is an internal planning target, not a blocker while the submission venue is TBD.
- The current PDF was rebuilt with bundled Tectonic on 2026-07-18. Its first page contains the anonymous title and current abstract, with no unresolved citation/reference or fatal TeX diagnostic. Remaining output is limited to existing underfull-box layout warnings and the local `lineno.sty` UTF-8 warning.
- Private split-integrity metadata now records per-split ID-set, ordered-ID-sequence, and raw-file SHA-256 hashes, with zero duplicate IDs and zero pairwise split overlap.  Section 3 distinguishes the 5,261 initial reviewer disagreements across the eight recorded fields from the zero unresolved disagreements after fixed adjudication.

## 1. Locked research question

> On an adjudicated Vietnamese-language test set covering six pragmatic phenomena and including a documented social-media source component, how do standard Vietnamese/multilingual encoders, 7B instruction-tuned models, prompted API baselines, and a multi-task encoder compare; and what trade-off does the recorded ablation study show between pragmatic detection and an external ordinary-task diagnostic?

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
| C1 | Defines and evaluates six pragmatic phenomena in 12,000 adjudicated Vietnamese-language records: 10,000 retained local social-media export records and 2,000 VIVID-identified replacement candidates, with polarity and emotion annotations. | VERIFIED, QUALIFIED | Do not call all records social-media or claim public raw-text release. |
| C2 | Provides a comparative evaluation across encoders, 7B SFT models, and prompted API baselines. | VERIFIED | API baselines are single-run; learned encoders/7B systems have recorded three-seed runs. |
| C3 | Identifies Vistral-7B SFT as the strongest evaluated system (82.83 macro F1), while ViPragSent records 73.75. | VERIFIED | Strongest evaluated system, not universal SOTA; ViPragSent is not the winner. |
| C4 | Records an ablation trade-off: lower pragmatic F1 than the no-multitask PhoBERT reference and higher values in the external ordinary-task diagnostic. | VERIFIED | One-seed observed trade-off; no causal, robust, or universal claim. |
| C5 | Traces reported comparisons to predictions, manifests, result hashes, and pinned external model archives. | VERIFIED | Artifact-level traceability, not an independent training rerun. |
| C6 | Records exploratory, non-monotonic low-resource sarcasm results across budgets. | VERIFIED | One registered seed per budget; no data-efficiency superiority claim. |

No contribution outside C1–C6 may be drafted unless an evidence-map entry is added and the user explicitly approves it.

## 4. Forbidden claims

- ViPragSent outperforms all baselines, is state of the art, or is the strongest evaluated model.
- Raw/processed source text is publicly released or openly reproducible.
- Generated rationales are manually faithfulness-verified.
- The low-resource study proves consistent data efficiency.
- The external ordinary-task diagnostic establishes broad retention or positive transfer.
- Auxiliary tasks causally produce the observed ablation differences.
- Any claim that treats absent future experiments as completed.

## 5. Locked section order and budgets

| Order | Section | Budget | Status | Evidence / drafting condition |
| ---: | --- | ---: | --- | --- |
| 0 | Abstract | 170–200 words | DRAFTED_COMPILED | 176 words; uses only locked RQ and C1–C5. |
| 1 | Introduction | 550–650 words | DRAFTED_COMPILED | 567 words; current evidence-bounded RQ and C1--C5 only; verified ViSoBERT context citation. |
| 2 | Related Work | 500–600 words | DRAFTED_COMPILED | 500 words; uses verified scholarly AIVIVN citation `nguyen-etal-2020-efficient`; canonical organizer provenance remains a qualified limitation. |
| 3 | Task and Data | 650–750 words | DRAFTED_COMPILED | 750 words; uses answer provenance for quantitative facts and preserves governance boundary. |
| 4 | Systems and Evaluation | 650–750 words | DRAFTED_COMPILED | 659 words; protocol is VERIFIED; only verified citation keys are used. |
| 5 | Results | 950–1,100 words | DRAFTED_COMPILED | 1,081 words; uses `answer/` only and reports paired intervals descriptively. |
| 6 | Analysis | 550–650 words | DRAFTED_COMPILED | 612 words; confusion/calibration are qualified; low-resource analysis remains exploratory. |
| 7 | Conclusion | 150–200 words | DRAFTED_COMPILED | 159 words; answers the bounded RQ without superiority framing. |
| 8 | Limitations | 250–350 words | DRAFTED_COMPILED | 326 words; preserves data, seed, provenance, and access limitations. |
| 9 | Ethical Considerations | 200–300 words | DRAFTED_COMPILED | 289 words; covers governance, privacy, annotation, and misuse. |
| 10 | References | no fixed page budget | DRAFTED_COMPILED | BibTeX style is restored and the verified scholarly AIVIVN record resolves; canonical organizer attribution remains unresolved but is not cited as verified. |
| 11 | Appendices | outside main content limit | PARTIALLY_DRAFTED_COMPILED | A1 agreement is included; A2--A6 remain planned, not drafted. |

The eight-page allocation is an internal layout target with a reserve for tables, figures, captions, and template variance; it is not a blocking submission requirement until a venue is selected.

## 6. Planned main artifacts

| Item | Placement | Authoritative source | Status |
| --- | --- | --- | --- |
| Table 1: task and data inventory | Section 3 | `answer/data_provenance/gold_build_report.json`, `configs/labels.yaml`, `configs/data_governance.yaml` | DRAFTED; source facts VERIFIED |
| Table 2: main pragmatic detection | Section 5 | `answer/tables/main_pragmatic.md` | DRAFTED; source values VERIFIED |
| Table 3: ablation and external ordinary-task diagnostic | Section 5 | `answer/tables/multitask_ablation.md`, `answer/tables/ordinary_sentiment.md` | DRAFTED; preserves single-seed qualifier |
| Figure 1: per-phenomenon comparison | Section 5 | `answer/figures/fig2_per_phenomenon.svg` | DRAFTED; source graphic VERIFIED |
| Figure 2: pragmatic-polarity confusion (appendix placement) | Appendix | `answer/figures/fig5_confusion.svg` | PREPARED; source graphic VERIFIED; main-text placement deferred for label legibility |

## 7. Unresolved citations

| ID | Needed for | Status / action |
| --- | --- | --- |
| CIT-01 | UIT-VSFC | VERIFIED: `nguyen-etal-2018-uit-vsfc` is available. |
| CIT-02 | UIT-VSMEC | VERIFIED: `ho-etal-2020-uit-vsmec` is available. |
| CIT-03 | AIVIVN scholarly description | VERIFIED: `nguyen-etal-2020-efficient` is a publisher-hosted proceedings record. Canonical organizer attribution and the local mirror's license remain non-blocking provenance limitations. |
| CIT-04 | Sailor-7B | PARTIAL: primary family source is verified, but the historical base-model revision is absent. |
| CIT-05 | Vistral-7B | PARTIAL: official model card is verified, but the historical base-model revision is absent. |
| CIT-06 | GPT-4.1-mini | VERIFIED: `openai-2025-gpt-4-1-mini` matches the recorded snapshot. |
| CIT-07 | Auxiliary-task literature | VERIFIED: `moore-barnes-2021-multi` is available. |

Existing verified BibTeX keys include `nguyen-etal-2023-visobert`, `nguyen-tuan-nguyen-2020-phobert`, `conneau-etal-2020-unsupervised`, `nguyen-etal-2018-uit-vsfc`, `ho-etal-2020-uit-vsmec`, `nguyen-etal-2020-efficient`, `openai-2025-gpt-4-1-mini`, and `moore-barnes-2021-multi`.

## 8. Unresolved experiments and scope boundaries

The following are **not** required to represent the currently VERIFIED inventory and must remain limitations/future work unless new verified artifacts are added:

- Multi-seed ablation extensions.
- ViSoBERT baseline.
- Manual rationale-faithfulness evaluation.
- Qualitative examples.
- New data-access arrangement or public raw-text release.

## 9. Anonymous-submission metadata

| Field | State |
| --- | --- |
| Target venue/cycle | TBD; do not label as EACL 2026. |
| Title | `ViPragSent: A Comparative Evaluation of Vietnamese Social-Media Pragmatics and Multi-Task Trade-offs` (anonymous; wording does not imply a model win). |
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
