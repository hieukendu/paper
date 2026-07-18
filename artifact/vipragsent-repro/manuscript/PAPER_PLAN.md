# EACL-style paper plan: ViPragSent

Status: **active manuscript drafting – Sections 1--9, title, and abstract are drafted; post-audit revision and final build verification are in progress**

## 1. Paper configuration record

| Item | Decision / status |
| --- | --- |
| Paper type | Empirical NLP conference paper |
| Target shape | Generic ACL/ARR-style long paper. The eight-page content allocation is an internal planning target only; final page limits and inclusions remain venue-specific and TBD. |
| Audience | NLP researchers in discourse/pragmatics, multilingual NLP, and sentiment/social-media analysis |
| Body and citations | English, LaTeX, official ACL style, BibTeX |
| Primary source of numbers | `../answer/results/` and tables generated from it |
| Primary method sources | `../README.md`, `../configs/`, `../src/`, and retained run manifests |
| Source restriction | `main(5).pdf` is not a framing, writing, or result source |
| Venue status | EACL 2026 is concluded. Treat its call as a style/policy reference and re-check the next target venue’s call before submission. |

## 2. Strongest defensible research question

**RQ:** *On an adjudicated Vietnamese-language test set covering six pragmatic phenomena and including a documented social-media source component, how do standard Vietnamese/multilingual encoders, 7B instruction-tuned models, prompted API baselines, and a multi-task encoder compare; and what trade-off does the recorded ablation study show between pragmatic detection and an external ordinary-task diagnostic?*

This is deliberately an **evaluative and analytical** RQ. It does not presume that the proposed model outperforms its baselines.

### Evidence and inference

- **Evidence:** the study has 12,000 adjudicated records (8,000/2,000/2,000 train/dev/test), seven main systems, three seeds for learned encoders and 7B systems, and generated result/analysis artifacts.
- **Evidence:** Vistral-7B SFT has the highest recorded main macro pragmatic F1 (82.83); the recorded ViPragSent multi-task model has 73.75.
- **Inference:** a benchmark-and-trade-off framing is more credible than a method-superiority framing.
- **Recommendation:** submit as an empirical resource/evaluation paper, with the multi-task model treated as an analyzed baseline rather than the headline state-of-the-art contribution.

## 3. Evidence-bounded contribution claims

| ID | Permitted claim | Evidence | Strength | Boundary |
| --- | --- | --- | --- | --- |
| C1 | The study defines and evaluates six pragmatic phenomena in 12,000 adjudicated Vietnamese-language records: 10,000 retained local social-media export records and 2,000 VIVID-identified replacement candidates, alongside polarity and emotion annotations. | `answer/data_provenance/gold_build_report.json`; `answer/results/annotation_agreement.json`; `README.md` | Strong | Do not call all records social-media or claim the raw text is publicly released. |
| C2 | The study provides a reproducible comparative evaluation across encoders, 7B SFT models, and prompted API baselines. | `answer/results/main_pragmatic.json`; `answer/run_manifests/`; `answer/reproducibility/verification_manifest.json` | Strong | API baselines are single-run prompted baselines, not three-seed training runs. |
| C3 | The recorded main evaluation identifies Vistral-7B SFT as the strongest evaluated system (82.83 macro F1), while ViPragSent records 73.75 macro F1. | `answer/tables/main_pragmatic.md`; `answer/results/main_pragmatic.json` | Strong | Do not claim ViPragSent wins this comparison. |
| C4 | In the single-seed ablation, the full multi-task configuration records lower pragmatic F1 than the no-multitask PhoBERT reference, but higher stored average external ordinary-task F1. | `answer/results/multitask_ablation.json`; `answer/tables/multitask_ablation.md` | Moderate | Say “observed trade-off”; do not assert causality, robustness, or a universal effect. |
| C5 | The comparison is traceable from reported values to prediction files, run manifests, result hashes, and pinned external model archives. | `answer/results/claim_ledger.csv`; `answer/reproducibility/verification_manifest.json`; `answer/reproducibility/artifact_registry.json` | Strong | This is file-level reproducibility, not an independent rerun. |
| C6 | The low-resource sarcasm results are exploratory and non-monotonic across budgets. | `answer/results/low_resource_sarcasm.json`; `answer/tables/low_resource_sarcasm.md` | Moderate | One seed per budget; place in appendix or explicitly exploratory analysis. |

## 4. Claims that must not appear

- “ViPragSent outperforms all baselines,” “state of the art,” or an equivalent superiority claim.
- A public-data-release claim, because raw/processed social-media text is governed as private research material.
- A claim that generated rationales were manually faithfulness-verified.
- A claim that the low-resource experiment establishes a consistent data-efficiency advantage.
- A general transfer or retention claim based on the external ordinary-task diagnostic: the recorded external results do not support it.
- A causal claim about auxiliary tasks based on one-seed ablations.

## 5. Proposed architecture and page budget

Planning target: **approximately 8 pages of main content**, plus references and appendices as permitted by the eventual venue. Word budgets are planning estimates; they are not a submission-validity criterion until a venue is selected.

| Section | Target words | Approx. content pages | Purpose and evidence |
| --- | ---: | ---: | --- |
| Abstract | 170–200 | — | RQ, data/task scope, comparative result, and bounded trade-off finding. Draft only after claims are confirmed. |
| 1. Introduction | 550–650 | 0.9 | Gap: Vietnamese social-media evaluation rarely isolates the recorded pragmatic phenomena. End with RQ and bounded contributions C1–C5. |
| 2. Related Work | 500–600 | 0.8 | Vietnamese social-media resources/models; pragmatic/sentiment evaluation; multi-task auxiliary-task trade-offs. |
| 3. Task and Data | 650–750 | 1.1 | Six phenomena, annotation/adjudication, splits, label scope, governance/access boundary, agreement. |
| 4. Systems and Evaluation | 650–750 | 1.1 | Systems, seeds, training/evaluation protocol, macro-F1, bootstrap CI, significance, traceability. |
| 5. Results | 950–1,100 | 1.8 | Main comparison, per-phenomenon behavior, ablation/external-ordinary-task trade-off. Report rather than explain unverified mechanisms. |
| 6. Analysis | 550–650 | 0.9 | Confusion/calibration; exploratory low-resource results, clearly bounded. |
| 7. Conclusion | 150–200 | 0.3 | Answer the RQ without a superiority conclusion. |
| Main-content reserve | — | 1.1 | Tables, figures, captions, and layout variance. |
| Limitations | 250–350 | outside limit | Required dedicated section before references. |
| Ethical Considerations | 200–300 | outside limit | Data governance, sensitive social text, access, annotation, misuse. |

### Section transitions

1. Introduction establishes the evaluation gap and RQ.
2. Related work positions the task against Vietnamese social-media resources and multi-task evaluation.
3. Task and data operationalize what the evaluation measures.
4. Systems and evaluation make the comparison interpretable and reproducible.
5. Results answer the comparative part of the RQ; analysis explains its limits and secondary patterns.
6. Conclusion returns to the RQ, not to an unsupported method-superiority narrative.

## 6. Planned tables and figures

| Placement | Artifact | Evidence source | Use / restriction |
| --- | --- | --- | --- |
| Main Table 1 | Dataset/task inventory: six pragmatic labels, polarity/emotion auxiliaries, split sizes | `README.md`; `configs/labels.yaml`; `answer/data_provenance/gold_build_report.json` | New compact presentation is allowed only if it reproduces source facts exactly. |
| Main Table 2 | Main pragmatic results by system and macro F1 | `answer/tables/main_pragmatic.md` | Include confidence intervals and seed/run qualification in caption or note. |
| Main Table 3 | Ablation plus external ordinary-task diagnostic | `answer/tables/multitask_ablation.md`; `answer/tables/ordinary_sentiment.md` | Separate the one-seed ablation from the three-seed main evaluation. |
| Main Figure 1 | Per-phenomenon comparison | `answer/figures/fig2_per_phenomenon.svg` | Use to show variation across phenomena, not a claim of general superiority. |
| Main Figure 2 (if space) | Pragmatic-polarity confusion | `answer/figures/fig5_confusion.svg`; `answer/tables/error_confusion.md` | Analysis only; name the evaluated system in caption. |
| Appendix Table A1 | Inter-annotator agreement | `answer/tables/annotation_agreement.md` | State the fixed adjudicator/post-adjudication qualification. |
| Appendix Table A2 | Paired bootstrap results | `answer/tables/significance.md` | Retain comparison direction and intervals; if tail proportions are shown, label them as uncorrected finite-resample diagnostics rather than inferential p values. |
| Appendix Figure A1 | Learning curves | `answer/figures/fig6_learning_curves.svg` | Supplemental diagnosis, not a performance claim. |
| Appendix Figure A2 | Calibration | `answer/figures/fig7_calibration.svg`; `answer/tables/calibration.md` | Compare only systems with stored confidence. |
| Appendix Table A3 | Low-resource sarcasm | `answer/tables/low_resource_sarcasm.md` | Label exploratory, one seed per budget. |

## 7. Missing information and experiments

| Priority | Gap | Why it matters | Recommended action |
| --- | --- | --- |
| High | No public raw-data release or confirmed controlled-access protocol | A resource claim needs an access story. | Obtain source-rights permission or document a reviewer-access protocol and release only permitted metadata/code. |
| High | Ablation is single seed | It cannot establish robust auxiliary-task effects. | Run three seeds for the ablation suite and recompute paired tests before making a strong mechanism claim. |
| High | ViSoBERT is the source-domain model but is not an evaluated baseline | Reviewers may consider it an expected social-media baseline. | Add it if legally/technically feasible; otherwise state this as a limitation. |
| Medium | No qualitative error examples suitable for release | Confusion matrices alone give limited pragmatic insight. | Prepare de-identified, permission-safe error examples or state that privacy prevents sharing examples. |
| Medium | Rationale faithfulness audit is waived | Rationale outputs cannot support explanation-quality claims. | Do not claim faithful explanations; add a human audit only if the paper needs that contribution. |
| Medium | Low-resource comparison is one-seed and non-monotonic | It cannot support data-efficiency claims. | Keep it exploratory or repeat with multiple seeds. |
| Medium | External ordinary-task diagnostic does not demonstrate broad retention | The proposed model is not the best reported system on those datasets. | Present as a trade-off diagnostic, not positive transfer. |
| Submission gate | Next target venue is unknown | EACL 2026’s cycle is complete. | Before drafting for submission, identify the target ARR venue/cycle and re-verify its current call, template, anonymity, checklist, and page rules. |

## 8. Drafting roadmap

1. **Confirm this framing.** Decide whether the paper is an evaluation/resource paper (recommended) rather than a proposed-method paper.
2. **Lock the contribution ledger.** Use C1–C6 as the only claim pool; add no claim until it has an evidence row.
3. **Build related-work matrix.** Start from verified records in `references.bib`; add dataset, pragmatics, and multi-task sources only after primary-source verification.
4. **Create an anonymous ACL project.** Download the current official style files into `manuscript/latex/` only after the submission venue/cycle is chosen. Do not put identifiable repository links in the anonymous version.
5. **Draft in evidence order.** Task/Data → Systems/Evaluation → Results → Analysis → Related Work → Introduction → Abstract → Limitations/Ethics.
6. **Run paper checks.** Every numerical sentence must map to `answer/results/claim_ledger.csv`; every external claim must map to a verified BibTeX key.
7. **Submission preparation.** Add responsible-NLP checklist, data-access statement, ethical considerations, AI-use disclosure as applicable, author contributions, funding, and conflict-of-interest statement.

## 9. Current plan gate

Before any full manuscript drafting, confirm:

1. the evaluation-and-trade-off RQ;
2. that C1–C6 are the permitted contribution claims;
3. whether to pursue the missing three-seed ablations and a ViSoBERT baseline; and
4. the intended future venue/cycle.
