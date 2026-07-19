# EACL-style paper plan: ViPragSent

Status: **active manuscript drafting – Sections 1--9, title, and abstract are drafted; post-audit revision and final build verification are in progress**

## 1. Paper configuration record

| Item | Decision / status |
| --- | --- |
| Paper type | Empirical NLP conference paper |
| Target shape | Generic ACL/ARR-style long paper. No page limit is active until a target venue and its current call are selected. |
| Audience | NLP researchers in discourse/pragmatics, multilingual NLP, and sentiment/social-media analysis |
| Body and citations | English, LaTeX, official ACL style, BibTeX |
| Primary source of numbers | `../answer/results/` and tables generated from it |
| Primary method sources | `../README.md`, `../configs/`, `../src/`, and retained run manifests |
| Source restriction | `main(5).pdf` is not a framing, writing, or result source |
| Venue status | EACL 2026 is concluded. Treat its call as a style/policy reference and re-check the next target venue’s call before submission. |

## 2. Strongest defensible research question

**RQ:** *On an adjudicated Vietnamese-language test set covering six pragmatic phenomena and including a documented social-media source component, how do Vietnamese, multilingual, and Vietnamese-social-media encoders, 7B instruction-tuned models, prompted API baselines, and a multi-task encoder compare; and what does the three-seed ablation record about the corresponding configurations?*

This is deliberately an **evaluative and analytical** RQ. It does not presume that the proposed model outperforms its baselines.

### Evidence and inference

- **Evidence:** the study has 12,000 adjudicated records (8,000/2,000/2,000 train/dev/test), eight main systems including ViSoBERT, three seeds for learned encoders and 7B systems, and generated result/analysis artifacts.
- **Evidence:** Vistral-7B SFT has the highest recorded main macro pragmatic F1 (82.83); the recorded ViPragSent multi-task model has 73.75.
- **Inference:** a benchmark-and-trade-off framing is more credible than a method-superiority framing.
- **Recommendation:** submit as an empirical resource/evaluation paper, with the multi-task model treated as an analyzed baseline rather than the headline state-of-the-art contribution.

## 3. Evidence-bounded contribution claims

| ID | Permitted claim | Evidence | Strength | Boundary |
| --- | --- | --- | --- | --- |
| C1 | The study defines and evaluates six pragmatic phenomena in 12,000 adjudicated Vietnamese-language records: 10,000 retained local social-media export records and 2,000 author-created, context-augmented derivatives based on VIVID idiom/proverb materials, alongside polarity and emotion annotations. | `answer/data_provenance/gold_build_report.json`; `answer/results/annotation_agreement.json`; `README.md` | Strong | Do not call all records social-media or claim the raw text is publicly released. |
| C2 | The study provides a reproducible comparative evaluation across encoders (including ViSoBERT), 7B SFT models, and prompted API baselines, with recorded external three-seed diagnostics where available. | `answer/results/main_pragmatic.json`; `answer/results/p0_visobert_baseline.json`; `answer/run_manifests/`; `answer/reproducibility/verification_manifest.json` | Strong | API baselines are single-run prompted baselines, not three-seed training runs. |
| C3 | The recorded main evaluation identifies Vistral-7B SFT as the strongest evaluated system (82.83 macro F1), while ViPragSent records 73.75 macro F1. | `answer/tables/main_pragmatic.md`; `answer/results/main_pragmatic.json` | Strong | Do not claim ViPragSent wins this comparison. |
| C4 | In the three-seed ablation, the full multi-task configuration records lower pragmatic F1 than the no-multitask PhoBERT reference; external diagnostics remain separately scoped evidence. | `answer/results/p0_multi_seed_ablation.json`; `answer/results/p0_visobert_baseline.json`; `answer/tables/p0_p1_p2_experiments.md` | Moderate | Say “recorded trade-off association”; do not assert causality, robustness, or a universal effect. |
| C5 | The comparison is traceable from reported values to prediction files, run manifests, result hashes, and pinned external model archives. | `answer/results/claim_ledger.csv`; `answer/reproducibility/verification_manifest.json`; `answer/reproducibility/artifact_registry.json` | Strong | This is file-level reproducibility, not an independent rerun. |
| C6 | The three-seed low-resource sarcasm results are exploratory, with mixed monotonicity across the two systems. | `answer/results/p2_multi_seed_low_resource.json`; `answer/tables/p0_p1_p2_experiments.md` | Moderate | Do not claim data-efficiency superiority; summarize in analysis and retain full detail in the appendix. |
| C7 | The source-stratified sensitivity analysis reports descriptive results for the evaluated `visobert_local` and VIVID-labelled strata. | `answer/results/p1_source_stratified_sensitivity.json`; `answer/tables/p0_p1_p2_experiments.md` | Moderate | No causal source effect, dataset-source superiority, or general domain-transfer claim. |

## 4. Claims that must not appear

- “ViPragSent outperforms all baselines,” “state of the art,” or an equivalent superiority claim.
- A public-data-release claim, because raw/processed social-media text is governed as private research material.
- A claim that generated rationales were manually faithfulness-verified.
- A claim that the low-resource experiment establishes a consistent data-efficiency advantage.
- A general transfer or retention claim based on the external ordinary-task diagnostic: the recorded external results do not support it.
- A causal, universal, or robustness claim about auxiliary tasks based on the recorded ablations.

## 5. Proposed architecture

No page or word-count constraint is active at this stage. Section lengths are editorial guidance only and will be recalibrated after the target venue is selected.

| Section | Purpose and evidence |
| --- | --- |
| Abstract | RQ, data/task scope, comparative result, and bounded trade-off finding. Draft only after claims are confirmed. |
| 1. Introduction | Gap: Vietnamese social-media evaluation rarely isolates the recorded pragmatic phenomena. End with RQ and bounded contributions C1–C7. |
| 2. Related Work | Vietnamese social-media resources/models; pragmatic/sentiment evaluation; multi-task auxiliary-task trade-offs. |
| 3. Task and Data | Six phenomena, annotation/adjudication, splits, label scope, governance/access boundary, agreement. |
| 4. Systems and Evaluation | Systems, seeds, training/evaluation protocol, macro-F1, bootstrap CI, significance, traceability. |
| 5. Results | Main comparison, per-phenomenon behavior, three-seed ablation, and qualified external diagnostics. |
| 6. Analysis | Confusion/calibration plus descriptive P1 and exploratory P2 analyses. |
| 7. Conclusion | Answer the RQ without a superiority conclusion. |
| Limitations and Ethical Considerations | Data governance, sensitive social text, access, annotation, and misuse. |

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
| Main Table 3 | Three-seed ablation | `answer/tables/p0_p1_p2_experiments.md` | Keep configuration scope and intervals explicit. |
| Main Figure 1 | Per-phenomenon comparison | `answer/figures/fig2_per_phenomenon.svg` | Use to show variation across phenomena, not a claim of general superiority. |
| Main Figure 2 (if space) | Pragmatic-polarity confusion | `answer/figures/fig5_confusion.svg`; `answer/tables/error_confusion.md` | Analysis only; name the evaluated system in caption. |
| Appendix Table A1 | Inter-annotator agreement | `answer/tables/annotation_agreement.md` | State the fixed adjudicator/post-adjudication qualification. |
| Appendix Table A2 | Label prevalence | `answer/data_provenance/gold_build_report.json` | Included to document evaluated label composition. |
| Appendix Table A3 | P1 source sensitivity and P2 low-resource sarcasm | `answer/tables/p0_p1_p2_experiments.md` | Label P1 descriptive and P2 exploratory three-seed evidence. |
| Appendix Table A4 | Three-seed external diagnostics | `answer/results/p0_multi_seed_ablation.json`; `answer/results/p0_visobert_baseline.json` | Keep distinct datasets and label spaces explicit; do not pool results. |
| Retained artifact detail (not current appendix) | Paired bootstrap, learning curves, calibration, reproducibility manifest | `answer/tables/significance.md`; `answer/figures/fig6_learning_curves.svg`; `answer/figures/fig7_calibration.svg`; `answer/reproducibility/verification_manifest.json` | Keep available in the verified artifact bundle; add only if a future venue requires these supplementary items. |

## 7. Missing information and experiments

| Priority | Gap | Why it matters | Recommended action |
| --- | --- | --- |
| High | No public raw-data release or confirmed controlled-access protocol | A resource claim needs an access story. | Obtain source-rights permission or document a reviewer-access protocol and release only permitted metadata/code. |
| Medium | No qualitative error examples suitable for release | Confusion matrices alone give limited pragmatic insight. | Prepare de-identified, permission-safe error examples or state that privacy prevents sharing examples. |
| Medium | Rationale faithfulness audit is waived | Rationale outputs cannot support explanation-quality claims. | Do not claim faithful explanations; add a human audit only if the paper needs that contribution. |
| Medium | P1 source strata are observational and unbalanced (1,666/334) | They cannot establish a causal source-domain effect. | Retain descriptive wording and the VIVID governance boundary. |
| Medium | Low-resource comparison is three-seed, has mixed monotonicity across systems, and includes wide intervals | It cannot support data-efficiency superiority. | Keep it exploratory and report uncertainty. |
| Medium | External ordinary-task diagnostic does not demonstrate broad retention | The proposed model is not the best reported system on those datasets. | Present as a trade-off diagnostic, not positive transfer. |
| Submission gate | Next target venue is unknown | EACL 2026’s cycle is complete. | Before drafting for submission, identify the target ARR venue/cycle and re-verify its current call, template, anonymity, checklist, and page rules. |

## 8. Drafting roadmap

1. **Confirm this framing.** Decide whether the paper is an evaluation/resource paper (recommended) rather than a proposed-method paper.
2. **Lock the contribution ledger.** Use C1–C7 as the current claim pool; add no claim until it has an evidence row.
3. **Build related-work matrix.** Start from verified records in `references.bib`; add dataset, pragmatics, and multi-task sources only after primary-source verification.
4. **Create an anonymous ACL project.** Download the current official style files into `manuscript/latex/` only after the submission venue/cycle is chosen. Do not put identifiable repository links in the anonymous version.
5. **Draft in evidence order.** Task/Data → Systems/Evaluation → Results → Analysis → Related Work → Introduction → Abstract → Limitations/Ethics.
6. **Run paper checks.** Every numerical sentence must map to `answer/results/claim_ledger.csv`; every external claim must map to a verified BibTeX key.
7. **Submission preparation.** Add responsible-NLP checklist, data-access statement, ethical considerations, AI-use disclosure as applicable, author contributions, funding, and conflict-of-interest statement.

## 9. Current plan gate

Before submission preparation, confirm:

1. the evaluation-and-trade-off RQ;
2. that C1–C7 are the permitted contribution claims;
3. that the P0 three-seed ablation and ViSoBERT baseline are verified at artifact level; and
4. the intended future venue/cycle, whose page rules will be checked then.
