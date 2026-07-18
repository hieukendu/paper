# Claim and artifact evidence map

All paths are relative to the repository root. “Evidence” identifies what is observed in the stored artifact. “Inference” is the maximum paper interpretation currently justified.

| Planned item | Evidence | Inference allowed | Status |
| --- | --- | --- | --- |
| Split and data scale | `answer/data_provenance/gold_build_report.json`; `README.md` | 12,000 adjudicated records; 8,000/2,000/2,000 split | Verified |
| Annotation quality | `answer/results/annotation_agreement.json`; `answer/tables/annotation_agreement.md` | Report observed agreement, with fixed-adjudicator/post-adjudication qualification | Verified |
| Main comparative result | `answer/results/main_pragmatic.json`; `answer/tables/main_pragmatic.md` | Vistral-7B SFT is the strongest recorded system at 82.83 macro F1; ViPragSent is 73.75 | Verified |
| Per-phenomenon behavior | `answer/figures/fig2_per_phenomenon.svg`; `answer/tables/main_pragmatic.md` | Phenomena differ in difficulty across evaluated systems | Verified |
| Statistical comparisons | `answer/results/significance.json`; `answer/tables/significance.md` | Report stored paired-bootstrap intervals and uncorrected finite-resample tail proportions descriptively for named comparisons; no p-value-based conclusion | Verified, qualified |
| Multi-task trade-off | `answer/results/multitask_ablation.json`; `answer/tables/multitask_ablation.md` | Single-seed observed trade-off only | Verified, qualified |
| External-task diagnostics | `answer/results/ordinary_sentiment.json`; `answer/tables/ordinary_sentiment.md` | Dataset-specific reported results; no broad transfer or retention claim | Verified, qualified |
| Low-resource analysis | `answer/results/low_resource_sarcasm.json`; `answer/tables/low_resource_sarcasm.md` | Exploratory, one-seed and non-monotonic comparison | Verified, qualified |
| Calibration and confusion | `answer/results/calibration.json`; `answer/results/error_confusion.json`; figures 5 and 7 | Analysis only for eligible systems / evaluated setup | Verified, qualified |
| Training protocol | `configs/training_h100_20gb.yaml`; `answer/run_manifests/**/run_manifest.json` | State recorded hyperparameters, seeds, and external archive locations | Verified |
| File-level traceability | `answer/results/claim_ledger.csv`; `answer/reproducibility/verification_manifest.json` | Claims can be traced to source result hashes; not a rerun | Verified |
| Raw-text availability | `configs/data_governance.yaml`; `README.md` | Private, non-commercial research material; no public raw-text release claim | Verified |

## Related-work source queue

These are verified primary sources suitable for the initial related-work matrix, but only `references.bib` keys may be cited after citation review.

| Theme | Primary source | Planned function |
| --- | --- | --- |
| Vietnamese social-media language modeling | Nguyen et al. (2023), ViSoBERT | Explain the social-media modeling context and identify the missing ViSoBERT-baseline question. |
| Vietnamese monolingual encoder | Nguyen and Tuan Nguyen (2020), PhoBERT | Identify the PhoBERT baseline’s origin. |
| Multilingual encoder | Conneau et al. (2020), XLM-R | Identify the XLM-R baseline’s origin. |
| Vietnamese sentiment benchmark | Nguyen et al. (2018), UIT-VSFC | Cite when describing external evaluation data. Add a verified BibTeX record before drafting. |
| Vietnamese social-media emotion benchmark | Ho et al. (2019), UIT-VSMEC | Cite when describing external evaluation data. Add a verified BibTeX record before drafting. |
| Auxiliary-task sentiment modeling | Moore and Barnes (2021) or another verified primary source | Position, not justify, the multi-task auxiliary-task design. |

## Traceability rule for drafting

For every numerical sentence, record a comment or drafting note containing:

`claim_id | answer artifact path | JSON path or ledger anchor | interpretation qualifier`

Example: `C3 | answer/results/main_pragmatic.json | $.systems.vistral_7b_sft.metrics.macro_pragmatic_f1.mean | strongest evaluated system, not SOTA`
