# ViPragSent open issues

This log separates blockers for specific drafting work from non-blocking study limitations. The existing experiment inventory remains **VERIFIED**.

## Pre-submission decisions (not blockers for drafting)

| ID | Issue | Affected work | Required resolution |
| --- | --- | --- | --- |
| P-03 | Submission venue/cycle is TBD. | Final page-limit, checklist, and policy compliance | Select a future venue and verify its live instructions before submission. The eight-page allocation is only an internal planning target and does not invalidate the current draft. |

## Non-blocking study limitations and future extensions

| ID | Item | Required manuscript treatment |
| --- | --- | --- |
| N-01 | Multi-task ablation has one seed. | Describe C4 as a single-seed observed trade-off; no causal/general claim. |
| N-02 | Low-resource study has one seed per budget and non-monotonic values. | Keep C6 exploratory, preferably in appendix. |
| N-03 | ViSoBERT is not an evaluated baseline. | State as a limitation/future evaluation, not as a completed experiment. |
| N-04 | No manual rationale-faithfulness audit. | Do not claim rationale faithfulness. |
| N-05 | No safe qualitative examples are currently available. | Do not add raw examples; explain the privacy boundary if necessary. |
| N-06 | Source data is private research material. | Include data-access and governance boundary; do not claim public release. |
| N-07 | Prompted API baselines are single-run. | Preserve run qualification in tables and prose. |
| N-08 | Calibration exists only for systems with stored confidence. | Do not compare missing confidence as zero/valid calibration. |
| N-09 | Canonical AIVIVN organizer attribution and license remain unresolved. | Keep the local artifact identified as a hash-verified Kaggle mirror; cite Nguyen et al. (2020) only as a scholarly dataset description. Do not claim an organizer-authored release, canonical split, original test labels, or redistribution permission. |
| N-10 | No primary license/permission record has been verified for the exact UIT-VSFC, UIT-VSMEC, or AIVIVN external artifacts; the upstream identity and terms for the 2,000 VIVID-identified replacement rows are also pending. | Keep all raw and processed text private. Before submission, archive applicable research-use authorization or reviewed terms for each exact source artifact. Do not redistribute, relicense, or promise a release until then. |

## Integrity controls

- Numerical claims must originate in `answer/` and map to `answer/results/claim_ledger.csv` or an equivalent explicit JSON path.
- Method/protocol claims must map to repository source/configuration/run-manifest evidence.
- External claims require a verified BibTeX entry before inclusion in manuscript prose.
- No text, evidence, title, framing, or conclusions may be drawn from `main(5).pdf`.
- No identifying authorship, affiliation, acknowledgement, funding, or repository URL may enter the anonymous draft.

## Current drafting status

The anonymous ACL scaffold is present and the manuscript compiles.  The remaining organizer/license and source-authorization issues do not make a local mirror canonical and must be resolved or author-approved as submission limitations before venue submission.
