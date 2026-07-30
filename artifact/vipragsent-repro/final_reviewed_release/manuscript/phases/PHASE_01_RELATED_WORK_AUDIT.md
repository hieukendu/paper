# Phase 1 — Related Work and Citation Foundation Audit

## Scope

Only `latex/sections/02_related_work.tex`, the synchronized bibliography files, and the listed citation/traceability records were edited. No Introduction, Methods, Results, Analysis, Conclusion, Limitations, or Abstract source file changed.

## Citation verification table

| Group | Verified primary/official sources | Claim boundary |
| --- | --- | --- |
| Vietnamese models | PhoBERT; XLM-R; ViSoBERT | Model-family context, not universal rank |
| Vietnamese resources | UIT-VSFC; UIT-VSMEC; ViGoEmotions; AIVIVN proceedings description | Task/dataset context; external datasets are not ViPragSent sources |
| Pragmatics and figurative language | Ghosh et al.; VIVID | Irony/figurative context, not complete pragmatics coverage |
| Multi-task learning | Moore and Barnes | Empirical design-family precedent, not a causal gain claim |
| Instruction-tuned baselines | Toukmaji and Flanigan; Sailor; Vistral official card | Baseline scope, not a predicted winner |

The bibliography contains 14 unique keys. The two added records have title, author, year, venue, pages, official URL, and DOI where supplied by the primary record. The two bibliography files have identical SHA-256 hashes.

## Audits

1. Citation existence: PASS — every Section 2 key resolves in the bibliography.
2. Metadata: PASS — the added records were verified via ACL Anthology; existing records retain their primary/official ledger links.
3. Claim alignment: PASS — every paragraph states prior work, its boundary, and the present study's distinction.
4. Missing-seminal-work: PASS WITH BOUNDARY — no claim of exhaustive literature coverage; sources are limited to those needed for the manuscript's comparison and task framing.
5. Domain review: PASS — the section separates task, dataset, model, and evaluation contributions and does not claim ensemble novelty.
6. AI-writing quality: PASS — no citation catalogue, no unsupported ``first''/``no prior work'' assertion, and bounded transitions.

## Compile result

Recorded after the Phase-1 compile: no undefined citations, duplicate BibTeX keys, or unresolved references.

## Verdict

**PASS.** Phase 1 establishes a bounded related-work and citation foundation. Later literature expansion or manuscript sections remain out of scope.
