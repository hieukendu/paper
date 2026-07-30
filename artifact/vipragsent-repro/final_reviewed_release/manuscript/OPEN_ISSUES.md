# ViPragSent open issues

## Phase-0 resolved

- Current result authority is `answer/final_best_tuned/`, not the historical 73.7469 configuration.
- Vistral-7B is a retained baseline at 82.8250, not the current aggregate leader.
- `true_anchor_arbiter_cycle` is stopped, has no canonical-test access, and is NOT_PROMOTED.
- Gold-corpus composition is 10,000 retained local ViSoBERT-export plus 2,000 VIVID-derived, context-augmented candidate records.

## Remaining dependencies

| ID | Issue | Required treatment |
| --- | --- | --- |
| P-01 | Canonical-PDF state | Resolved: `manuscript/latex/main.pdf` is tracked; checksum and clean-build evidence are in `eacl_review_loop/CANONICAL_BUILD_RECORD.md`. |
| P-02 | Citation review | Resolved: all in-text citation occurrences have a primary/official-source audit in `eacl_review_loop/FINAL_CITATION_AUDIT.md`. Source authorization remains a separate author/legal dependency. |
| P-03 | Final-system checkpoint access | Public checkpoint and QLoRA archives are verified; local workspace absence and public availability must remain distinct. The evidence supports artifact access, not an independent full training rerun. |
| P-04 | Three label-specific baseline leaders exceed ViPragSent-Final. | Preserve the aggregate-only qualification in all future prose. |
