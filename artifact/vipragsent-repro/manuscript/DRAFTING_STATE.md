# ViPragSent drafting state

**Phase-0 authority:** `answer/final_best_tuned/` is the current promoted manuscript-result bundle. `answer/true_anchor_arbiter_cycle/` is **NOT_PROMOTED**. This file supersedes earlier planning-state language that treated the 73.7469 multi-task configuration or Vistral-7B's 82.8250 result as current.

## Active research question

> On the fixed ViPragSent evaluation, how does the development-selected, label-wise ViPragSent-Final ensemble compare with the retained best-tuned baseline systems, and where does its aggregate result differ from per-label leadership?

The question is unlocked for future phase-specific refinement; Phase 0 makes no prose change.

## Result and provenance lock

- ViPragSent-Final: 84.3883 macro pragmatic F1, selected using five-fold development OOF evidence and evaluated once after freeze on canonical test labels.
- Strongest complete best-tuned baseline: Vistral-7B at 82.8250; aggregate gap: +1.5633.
- Per-label deficits remain for irony (-0.3955), idiom/figurative language (-0.1138), and code-switching (-0.1294); do not claim strict all-label dominance.
- The historical 73.7469 ViPragSent configuration is retained only as progression/ablation evidence.
- The 12,000-record gold corpus is 10,000 retained local ViSoBERT-export records plus 2,000 VIVID-derived, context-augmented candidate records; split sizes are 8,000/2,000/2,000. Access and authorization limitations remain active.

## Phase-0 boundaries

- Numerical authority: `answer/final_best_tuned/` for final-system claims; older `answer/results/` artifacts only for explicitly historical analyses.
- Method/provenance authority: repository configuration, processed-split provenance, and final-bundle metadata.
- No Section 1–9 LaTeX prose is changed in Phase 0.
- `main.pdf` is not tracked at `manuscript/latex/main.pdf`; the untracked local PDF matches the current final-source abstract, but the committed-PDF state is **STALE** until Phase 8 rebuilds and commits it.
