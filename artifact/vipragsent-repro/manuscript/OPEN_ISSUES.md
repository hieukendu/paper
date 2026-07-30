# ViPragSent open issues

## Phase-0 resolved

- Current result authority is `answer/final_best_tuned/`, not the historical 73.7469 configuration.
- Vistral-7B is a retained baseline at 82.8250, not the current aggregate leader.
- `true_anchor_arbiter_cycle` is stopped, has no canonical-test access, and is NOT_PROMOTED.
- Gold-corpus composition is 10,000 retained local ViSoBERT-export plus 2,000 VIVID-derived, context-augmented candidate records.

## Remaining dependencies

| ID | Issue | Required treatment |
| --- | --- | --- |
| P-01 | `manuscript/latex/main.pdf` is untracked; the local copy matches current source but no committed PDF is available. | Mark PDF state STALE; rebuild/commit only in Phase 8. |
| P-02 | Citation and source-authorization review remain incomplete. | Do not advance submission claims without their designated phases. |
| P-03 | Final system checkpoints are absent locally. | Describe result validation as artifact analysis, not independent training reproduction. |
| P-04 | Three label-specific baseline leaders exceed ViPragSent-Final. | Preserve the aggregate-only qualification in all future prose. |
