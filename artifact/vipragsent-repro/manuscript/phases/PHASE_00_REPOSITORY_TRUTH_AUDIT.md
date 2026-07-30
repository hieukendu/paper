# Phase 0 — Repository Truth Audit

## 1. Latest commit and PR inventory

- Base and current `origin/main`: `d49fa0b58fe05379fee5215d4fde57e4a5d89ef0`, merged PR #12, **Revise ViPragSent manuscript for final best-tuned results**.
- PR #12 parent: `f7ef5c7`, which added final-system validation, claim locks, final tables/figures, and revised LaTeX source.
- Prior merged PR #11: `0bc584a2bc2ebffc81e5eb8ef2f5a328474cd3ce`, **Add true-anchor arbiter foundation and OOF evidence**; it recorded a stopped development cycle.

## 2. Authoritative-result hierarchy

1. `answer/final_best_tuned/` is PROMOTED_CURRENT: 84.3883 macro pragmatic F1; five-fold development selection; one canonical-test evaluation.
2. `answer/final_best_tuned/archive/38fa5fc/` is SUPERSEDED incumbent evidence.
3. Historical `answer/results/` items are VALID_HISTORICAL only when their evaluated configuration is named.
4. `answer/true_anchor_arbiter_cycle/` is NOT_PROMOTED and cannot supply a final test result.

## 3. Promoted/non-promoted experiment matrix

See `../EXPERIMENT_STATUS_INVENTORY.md`. In particular, the 97.3861277 OOF idiom value is development-only: the cycle was stopped, its target methods and canonical test were not run, and its own report says `NOT_PROMOTED`.

## 4. Stale-control-file list

The following pre-Phase-0 control records contained active 73.75/82.83 framing and were repaired: `DRAFTING_STATE.md`, `PAPER_PLAN.md`, `DETAILED_OUTLINE.md`, `OPEN_ISSUES.md`, `SECTION_01_TRACEABILITY.md`, and `EVIDENCE_MAP.md`. The root README's corpus-composition wording was also repaired.

## 5. README and manuscript contradictions

- The root README formerly said all 12,000 records were solely local ViSoBERT exports; authoritative provenance records 10,000 retained local records plus 2,000 VIVID-derived, context-augmented candidate records.
- Earlier controls described Vistral-7B (82.8250) as current aggregate leader and ViPragSent (73.7469) as current. Those are historical/progression values, not the promoted final result.
- Current `main.tex` already uses the promoted 84.3883/+1.5633 framing and appropriate per-label qualifications; it was not edited in Phase 0.

## 6. Current PDF freshness verdict

**STALE.** No tracked `artifact/vipragsent-repro/manuscript/latex/main.pdf` exists at the base commit. An untracked local PDF text extract agrees with the current `main.tex` final-system abstract, but it cannot serve as the committed manuscript PDF. Phase 8 owns rebuild and commit; no PDF was rebuilt or committed here.

## 7. Exact files repaired

- `README.md`
- `manuscript/DRAFTING_STATE.md`
- `manuscript/PAPER_PLAN.md`
- `manuscript/DETAILED_OUTLINE.md`
- `manuscript/OPEN_ISSUES.md`
- `manuscript/SECTION_01_TRACEABILITY.md`
- `manuscript/EVIDENCE_MAP.md`
- `manuscript/EXPERIMENT_STATUS_INVENTORY.md` (new)
- `manuscript/phases/PHASE_00_REPOSITORY_TRUTH_AUDIT.md` (new)

## 8. Remaining phase dependencies

Literature review and prose expansion are intentionally deferred. Outstanding dependencies are citation verification, source authorization/licence review, final checkpoint availability for an independent rerun, and Phase-8 PDF rebuild/commit.

## Phase audit

**PASS.** Active Phase-0 controls no longer identify 73.75 as the final system or Vistral-7B as current aggregate leader; corpus composition is consistent with gold-build/provenance records; `true_anchor_arbiter_cycle` is NOT_PROMOTED; the PDF is explicitly STALE; and no Section 1–9 manuscript prose file changed.
