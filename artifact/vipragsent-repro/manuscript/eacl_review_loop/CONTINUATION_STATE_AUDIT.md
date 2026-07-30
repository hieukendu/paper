# EACL 2027 Continuation State Audit

Date: 2026-07-30
Base commit: `cdaf2793fd9c3b592325e2193efe9447b62d635d` (merged PR #24)
Working branch: `codex/vipragsent-eacl2027-review-loop-continuation`

## Recovered state

PR #24 repaired citation metadata, clarified the public checkpoint boundary, rebuilt a 14-page anonymous review PDF, and recorded an initial independent review. It did **not** complete the required fresh citation-by-citation audit and post-repair re-review; its own summary correctly recorded zero completed review loops. This continuation therefore resumes at the missing verification stage, rather than treating PR #24 as convergence.

## Rechecked facts

| Check | Verdict | Evidence |
| --- | --- | --- |
| Current promoted system | `ViPragSent-Final`, 84.3883 macro pragmatic F1 | `answer/final_best_tuned/final_comparison.json` and current results section |
| Complete best-tuned comparator | Vistral-7B, 82.8250; difference +1.5633 | same comparison artifact and Table 2 |
| Historical 73.7469 result | historical original system only | Appendix E and system-version records |
| `true_anchor_arbiter_cycle` | not promoted | Phase 0 state records and current introduction |
| Bibliography synchronization | byte-identical | SHA-256 `0E30C7AC880094308ECE17E9679A6DF31D89310FBDB82DAED7C55157CB05B752` for both bibliography files at audit start |
| Public checkpoints | public artifacts, but not an independent retraining reproduction | `PUBLIC_CHECKPOINT_AUDIT.md`; evaluation-ready encoder checkpoint and public QLoRA adapters have distinct documentation scope |
| Page-limit work | deliberately deferred | `DEFERRED_PAGE_REDUCTION_PLAN.md`; no shortening is performed in this continuation |

## Date-like identifier audit

The repository uses date-shaped numeric values as seed/run identifiers in configurations, manifests, checkpoint-card examples, and final selection matrices (for example `20260520`, `20260831`, and `20260901`). The repository does not establish that every such value is a calendar assertion; the older three-seed values occur consistently as `seed_base`/`seeds` fields, while later values occur as experiment identifiers in development matrices. They are retained as opaque identifiers. The authors must confirm whether the later values encode planned dates or arbitrary IDs before any public release; the manuscript does not turn them into calendar claims.

## Repairs queued from the prior re-review

1. Figure 2 had crowded x-axis labels.
2. The Ethical Considerations heading and first word fragment were split across pages.
3. `\onecolumn` made the final appendix inconsistent and left an almost empty final page.

The first continuation build repairs all three locally. It preserves the 14-page form and does not change results, model selection, datasets, or prose claims beyond layout control.
