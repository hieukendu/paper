# Loop 01 Independent Re-review

Reviewer mode: `academic-paper-reviewer` re-review, independent read-only agent
Reviewed build: 14-page PDF, SHA-256 `1C4A717A39B27A50523DCC7AD275C2D2D0DDFE7C9D347193AF01CF137719E0C4`
Date: 2026-07-30

## Verified strengths

- The numerical claims are consistent with `final_comparison.json` and `final_config.yaml`: 84.3883, 82.8250, +1.5633, six label gaps, blends, thresholds, OOF scope, and the one-time-test boundary.
- Citation use and synchronized bibliography status passed at the time of review.
- Figure 2 labels are legible; Ethical Considerations begins intact; the earlier forced one-column appendix problem is removed.
- No critical scientific-result defect was identified.

## Issues found

| ID | Severity | Finding | Decision |
| --- | --- | --- | --- |
| RER-01 | MAJOR | Section commands and two captions inside a single `table*` made Appendix D/E structurally fragile and left p.14 poorly balanced. | ACCEPTED and repaired in the post-review build. |
| RER-02 | MAJOR | Direct Hugging Face and GitHub URLs in bibliography contradicted the anonymous-PDF link plan. | ACCEPTED and repaired in the post-review build. |
| RER-03 | MINOR | 5.11388pt appendix overfull `\hbox` was unresolved. | ACCEPTED and repaired by reflow; post-repair log is clean. |
| RER-04 | MINOR | Date-shaped source IDs appeared in Section 4 prose. | ACCEPTED and repaired by retaining opaque IDs only in traceability artifacts. |

## Author-only matters retained

- Written authorization for local ViSoBERT-export text.
- Permission/licence decision for VIVID-derived and third-party diagnostic artifacts.
- Confirmation that date-shaped experiment identifiers are opaque run IDs rather than calendar assertions.
- ARR/EACL registration, author/conflict fields, Responsible NLP checklist answers, disclosure, and release-host decisions.

This review did not count as a completed stable loop because the cited agent-fixable issues were found before the post-review repair. The next fresh re-review determines whether Loop 01 can be counted.
