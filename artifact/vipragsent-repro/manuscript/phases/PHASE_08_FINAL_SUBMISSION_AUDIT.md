# Phase 8 — Final Submission Audit

## Rebuild

- Command: `xelatex -interaction=nonstopmode -halt-on-error main.tex; bibtex main; xelatex -interaction=nonstopmode -halt-on-error main.tex; xelatex -interaction=nonstopmode -halt-on-error main.tex`.
- Result: PASS (all commands exit 0).
- PDF: `latex/main.pdf`.
- SHA-256: `5459EEF749152BC8C33C322B770BE2A6087F5720DB475D4B4B2817F476014755`.
- Page count: 14.
- Text-extraction checks: PASS for `ViPragSent-Final`, `84.39`, `1.56`, the qualified aggregate-leadership statement, and the three per-label deficits (irony, idiom/figurative language, and code-switching). The three specified stale aggregate-result statements were absent.
- Visual audit: PASS. All 14 rendered pages were inspected; figures and tables are legible, references resolve visually, anonymity is preserved, and no clipped content, broken float, or material layout defect was found.

## Integrity status

Numerical, provenance, result-scope, PDF text, and compile checks pass. The rebuilt PDF is included in this Phase 8 commit. However, `ARS_FINAL_INTEGRITY_REPORT.md` retains a human-only item-by-item citation verification requirement. The configured GitHub CLI is not installed, so a draft PR cannot be opened through the required publish workflow in this environment.

The final global search found no active stale result claim. Hits for the 73.75 historical configuration and the stopped `true_anchor_arbiter_cycle` are explicitly labelled historical or NOT_PROMOTED; the active manuscript uses Vistral-7B only as the strongest complete baseline and qualifies the aggregate result.

## Final gate

**NOT PASSED FOR SUBMISSION.** Two consecutive complete audits cannot be truthfully recorded: individual external verification of every bibliography item and every in-text use remains outstanding. No claim of a fully passed citation audit or submission-ready draft PR is made.
