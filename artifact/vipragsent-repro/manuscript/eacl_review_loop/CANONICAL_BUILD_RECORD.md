# Canonical Build Record

- Source: `artifact/vipragsent-repro/manuscript/latex/main.tex` in the continuation worktree based on merged-main commit `cdaf2793fd9c3b592325e2193efe9447b62d635d`; the final continuation commit is recorded in `FINAL_CONTINUATION_LOOP_REPORT.md`.
- PDF: `artifact/vipragsent-repro/manuscript/latex/main.pdf`.
- SHA-256: `04b5be0d80b605a216d0935fda423008935f753f5231895fe0e08c9a303c0ffb`.
- Pages: 14.
- Result-text check: PASS for `ViPragSent-Final`, `84.39`, `1.56`, irony, idiom/figurative-language, and code-switching qualifications.
- Historical `73.75` occurs only in explicitly historical context; it is not the current-system result.

No committed `final_build/` or `build_final/` tree is authoritative. Earlier untracked build directories are excluded from canonical status.

Build verification (2026-07-30): `xelatex -interaction=nonstopmode -halt-on-error main.tex`, `bibtex main`, then two further XeLaTeX passes. The 14-page PDF has no undefined citation/reference, duplicate-label, rerun, or overfull-box match in the compiler-log audit. Ordinary underfull-box and local caption-anchor warnings were visually reviewed and do not cause a visible defect.
