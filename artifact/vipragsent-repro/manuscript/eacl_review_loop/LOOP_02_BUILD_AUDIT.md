# Loop 02 Build Audit

- Build date: 2026-07-30.
- Command: `xelatex -interaction=nonstopmode -halt-on-error main.tex`, `bibtex main`, then two further XeLaTeX passes in `manuscript/latex`.
- Canonical PDF: `artifact/vipragsent-repro/manuscript/latex/main.pdf`.
- Pages: 14.
- SHA-256: `04b5be0d80b605a216d0935fda423008935f753f5231895fe0e08c9a303c0ffb`.
- Compiler-log audit: PASS -- no undefined citations/references, duplicate labels, rerun warning, or overfull hbox/vbox match.
- Visual audit: PASS -- rendered review of the complete PDF found no clipping, overflow, anonymous-link leak, or material float/heading defect. Figure 2, the ethics heading, and Appendix D/E were specifically rechecked after the preceding repairs.
- Result-text checks: PASS for `ViPragSent-Final`, `84.39`, `1.56`, and the three per-label qualifications. Historical `73.75` remains explicitly historical only.
