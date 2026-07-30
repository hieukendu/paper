# ViPragSent manuscript workspace

This directory is the working area for an EACL-style paper. It intentionally contains planning and source-traceability material only; it does not contain a manuscript draft yet.

## Start here

- `PAPER_PLAN.md` — confirmed-evidence paper blueprint, page budget, and drafting roadmap.
- `EVIDENCE_MAP.md` — claim, table, and figure traceability map.
- `references.bib` — a small set of externally verified BibTeX records. It is not yet the complete bibliography.
- `latex/README.md` — the official ACL template source and the pre-drafting setup rule.

## Authority rules

1. Use `../answer/` as the sole source of numerical results.
2. Use repository code, configuration, and `../README.md` for methods and protocol.
3. Add related-work citations only after verifying primary sources.
4. Do not use `main(5).pdf` for title, claims, prose, paper organization, or conclusions.

The paper must not claim that the proposed multi-task model is the best-performing system: the recorded main evaluation has a higher macro-F1 for Vistral-7B SFT and for the no-multitask PhoBERT reference.
