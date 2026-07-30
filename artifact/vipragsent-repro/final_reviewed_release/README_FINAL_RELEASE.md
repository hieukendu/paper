# ViPragSent Final Reviewed Release

This folder packages the final reviewed manuscript state from commit `4296b2e56a8ecc061dcc2b452a1f3fd0124088d6`.

## Canonical manuscript

- LaTeX source: `manuscript/latex/main.tex`
- PDF: `manuscript/latex/main.pdf`
- PDF SHA-256: `04b5be0d80b605a216d0935fda423008935f753f5231895fe0e08c9a303c0ffb`
- Review-loop record: `manuscript/eacl_review_loop/FINAL_CONTINUATION_LOOP_REPORT.md`
- Review result: two completed clean review loops; no unresolved agent-fixable content, citation, or PDF-format issue.

## Included

- `manuscript/`: final LaTeX, canonical PDF, bibliography, figures, audit trail, and OpenReview preparation.
- `src/`, `scripts/`, and `tests/`: project source, evaluation/build utilities, and tests.
- `configs/`, `docs/`, `figures/`, `tables/`, and `reports/`: supporting configuration and documentation.
- `answer/final_best_tuned/`: promoted final-system selection, predictions, metrics, thresholds, and provenance artifacts.
- `answer/data_provenance/`: aggregate provenance and gold-build reports.
- Runtime/project files: `pyproject.toml`, `requirements.txt`, `.env.example`, and repository README/licence notices.

## Intentionally excluded

- Raw/private source data, model checkpoints, credentials, caches, compiler by-products, and unrelated historical experiment directories.
- Local spreadsheets, downloaded archives, logs, and other non-final workspace material.

## Important boundary

This is the final reviewed manuscript package, not an assertion that it is ready to submit. The author-only legal, permission, registration, anonymous-hosting, and separate page-reduction actions remain in `manuscript/eacl_review_loop/AUTHOR_ONLY_ACTIONS.md`.
