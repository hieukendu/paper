# Final Best-Tuned Manuscript Revision Plan

## Evidence Rule

Promoted machine-readable artifacts under `answer/final_best_tuned/` are authoritative for final-system values. The older `answer/results/` artifacts remain authority only for legacy ablation, source-stratified, low-resource, calibration, and governance material that is explicitly attributed to its evaluated system version.

## Affected Manuscript Files

| File group | Current role | Required revision |
|---|---|---|
| `latex/sections/04_systems_and_evaluation.tex` | System and protocol | Add the frozen label-wise ensemble, three-seed sources, development-only OOF selection, thresholds, one-time test evaluation, and artifact-level reproducibility boundary. |
| `latex/tables/table_main_pragmatic.tex` | Main results table | Replace obsolete 73.75/current-system framing with complete best-tuned baselines and ViPragSent-Final. |
| `latex/tables/` (new supplementary tables) | Supporting comparisons | Add historical progression and two-view gap tables. |
| `latex/figures/` and figure source | Visual evidence | Generate macro-comparison, per-label gap, and final-versus-leader figures from `final_comparison.json`; do not edit legacy SVG labels. |
| `latex/sections/05_results.tex` | Results | Report aggregate lead, all six label gaps, mixed leadership, and legacy-experiment attribution. |
| `latex/sections/06_analysis.tex` | Analysis | Add observed evidence, bounded interpretation, and untested-hypothesis treatment of the final ensemble. |
| `latex/sections/01_introduction.tex`, `latex/main.tex`, `latex/sections/07_conclusion.tex` | Positioning | Update RQ, contributions, title, abstract, and conclusion after results stabilize. |
| `latex/sections/08_limitations.tex`, `latex/sections/09_ethics.tex` | Boundaries | Retain governance limits and add final-system complexity, one-time test evaluation, and no per-label dominance. |
| Control, traceability, and audit files | Project state | Replace superseded result statements; create claim-evidence and numerical consistency records. |

## Dependency Order

1. Systems and evaluation.
2. Result tables and figures.
3. Results and analysis.
4. Reproducibility, limitations, ethics, conclusion, introduction, abstract, and title.
5. Integrity, review, revision, re-review, and final formatting audits.
