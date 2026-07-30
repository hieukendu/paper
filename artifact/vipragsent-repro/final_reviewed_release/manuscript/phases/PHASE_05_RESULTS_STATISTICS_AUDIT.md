# Phase 5 — Results and Statistics Audit

- Metric recomputation: PASS. Final label macro-F1 values are 67.9829, 80.2781, 97.0177, 97.1820, 81.8164, and 82.0528; their equal-weight mean is 84.3883. Confusion counts are recorded for every label.
- Comparison arithmetic: PASS. Vistral is 82.8250; final minus Vistral is +1.5633; all six displayed label gaps match `final_comparison.json`.
- Uncertainty: PASS WITH LIMITATION. A 1,000-replicate paired bootstrap exists only for continuation versus archived incumbent: +0.2979, 95\% CI [-0.0714, 0.6463]. No aligned promoted-bundle predictions support paired final-versus-Vistral or per-label-leader inference, so no significance claim or Holm correction is made.
- Tables/figures: PASS. Main comparison table, gap table, macro figure, and label-gap figure match JSON; the redundant label-leader figure was removed.
- Historical scope: PASS. Original 73.7469 and ablations/diagnostics remain historical only.

**Verdict: PASS WITH PAIRED-COMPARISON LIMITATION DISCLOSED.**
