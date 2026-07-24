# Targeted-cycle trainer and OOF audit

Status: no candidate was frozen or evaluated on the canonical test split. The
authoritative `../final_best_tuned/` bundle is unchanged.

## Completed changes

- Recomputed the incumbent from canonical predictions: macro pragmatic
  binary-macro-F1 is `84.38832574644515`; the raw verification is retained in
  `../next_cycle/incumbent_verification.json`.
- Added a provenance- and co-occurrence-aware five-fold train manifest at
  `configs/train_oof_folds.json` (8,000 records; fold sizes 1600, 1600, 1599,
  1601, 1600) and the corresponding held-out records per fold.
- Corrected `train_target_binary_expert.py` before accepting any new result:
  checkpoint selection now uses an ID-partitioned train-holdout calibration and
  target binary macro-F1, PR-AUC, precision, recall, TP/TN/FP/FN, and the
  predeclared FP constraints. It retains the top three checkpoints.
- Replaced the prior asymmetric-loss heuristic with standard asymmetric loss
  (positive/negative gamma, probability clipping, class weights), added
  symmetric Bernoulli R-Drop, EMA checkpoint selection, separate head LR, and
  train-fold exclusion support.

## Execution outcome

The initial apparent runtime failure was not an out-of-memory result. It was a
trainer logging/calibration defect: the train holdout used hash byte 0 and the
calibration split accidentally reused that byte, making its parity constant.
Calibration now uses independent hash byte 1 and manifest serialization handles
path values. The resource-adapted configuration (BF16, `num_workers=0`, maximum
length 96, gradient accumulation, and frozen lower encoder) completed all five
true train-OOF folds on the allocated H100 MIG partition.

`oof_train/irony_train_oof.jsonl` covers exactly 8,000 train records once each;
its train-only OOF result is binary macro-F1 `0.9648146663`, PR-AUC
`0.9361125413`, with 74 FN and 17 FP. The errors are mainly short, colloquial
or context-dependent positive examples rather than a single source/platform
slice. The 74 FN IDs were exported in `oof_train/irony_oof_false_negatives.json`
and used only as an additional train weight in the next run.

The resulting four-top-layer targeted IronyExpert was selected on its separate
train holdout (F1 `0.9613358618`, precision `1.0`, recall `0.8666666667`). It
was then frozen and audited with the existing nested development protocol:
development OOF F1 was `97.2107812602` (1 FP, 17 FN). This is below the existing
best Irony candidate (`97.5206944535`, 0 FP, 16 FN), so it is **rejected**.

No canonical-test labels or predictions were used in this cycle. No rescue/veto
gate, model soup, or final candidate was promoted, and the authoritative
`../final_best_tuned/` bundle remains unchanged.
