# Exact OOF continuation cycle

Status: completed without promotion. The authoritative `../final_best_tuned/`
bundle was not modified.

## Protocol

- Canonical incumbent was recomputed before the cycle: macro pragmatic
  binary-macro-F1 `84.3883`.
- Selection used only the 2,000-row development split. Five deterministic folds
  preserve label marginals, pairwise label co-occurrences, platform, source
  dataset, augmented/natural status, and sizeable annotation batches. The exact
  assignments and fold diagnostics are in `configs/development_folds.json`.
- The probability bank contains every aligned three-seed ViSoBERT/PhoBERT source,
  all individual seeds, all two-seed/leave-one-out variants, and the incumbent
  probability source. There are no valid five-seed or additional checkpoint
  probability artifacts, so those variants were not invented.
- Cross-backbone blends were evaluated for every alpha from `0.300` through
  `0.850` in `0.025` increments for irony, idiom, and code-switching. Thresholds
  were searched over all unique development probabilities and their midpoints.
  Idiom uses the predeclared OOF false-positive constraint of at most one.
- The candidate was frozen before its only canonical-test evaluation. Test
  labels were not used in source, alpha, threshold, seed, or checkpoint choice.

## Frozen exact candidate and one test evaluation

The selected development-only configuration was:

| Label | ViSoBERT weight | PhoBERT weight | Exact full-dev threshold |
| --- | ---: | ---: | ---: |
| irony | 0.500 | 0.500 | 0.962295 |
| idiom_figurative | 0.475 | 0.525 | 0.926498 |
| code_switching | 0.525 | 0.475 | 0.497932 |

Protected labels remained byte-for-byte the incumbent source and threshold.
The single post-freeze test result was `84.3797`, versus incumbent `84.3883`.
It improved idiom by `+0.1634`, but lowered irony by `-0.1651` and code-switching
by `-0.0499`. It therefore fails the predeclared promotion rule and is retained
only as a rejected candidate at `frozen_exact_target_cross/`.

## Compact encoder screen

One materially different Stage-A encoder was trained on train/dev only:
ViSoBERT with CLS+mean+max pooling, an MLP pragmatic head, LR `1.5e-5`, dropout
`0.05`, weight decay `0.03`, warmup `0.03`, seed `20261021`, and no test
prediction. Its exact OOF macro was `82.3847`; target selection score was
`90.1011`, below the probability-bank frozen candidate's `91.7152`. It was
rejected under successive halving and was not advanced to multi-seed training.

## Interpretation

The valid incumbent still beats the registry best-tuned baseline on implicit
sentiment, sarcasm, mocking, and overall macro-F1. It still trails the verified
best-tuned baseline maxima on irony (`-0.3955`), idiom (`-0.1138`), and
code-switching (`-0.1294`). This cycle does not support a full-superiority claim.

All branch-level results are retained in `experiment_registry.csv` and
`experiment_registry.json`; OOF details are in
`oof/exact_existing_probabilities/`; the compact screen report is in
`encoder_screen/`. No baseline configuration, prediction, metric, or paper file
was altered.
