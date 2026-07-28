# ViPragSent Fair-Framework Cycle

## Verification

- Raw baseline predictions were recomputed with the repository metric implementation.
- The immutable four-decimal registry is retained for display; every value is consistent by four-decimal rounding or within `1e-4` percentage points.
- Full-precision raw seed-mean maxima, recorded in `initial_verification.json`, were used for every margin below.
- Train/dev/test SHA-256 hashes are unchanged and all development/test ID sets contain exactly 2,000 unique records.

## Completed stages

1. Nested five-fold robust probability calibration, non-uniform ensemble, and four dynamic gate families.
2. Single-seed PhoBERT targeted screens trained only on canonical training records:
   - irony: attentive pooling + residual head (seed 20260701);
   - idiom: CLS/mean/max pooling + residual head (seed 20260711);
   - code-switching: CLS/mean/max pooling + residual head + deterministic token auxiliary loss (seed 20260721).

The ViSoBERT targeted smoke screen was rejected before training because the pinned archive lacks a tokenizer configuration compatible with the installed Transformers runtime. It is retained as a smoke-test failure, not a final experiment.

## Best rejected candidate

`FFO-targeted-labelwise-screen` was selected by maximum minimum full-precision baseline margin. Its target experts were combined labelwise with incumbent development sources; every threshold was fit without the corresponding outer-fold labels.

| Metric | Candidate OOF F1 | Full-precision baseline maximum | Margin |
| --- | ---: | ---: | ---: |
| implicit_sentiment | 67.6774016397 | 60.8469599667 | +6.8304416729 |
| sarcasm | 79.4684158351 | 80.0317999369 | -0.5633841019 |
| irony | 97.5206944535 | 97.4131500651 | +0.1075443884 |
| idiom_figurative | 97.0021876141 | 97.2957617951 | -0.2935741809 |
| code_switching | 80.3568055770 | 81.9458209143 | -1.5890153373 |
| mocking | 81.3273580879 | 81.9802025151 | -0.6528444272 |
| macro_pragmatic_f1 | 83.8921438679 | 82.8250207520 | +1.0671231159 |

The target screens improve irony over its required maximum, but idiom, code-switching, sarcasm, and mocking remain below their full-precision baseline thresholds. Under successive halving there is no development-safe candidate to advance to multi-seed confirmation, freezing, or the one permitted candidate-test evaluation. `final_best_tuned` and all baseline artifacts remain unchanged.

NOT_PROMOTED
