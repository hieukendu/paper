# FINAL RESULTS

All values are **binary macro-F1 (%)**, the metric used consistently by the verified baseline registry and the original experiment tables.

| Metric | Final binary macro-F1 | Best-tuned max | Delta | Status |
|---|---:|---:|---:|---|
| implicit_sentiment | 67.9829 | 60.8470 | +7.1359 | PASS |
| sarcasm | 80.2781 | 80.0318 | +0.2463 | PASS |
| irony | 96.9009 | 97.4132 | -0.5123 | FAIL |
| idiom_figurative | 97.1965 | 97.2958 | -0.0993 | FAIL |
| code_switching | 80.1313 | 81.9458 | -1.8145 | FAIL |
| mocking | 82.0528 | 81.9802 | +0.0726 | PASS |
| macro_pragmatic_f1 | 84.0904 | 82.8250 | +1.2654 | PASS |

## Metric validation

The previous positive-class-F1 correction was withdrawn: it compared a different metric with the binary macro-F1 baseline registry. The scores above were directly recomputed from frozen predictions and canonical test labels using `sklearn.metrics.f1_score(..., average="macro")` for every pragmatic label.

## Final decision

**Strict all-metric superiority was not achieved.** ViPragSent leads on implicit sentiment, sarcasm, mocking, and macro pragmatic F1, but remains below the best-tuned maximum on irony (-0.5123), idiom (-0.0993), and code-switching (-1.8145).
