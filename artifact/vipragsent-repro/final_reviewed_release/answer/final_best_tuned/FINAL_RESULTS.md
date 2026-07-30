# FINAL RESULTS

All values are binary macro-F1 (%). The final candidate was selected exclusively with five-fold OOF development evaluation and was evaluated once after freeze on the canonical test split.

| Metric | New final | Incumbent | Change | Best-tuned max | Gap to baseline | Promotion |
|---|---:|---:|---:|---:|---:|---|
| implicit_sentiment | 67.9829 | 67.9829 | +0.0000 | 60.8470 | +7.1359 | PASS |
| sarcasm | 80.2781 | 80.2781 | +0.0000 | 80.0318 | +0.2463 | PASS |
| irony | 97.0177 | 96.9009 | +0.1168 | 97.4132 | -0.3955 | PASS |
| idiom_figurative | 97.1820 | 97.1965 | -0.0145 | 97.2958 | -0.1138 | PASS |
| code_switching | 81.8164 | 80.1313 | +1.6851 | 81.9458 | -0.1294 | PASS |
| mocking | 82.0528 | 82.0528 | +0.0000 | 81.9802 | +0.0726 | PASS |
| macro_pragmatic_f1 | 84.3883 | 84.0904 | +0.2979 | 82.8250 | +1.5633 | PASS |

## Promotion decision

Promoted: macro pragmatic F1 increased from 84.0904 to **84.3883**; irony and code-switching improved; no pragmatic label declined by more than 0.10. The incumbent bundle is archived under `archive/38fa5fc/`.

The new system still does not establish strict all-metric superiority over best-tuned historical baselines: irony is 0.3955 below Sailor-7B, idiom is 0.1138 below PhoBERT/XLM-R-large, and code-switching is 0.1294 below Vistral-7B.
