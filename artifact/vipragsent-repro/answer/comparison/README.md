# ViPragSent vs verified best-tuned baselines

This directory compares only the final, raw-verified ViPragSent run with verified best-tuned baselines. Deployment-default artifacts were removed. All scores are binary macro-F1 (%).

## Metric gaps

| Metric | ViPragSent final | Best-tuned leader | Leader score | Gap | Status |
|---|---:|---|---:|---:|---|
| implicit_sentiment | 67.9829 | PhoBERT best-tuned | 60.8470 | +7.1359 | PASS |
| sarcasm | 80.2781 | Vistral-7B best-tuned | 80.0318 | +0.2463 | PASS |
| irony | 96.9009 | Sailor-7B best-tuned | 97.4132 | -0.5123 | BELOW |
| idiom_figurative | 97.1965 | PhoBERT best-tuned, XLM-R-large best-tuned | 97.2958 | -0.0993 | BELOW |
| code_switching | 80.1313 | Vistral-7B best-tuned | 81.9458 | -1.8145 | BELOW |
| mocking | 82.0528 | Vistral-7B best-tuned | 81.9802 | +0.0726 | PASS |
| macro_pragmatic_f1 | 84.0904 | Vistral-7B best-tuned | 82.8250 | +1.2654 | PASS |

## Conclusion

ViPragSent final leads on four of seven reported metrics (implicit sentiment, sarcasm, mocking, and macro pragmatic F1). It remains below best-tuned baselines on irony, idiom, and code-switching; strict all-metric superiority is therefore not established.
