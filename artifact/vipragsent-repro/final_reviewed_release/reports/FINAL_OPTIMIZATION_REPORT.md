# Final status

NOT_YET_ACHIEVED

The completion criterion requires a strict win on every listed metric. This candidate does not meet it; no success claim is made.

| Task/Metric | Original ViPragSent | Best Baseline | Optimized ViPragSent | Difference | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| implicit_sentiment | 46.9214 | 60.8470 (phobert_finetune) | 64.6340 | 3.7870 | PASS |
| sarcasm | 75.6892 | 80.0318 (vistral_7b_sft) | 77.4834 | -2.5484 | FAIL |
| irony | 96.9829 | 97.4132 (sailor_7b_sft) | 97.3454 | -0.0678 | FAIL |
| idiom_figurative | 97.2365 | 97.2958 (phobert_finetune) | 96.7542 | -0.5416 | FAIL |
| code_switching | 47.6303 | 81.9458 (vistral_7b_sft) | 76.3569 | -5.5889 | FAIL |
| mocking | 78.0213 | 81.9802 (vistral_7b_sft) | 79.8453 | -2.1349 | FAIL |
| macro_pragmatic_f1 | 73.7469 | 82.8250 (vistral_7b_sft) | 82.0699 | -0.7551 | FAIL |

Frozen checkpoint hashes: `PASS`.

The final configuration is selected exclusively by development metrics; test scores are reported once per configuration and are not used to select its parameters. Experiments: E001 threshold ensemble; E002 max-length control; E003 train-only lexical retrieval; E004 frozen contextual k-NN; E005 dev-selected four-way hybrid; E006 source thresholds; E007 extended k-NN; E008 text normalization variants; E009 mean-pooled k-NN; E010 frozen-logit calibration; E011 prototype retrieval; E012 similarity-weighted k-NN/six-way hybrid; E013 XLM-R score blends; E014 expanded lexical rule; E015 Vistral-base frozen score blend; E016 lexical-neighbour retrieval; E017 label dependency; E018 seed weights; E019 weighted ViSoBERT/XLM-R; E020 source/dependency hybrid; E021 nonlinear score aggregation; E022 frozen Vistral few-shot; E023 teencode normalization; E024 Sailor few-shot probe; E025 rationale-projection retrieval; E026 emoji/laughter normalization.
