# Flexible fine-tuning optimization report

## Status

**NOT_YET_ACHIEVED.** The criterion is a strict test-set improvement over every listed baseline metric. The selected system improves the overall macro-F1 and four individual targets, but misses three targets. No success claim is made.

## Selection protocol

- All neural and lexical models used only `data/processed/vipragsent_train.jsonl` for training.
- Checkpoints, thresholds, and per-label source selection were selected only on `data/processed/vipragsent_dev.jsonl`.
- The test labels were read once, after this configuration was locked, to produce the table below.
- The selected hybrid uses ViSoBERT for implicit sentiment, sarcasm, irony, and mocking; PhoBERT for idiom/figurative language and code-switching.

## Locked test result

| Metric | Best baseline | Selected fine-tuned hybrid | Difference | Result |
| --- | ---: | ---: | ---: | --- |
| implicit_sentiment | 60.8470 | 68.0293 | +7.1823 | PASS |
| sarcasm | 80.0318 | 80.4581 | +0.4263 | PASS |
| irony | 97.4132 | 97.3454 | -0.0678 | FAIL |
| idiom_figurative | 97.2958 | 96.1748 | -1.1210 | FAIL |
| code_switching | 81.9458 | 82.0607 | +0.1149 | PASS |
| mocking | 81.9802 | 81.8970 | -0.0832 | FAIL |
| macro_pragmatic_f1 | 82.8250 | 84.3275 | +1.5025 | PASS |

## Evidence

- Final predictions: `answers/optimized_vipragsent/flexible_test_predictions/encoder_lexical_hybrid_test.json`.
- Final metrics: `answers/optimized_vipragsent/flexible_metrics/encoder_lexical_hybrid_test.csv`.
- Development threshold and source-selection records: `answers/optimized_vipragsent/flexible_configs/`.
- Complete configuration, seed, history, model-selection and environment registry: `answers/optimized_vipragsent/flexible_finetuning/run_registry.json`.

The 4k-example Vistral direct-classification QLoRA run is retained in the registry but was not selected: it reached 81.2804 macro-F1 on development, below the encoder hybrid. The original 8k QLoRA attempts are not evidence because their pre-fix evaluation failed before artifact generation.
