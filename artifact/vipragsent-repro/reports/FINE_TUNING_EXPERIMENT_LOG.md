# Fine-tuning experiment provenance

This report is generated from the machine-readable registry and is suitable as a paper appendix/run log.

## Protocol

- Training labels: `vipragsent_train.jsonl` only.
- Architecture/threshold/source selection: `vipragsent_dev.jsonl` only.
- Test labels have not been used in this optimization phase.

## Runs

| System | Seed | Epochs logged | Best logged dev macro-F1 |
| --- | ---: | ---: | ---: |
| finetune_phobert_weighted_cls_s2 | 20260725 | 11 | 0.8218 |
| finetune_phobert_weighted_mean_s1 | 20260724 | 6 | 0.8187 |
| finetune_visobert_weighted_cls_s1 | 20260726 | 5 | 0.8305 |
| finetune_vistral_multilabel_4k_v2 | 20260728 | 1 | 81.2804 |
| qlora_smoke_fix | 20260729 | 1 | 52.6304 |

Machine-readable registry: `answers/optimized_vipragsent/flexible_finetuning/run_registry.json`.
