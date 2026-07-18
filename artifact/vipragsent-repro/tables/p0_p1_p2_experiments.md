# P0/P1/P2 strengthening experiments

All values are macro F1 (%), mean across three seeds with normal 95% CI across seeds.

## P0: multi-seed ablation and ViSoBERT

| System | Macro pragmatic F1 |
|---|---:|
| vipragsent_full | 73.75 [73.72, 73.78] |
| phobert_finetune | 81.71 [81.48, 81.94] |
| vipragsent_no_rationale | 73.79 [73.60, 73.97] |
| vipragsent_no_emotion | 74.34 [73.88, 74.80] |
| vipragsent_no_polarity | 73.56 [73.36, 73.76] |
| vipragsent_no_uncertainty | 73.75 [73.62, 73.87] |
| visobert_finetune | 82.25 [81.61, 82.90] |

## P1: source-stratified sensitivity

| System | ViSoBERT-local test | VIVID-derived test |
|---|---:|---:|
| vipragsent_full | 52.56 [52.38, 52.73] | 79.84 [79.71, 79.96] |
| phobert_finetune | 60.60 [58.63, 62.57] | 87.21 [85.48, 88.95] |
| vipragsent_no_rationale | 52.46 [52.30, 52.62] | 79.88 [79.84, 79.92] |
| vipragsent_no_emotion | 53.56 [53.23, 53.90] | 81.61 [79.35, 83.87] |
| vipragsent_no_polarity | 52.51 [52.10, 52.93] | 79.63 [79.35, 79.92] |
| vipragsent_no_uncertainty | 52.52 [52.31, 52.74] | 79.75 [79.41, 80.09] |
| visobert_finetune | 62.08 [60.69, 63.47] | 93.04 [90.55, 95.54] |

## P2: multi-seed low-resource sarcasm

| Positive budget | System | Sarcasm macro F1 |
|---:|---|---:|
| 64 | phobert_finetune | 45.24 [45.24, 45.24] |
| 64 | vipragsent_full | 47.45 [44.84, 50.06] |
| 128 | phobert_finetune | 45.15 [44.97, 45.32] |
| 128 | vipragsent_full | 47.79 [46.84, 48.75] |
| 256 | phobert_finetune | 75.72 [74.92, 76.53] |
| 256 | vipragsent_full | 64.44 [45.57, 83.30] |
| 512 | phobert_finetune | 75.41 [74.48, 76.34] |
| 512 | vipragsent_full | 74.35 [74.30, 74.41] |
| 1024 | phobert_finetune | 76.34 [75.74, 76.93] |
| 1024 | vipragsent_full | 74.89 [74.07, 75.72] |

P1 is descriptive only: VIVID rows retain `unknown_or_dataset_card` licensing in the stored source metadata and require authorization/license review before public release or causal source claims.
