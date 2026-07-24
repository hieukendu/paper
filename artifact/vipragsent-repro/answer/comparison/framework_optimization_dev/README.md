# ViPragSent framework-optimization addendum (development only)

This bundle records a predeclared four-run framework ablation over ViPragSent's rationale, polarity, emotion, loss-balancing, MLP-head, and focal-loss choices. It does **not** modify the manuscript.

## Outcome

The selected development-only hybrid reaches **84.0411 Macro-F1**. It keeps the locked encoder hybrid for five labels and uses the full multitask ViSoBERT candidate only for `mocking` (80.7145 on dev, versus 80.5806 for the prior locked source).

This is not a new test claim: the historical test split was already observed before this search. Therefore neither all-metric superiority over historical best-tuned baselines nor per-label dominance over every deployment-default baseline is established here.

The deployment-default comparison is deliberately unchanged. Its profile remains a declared, fixed protocol; it was not weakened or adjusted to force a ranking.

## Calibrated development results

| System | Implicit | Sarcasm | Irony | Idiom | Code-switch | Mocking | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViSoBERT full multitask + uncertainty | 65.3069 | 79.5901 | 97.5207 | 97.0177 | 78.2291 | 80.4899 | 83.0257 |
| ViSoBERT full multitask, fixed loss weights | 65.8534 | 79.8894 | 97.5207 | 97.0177 | 78.0093 | 80.7145 | 83.1675 |
| ViSoBERT full multitask, MLP + focal | 66.7774 | 79.4927 | 97.5207 | 96.8689 | 74.9409 | 80.1131 | 82.6189 |
| PhoBERT full multitask, fixed loss weights | 65.9071 | 77.6723 | 97.3724 | 97.0022 | 79.7292 | 78.5177 | 82.7002 |
| Dev-selected hybrid: locked + full multitask mocking | 67.5939 | 80.8124 | 97.6816 | 97.0483 | 80.3963 | 80.7145 | 84.0411 |

## Audit contents

- `dev_ablation_metrics.csv`: the calibrated results in the table.
- `status.json`: scope and claim-status guards.
- `artifact_index.json`: paths, sizes, and hashes for configuration, histories, threshold files, dev predictions, and metrics; checkpoint paths are indexed by size but not rehashed.
- The canonical generated predictions, threshold configurations, selection file, histories, manifests, and checkpoints remain under `answers/framework_optimization/` and `outputs/framework_optimization/`.

All hyperparameters were fixed in `configs/framework_optimization_dev_matrix.yaml` before these four runs. Training used the 8,000-record train split; threshold and source selection used the 2,000-record dev split only.
