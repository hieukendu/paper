# ViPragSent comparison bundle

This directory is a comparison artifact; it does not edit or replace the manuscript.

The green system in `macro_f1_comparison.svg` is the locked current ViPragSent fine-tuned hybrid. The other rows are newly executed **deployment-default** runs. They use the fixed, non-searched profile in `protocol/deployment_default_baselines.yaml`; they are not replacements for historical best-tuned baseline upper bounds.

## Test-set comparison

| System | Implicit | Sarcasm | Irony | Idiom | Code-switch | Mocking | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViPragSent current | 68.0293 | 80.4581 | 97.3454 | 96.1748 | 82.0607 | 81.8970 | 84.3275 |
| ViSoBERT default | 57.4438 | 80.7630 | 97.0483 | 97.3454 | 76.9682 | 81.6943 | 81.8772 |
| XLM-R large default | 52.3036 | 77.4534 | 97.3454 | 97.1965 | 80.3129 | 81.0874 | 80.9499 |
| PhoBERT default | 46.9214 | 77.3049 | 96.9009 | 97.1965 | 73.4842 | 80.6390 | 78.7411 |
| Vistral-7B default | 49.4282 | 71.5297 | 87.0266 | 94.1710 | 50.1903 | 52.9180 | 67.5440 |
| Sailor-7B default | 47.6599 | 46.2759 | 47.4928 | 47.5341 | 48.6504 | 47.5468 | 47.5267 |

## Contents

- `comparison_metrics.csv`: every metric for every system.
- `macro_f1_comparison.svg`: visual macro-F1 comparison.
- `systems/`: test predictions, metrics, run history, configuration manifests, and symbolic links to canonical trained weights/adapters.
- `vipragsent_current/`: selected-hybrid provenance and locked current output.
- `protocol/`: fixed deployment-default configuration and the 512-example sample provenance. The raw private training texts are intentionally not copied.
- `artifact_index.json`: source paths and checksums for copied artifacts.

## Framework-optimization addendum

`framework_optimization_dev/` contains the subsequent development-only
ViPragSent ablation, calibrated development predictions, run histories, and
canonical artifact aliases. It is explicitly separate from this locked test
comparison: the historical test split was already observed, so the addendum
does not make a new test-superiority claim or alter the deployment-default
protocol.
