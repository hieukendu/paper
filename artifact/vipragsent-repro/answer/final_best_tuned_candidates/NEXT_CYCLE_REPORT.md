# Representation-learning next cycle

Status: completed without promotion. The authoritative `../final_best_tuned/`
bundle remains unchanged, and no new test prediction was generated.

## Integrity protocol

- TAPT used only `vipragsent_train.jsonl` and `vipragsent_dev.jsonl` text.
  It did not read labels or test text. Both original encoders were kept intact;
  adapted MLM states live under `outputs/final_best_tuned_next_cycle/`.
- Target experts use an ID-hashed held-out portion of the training split for
  early stopping. Development is used only for the deterministic five-fold
  threshold audit described in `configs/development_folds.json`.
- The canonical test split was not accessed after the incumbent verification.
  Since no new expert passed Stage B, there is no frozen next-cycle test
  candidate and no opportunity for post-hoc test selection.

## TAPT

| Encoder | Corpus | Epochs | Held-out MLM loss |
| --- | --- | ---: | ---: |
| ViSoBERT | train + dev text (10,000 rows) | 3 | 2.5893 |
| PhoBERT | train + dev text (10,000 rows) | 3 | 4.1352 |

No licensed, repository-local extra social-media corpus with a confirmed
license was found, so DAPT was not introduced.

## Dedicated-expert Stage A/B results

All values below are development-only binary macro-F1 using exact nested OOF
thresholding. `selection = fold mean - 0.25 * fold standard deviation`.

| Candidate | Target | OOF F1 | Fold mean ± sd | Selection | FP / FN | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ViSoBERT TAPT + phrase attention + hard curriculum (seed 20261111) | irony | 97.5207 | 97.5026 ± 1.0640 | 97.2366 | 0 / 16 | advanced to 3 seeds |
| Same, uniform 3-seed | irony | 97.3590 | 97.3313 ± 1.3237 | 97.0004 | 0 / 17 | rejected: ensemble regressed |
| PhoBERT TAPT + CLS/mean + hard curriculum | irony | 97.3724 | 97.3555 ± 0.8401 | 97.1455 | 1 / 16 | rejected |
| ViSoBERT TAPT + phrase attention + hard-positive curriculum | idiom | 96.8689 | 96.8679 ± 1.2915 | 96.5450 | 2 / 18 | rejected: violates FP <= 1 |
| PhoBERT TAPT + CLS/mean/max + hard-positive curriculum | idiom | 97.0022 | 96.9855 ± 1.0897 | 96.7131 | 0 / 19 | rejected |
| ViSoBERT TAPT + contextual code expert + token-language auxiliary + asymmetric loss | code switching | 78.5063 | 78.3900 ± 3.7352 | 77.4563 | 64 / 74 | rejected |

The code expert includes a deterministic pseudo token-language head for six
classes (Vietnamese, English, proper name, brand/product, acronym, other),
plus sentence-level English-token statistics. It is genuinely contextual and
not the previously rejected character TF-IDF model.

## Decision

The only single-seed OOF result above the irony baseline was not reproducible
as a uniform three-seed expert. No idiom or code-switching expert was
competitive. Therefore, there is no development-safe basis to replace any
target source in the incumbent; promotion condition 2 cannot be met without a
target decrease. The incumbent is retained and no full-superiority claim is
supported.

Per-run manifests/checkpoints are in
`outputs/final_best_tuned_next_cycle/experts/`; exact OOF reports and the
append-only branch registry are in `next_cycle/oof/` and
`experiment_registry.{csv,json}` respectively.
