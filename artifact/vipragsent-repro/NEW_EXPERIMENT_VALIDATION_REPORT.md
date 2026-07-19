# New Experiment Validation Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-19
- Verification Status: VERIFIED AT ARTIFACT LEVEL
- Version Label: new_experiment_validation_v1
- Scope: P0 multi-seed ablation, ViSoBERT baseline, P1 source-stratified sensitivity, P2 low-resource sarcasm, and their newly added external-benchmark predictions.

## Preflight and verdict

Pre-edit `git status --short` recorded only pre-existing/generated manuscript build outputs (`manuscript/latex/$out/`, `build/`, `main.aux`, `main.bbl`, `main.blg`, `main.log`, `main.out`, `main.pdf`) and `.pytest_cache/v/cache/nodeids`; no raw experiment artifact was modified during this validation.

**Verdict: PASS for artifact-level numerical and split-integrity validation.** The experiment outputs are not an independent reproduction: no training workflow was rerun in this validation.

The repository’s result objects comprise 93 per-seed report records: 18 P0 configuration reports, 3 ViSoBERT reports, 42 P1 source-stratum reports, and 30 P2 reports. Their point estimates and deterministic 200-resample bootstrap confidence intervals were recomputed from the prediction JSONL and the designated gold data using the repository evaluator. The 45 additional external-benchmark per-seed metric records were also recomputed.

## Experiment inventory and seed matrices

All seed matrices use `20260520`, `20260521`, and `20260522`.

| Experiment | Verified systems / conditions | Seeds | Result authority |
|---|---|---:|---|
| P0 multi-seed ablation | `vipragsent_full`, `phobert_finetune`, `vipragsent_no_rationale`, `vipragsent_no_emotion`, `vipragsent_no_polarity`, `vipragsent_no_uncertainty` | 3 each | `results/p0_multi_seed_ablation.json` |
| P0 ViSoBERT | `visobert_finetune` | 3 | `results/p0_visobert_baseline.json` |
| P1 source sensitivity | seven systems above plus `visobert_finetune`, evaluated separately on `visobert_local` and `VIVID_seed_and_irony_generation` | 3 each | `results/p1_source_stratified_sensitivity.json` |
| P2 low-resource sarcasm | `phobert_finetune`, `vipragsent_full`; positive budgets 64, 128, 256, 512, 1024 | 3 for every system/budget | `results/p2_multi_seed_low_resource.json` |
| External benchmarks | four ablations and ViSoBERT on `uit_vsfc`, `uit_vsmec`, `aivivn_2019` | 3 each | the P0 result objects above |

The 90 newly introduced prediction files are exactly: 15 P0/ViSoBERT main-test files, 45 external-benchmark files, and 30 P2 files. Six pre-existing `vipragsent_full`/`phobert_finetune` main-test files are additionally reused by P1; hence the wider audit checked 96 prediction files.

## ID-set, duplicate, and seed-copy checks

- 90/90 newly introduced prediction files matched the appropriate gold-ID set exactly.
- 96/96 files in the wider audit matched exactly; every file had the expected row count, no missing IDs, no extra IDs, and no duplicate IDs.
- Main and P2 files matched the 2,000-record adjudicated ViPragSent test split. External files matched their corresponding public-evaluation split: UIT-VSFC (3,166), UIT-VSMEC (693), or AIVIVN-2019 (3,217).
- Canonical `(id, predictions)` fingerprints and raw-byte hashes differed across the three seeds within all 32 evaluated system/condition groups. No copied or identical seed prediction file was detected.
- The fixed train/dev/test split manifest continues to report zero pairwise overlap and zero within-split duplicate IDs. No prediction-side split leakage was found.

## Metric and aggregate verification

Per-seed results were recomputed from raw prediction JSONL with `bootstrap_resamples=200`, the repository’s fixed bootstrap seed (`20260520`), and its macro-F1 definitions. This checked 651 pragmatic metric/CI cells across the 93 core per-seed reports and 45 external metric/CI cells. The recomputed values match the stored result objects.

All stored aggregate means and normal 95% confidence intervals across the three seed values recompute exactly.

| Aggregate scope | Count | Status |
|---|---:|---|
| P0 multi-seed ablation | 6 | PASS |
| ViSoBERT main result | 1 | PASS |
| P1 source strata | 14 | PASS |
| P2 budget × system | 10 | PASS |
| New external benchmark aggregates | 15 | PASS |
| Total present in result objects | 46 | PASS |

The user-provided expected count of 32 aggregates is not the count represented by the repository. The core P0/P1/P2 objects contain 31 aggregates (6 + 1 + 14 + 10); including the 15 external aggregates yields 46. This is a counting-scope discrepancy, not a numerical mismatch.

## Configuration, manifests, and model provenance

- `configs/experiments_p0_p1_p2.yaml` specifies all three seeds, the five P2 budgets, the two P2 systems, and the four ablation systems.
- P2 training subsets contain exactly the asserted 1:3 positive-to-other mix: 64/192, 128/384, 256/768, 512/1536, and 1,024/3,072; total sizes are 256, 512, 1,024, 2,048, and 4,096.
- The P1 test composition is exactly 1,666 `visobert_local` and 334 `VIVID_seed_and_irony_generation` records.
- All 60 copied `answer/run_manifests/**/run_manifest.json` records resolve to an extant raw prediction path. Result → prediction → run-manifest linkage is intact for the retained manifest set.
- ViSoBERT is pinned consistently in `configs/experiments_p0_p1_p2.yaml`, `configs/models.yaml`, `scripts/run_p0_p1_p2_experiments.py`, and `results/p0_visobert_baseline.json` to `uitnlp/visobert` revision `196a62afad9cbe4f52a54aabad828b13f0eec59a`. The result trace records the same resolved revision. The tokenizer is the repository’s SentencePiece configuration at that model revision. The local model snapshot is not present in this Windows checkout, so this is verification of the recorded run provenance, not a fresh local snapshot checksum.

## Checksums and answer/results copies

The following `answer/results/` files byte-match their `results/` counterparts:

| File | SHA-256 |
|---|---|
| `p0_multi_seed_ablation.json` | `63016d0f0fad0a5f43fbdf80cdcb4481ad8ee11c5172f6833ac0f8a9652b8e7a` |
| `p0_visobert_baseline.json` | `50b8df6ab9312daf2b8e5e801d7ca341251a1f4cb1053b58471839666d7ed573` |
| `p1_source_stratified_sensitivity.json` | `c0e9066368ae1d4bc0bfbe9d57b0f4b996e2a8d05326d5fd01abf8aaae0653df` |
| `p2_multi_seed_low_resource.json` | `28afd8e05e4d22310886156877e35a981c1ea86bcd2c111cfbeefe63b7c8c2ee` |
| `p0_p1_p2_summary.json` | `c6a440b31e2ef733e2436fa90838cc6397b6b2ad699c2c7bfa310b308d0d890f` |
| `p0p1p2_runs.json` | `760433b1a67c82ffe2cdc41ba00dda669fdc6a11710e90050e2d24f45b87e86d` |

`answer/reproducibility/verification_manifest.json` validates all of its 62 currently recorded hashes with no mismatch. It predates the new P0/P1/P2 result JSONs, however, and does not yet record their checksums. This is a required Phase 2 traceability update, not a raw-artifact integrity failure.

## Descriptive findings relevant to bounded interpretation

- P0 is now three-seed evidence, but it does not identify a causal or universal auxiliary-task effect. The full configuration mean macro pragmatic F1 is 73.7469, compared with 81.7130 for the no-multitask PhoBERT reference; ablation means range from 73.5623 to 74.3401.
- ViSoBERT is evaluated: its three-seed main-test aggregate is 82.2550 [81.6056, 82.9043].
- P1 is a descriptive split-by-source sensitivity analysis only. Both strata are covered by all three seeds, but the VIVID stratum has 334 records, so its wider uncertainty must remain explicit. It cannot identify a source-domain cause or source superiority.
- P2 is non-monotonic for at least one system/condition and has wide uncertainty in some cells (notably the full system at budget 256). It does not support an unqualified data-efficiency claim.
- External benchmark results are three-seed aggregates for the newly evaluated ablations and ViSoBERT. They are evaluation-only external data, not ViPragSent sources.

## Statistical fallacy scan (11/11)

| Fallacy | Status | Validation note |
|---|---|---|
| Simpson’s paradox | Checked | No aggregate-to-stratum reversal was used as an inference; retain stratum-specific P1 reporting. |
| Ecological fallacy | Checked | No group-level result should be converted to an individual-level claim. |
| Berkson’s paradox | CAUTION | P1 uses fixed source-labelled strata; selection/provenance should be described, not treated as a random population sample. |
| Collider bias | Checked | No covariate adjustment or collider control is reported. |
| Base-rate neglect | Checked | Reported outcomes are macro F1, not predictive values. |
| Regression to the mean | Checked | No pre/post selection design. |
| Survivorship bias | Checked | Fixed gold splits have complete prediction coverage; no prediction attrition detected. |
| Look-elsewhere effect | CAUTION | Multiple systems, tasks, budgets, and strata are evaluated; avoid selective significance language and do not imply confirmatory testing without correction. |
| Garden of forking paths | CAUTION | Multiple analysis slices exist and no preregistration is established; label P1/P2 exploratory or descriptive as appropriate. |
| Correlation is not causation | CAUTION | P1 source comparisons are observational/descriptive; causal wording is unsupported. |
| Reverse causality | Checked | No directional causal relation is identified by these evaluations. |

## Anomalies and required follow-up

1. **Traceability metadata is stale:** Phase 2 must add the six byte-verified result objects and their checksums to the verification manifest, artifact registry/index, and claim ledger. Do not portray their absence from the old manifest as a failed checksum.
2. **Aggregate-count wording:** use the explicit scope (31 core P0/P1/P2 aggregates; 46 including external results), rather than repeating “32” without definition.
3. **Governance remains unresolved:** VIVID authorization/license review is not completed by the P1 analysis. No public-release or unrestricted-use claim is supported.
4. **No independent reproduction:** this report validates committed artifacts; it does not rerun model training or establish an independent reproduction.
