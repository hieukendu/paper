# FINAL_BEST_TUNED ARS Validation

## Material Passport

- Origin Skill: `experiment-agent`
- Origin Mode: `validate`
- Origin Date: 2026-07-30
- Verification Status: ANALYZED
- Version Label: `final_best_tuned_validation_v1`

## Scope and Evidence

This validation inspected the promoted `answer/final_best_tuned/` bundle, its archived incumbent at `archive/38fa5fc/`, canonical test labels at `data/processed/vipragsent_test.jsonl`, the final-selection configuration, and the relevant frozen-selection scripts. No training rerun was performed.

## Numerical Validation

| Check | Result |
|---|---|
| JSON/CSV comparison agreement | PASS: both contain eight systems in the same order. |
| Final label metrics recomputed from saved test predictions | PASS: implicit 67.9829; sarcasm 80.2781; irony 97.0177; idiom/figurative 97.1820; code-switching 81.8164; mocking 82.0528. |
| Final macro pragmatic F1 | PASS: mean of the six binary macro-F1 values is 84.3883. |
| Strongest complete baseline | PASS: Vistral-7B best-tuned is 82.8250; aggregate difference is +1.5633. |
| Per-label leader gaps | PASS: +7.1359 implicit sentiment, +0.2463 sarcasm, -0.3955 irony, -0.1138 idiom/figurative, -0.1294 code-switching, and +0.0726 mocking. |
| Canonical test ID coverage | PASS: 2,000 unique prediction IDs exactly match the 2,000 canonical test IDs. |
| Metric definition | PASS: stored and recomputed values use binary macro-F1 per pragmatic label, then the equal-weight mean over six labels. |

## Selection and Provenance Validation

- The stored selection specifies five folds and the objective `mean target label F1 - 0.25 * standard deviation across folds`.
- `final_selection.json` and `configs/final_config.yaml` state that test labels were not used for selection. The reviewed frozen-selection scripts operate on development labels for source/threshold fitting and apply frozen configuration to test predictions.
- Final sources are label-wise: ViSoBERT for implicit sentiment and sarcasm, a 0.6 ViSoBERT / 0.4 PhoBERT probability blend for irony and code-switching, uniform three-seed ViSoBERT for idiom/figurative language, and the targeted three-seed mocking component for mocking.
- The final threshold file records a full-development refit after five-fold OOF source selection. Its threshold payload is intentionally wrapped in metadata, rather than structurally identical to the `final_selection.json` threshold object; the six numerical threshold values agree.
- All 53 files recorded in `artifact_index.json` match their listed SHA-256 values after CRLF-to-LF normalization. The local checkout has `core.autocrlf=true`; the initial byte-level mismatch is therefore a line-ending transport effect, not a content discrepancy.
- Nine checkpoint paths and their recorded SHA-256 values are listed, but no checkpoint file is present locally. The checkpoint hashes are traceable metadata, not independently recomputed hashes.

## Statistical Fallacy Scan

| Fallacy class | Status | Evidence-bound finding |
|---|---|---|
| Simpson's paradox | CHECKED | No aggregate-to-stratum reversal was used to support the final aggregate claim. |
| Ecological fallacy | CHECKED | No group-level result is used to make an individual-level inference. |
| Berkson's paradox | CHECKED | No selected-population correlation analysis is presented. |
| Collider bias | CHECKED | No adjusted causal model is used in the promoted comparison. |
| Base-rate neglect | CHECKED | The report uses binary macro-F1 and does not recast it as prevalence-free predictive value. |
| Regression to the mean | CHECKED | No pre/post extreme-group inference is made. |
| Survivorship bias | CHECKED | Canonical test ID equality is verified; no completion-only analysis is used. |
| Look-elsewhere effect | CHECKED | Label-wise source/threshold search occurred; the manuscript must describe development-only selection and avoid post-hoc per-label dominance claims. |
| Garden of forking paths | CHECKED | Candidate selection is documented, but its development-selected ensemble complexity remains a limitation. |
| Correlation implies causation | CHECKED | No causal explanation for the ensemble gain is supported. |
| Reverse causality | CHECKED | No directional causal inference is made. |

Fallacy-scan coverage: **11/11 checked**.

## Leakage and Reproducibility Status

- Leakage status: **No test-label use for selection identified in the inspected configuration and selection code; saved final predictions were evaluated against the canonical test labels after freeze.**
- Multiple-comparison status: **Qualified.** Per-label source and threshold search is development-only, but the resulting label-wise mixture must be described as development-selected and not as a causal or universally superior architecture.
- Reproducibility status: **ANALYZED.** Stored predictions, IDs, metrics, configuration, and normalized artifact hashes were validated. No training rerun was performed, and locally absent checkpoints preclude a `VERIFIED` reproduction claim.

## Validation Verdict

**PASS for artifact-level numerical and provenance validation, with the mandatory qualification that this is not an independent training reproduction.**
