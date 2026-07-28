# ViPragSent Fair-Framework Cycle v3

**NOT_PROMOTED** — canonical test was not consumed.

## Outcome

The strongest development candidate was retained rather than discarded. It cannot be frozen because exact registered checkpoint files needed for label-free paired inference are absent; legacy probability files lack verifiable checkpoint provenance and were not substituted.

## Development selection

| Target | Selected candidate | Median repeated delta | Bootstrap P(delta > 0) | Status |
| --- | --- | ---: | ---: | --- |
| irony | incumbent_unchanged | +0.0000000000 | 0.0000 | no development candidate satisfied all eligibility rules |
| idiom_figurative | incumbent_unchanged | +0.0000000000 | 0.0000 | no development candidate satisfied all eligibility rules |
| code_switching | phobert_3+visobert_2+visobert_3 | +1.7648040740 | 0.9712 | advanced; inference blocked |

Raw baseline maxima were recomputed from raw baseline predictions before screening. Dataset hashes match before and after. Protected labels were not recalibrated or changed; no frozen manifest was created, and `final_best_tuned` was not modified.

Required artifact recovery: restore the exact checkpoint(s) listed in `probability_artifact_registry.json` (matching their recorded SHA-256), then run a new cycle from the frozen development selection. Do not reuse legacy unverified probability files.

NOT_PROMOTED
