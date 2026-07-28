# ViPragSent Repeated-OOF Fair Framework

**NOT_PROMOTED**: no genuinely changed target passed repeated OOF robustness; no canonical-test candidate was evaluated.

## Methods actually run

- Complete probability-bank screen: all individual sources, pairs, triples, and leave-one-out ensembles.
- Selective incumbent-positive-preserving rescue using the paired heterogeneous source bank.
- Repeated 5 outer folds × 5 deterministic seeds for every positive screen advanced by successive halving.

## Repeated candidates and exclusions

| Target | Candidate | Median delta | Non-decreasing fold-runs | Exclusion |
| --- | --- | ---: | ---: | --- |
| code_switching | phobert_2+visobert_2+visobert_3 | +0.7438399160 | 60% | missing paired canonical-test probability artifact |
| code_switching | phobert_3+visobert_2+visobert_3 | +1.4380840178 | 84% | missing paired canonical-test probability artifact |
| code_switching | phobert_1+visobert_1+visobert_2 | +0.8954438393 | 64% | missing paired canonical-test probability artifact |
| code_switching | phobert_3+visobert_1 | +1.3793201930 | 64% | missing paired canonical-test probability artifact |
| code_switching | phobert_1+phobert_3+visobert_1+visobert_2+visobert_3 | +1.0515522003 | 68% | missing paired canonical-test probability artifact |
| code_switching | phobert_1+visobert_2+visobert_3 | +0.7356782825 | 60% | fewer than 70% non-decreasing repeated fold-runs |
| code_switching | paired_source_preserving_rescue | -0.0973470195 | 72% | non-positive median repeated OOF delta |

The paired `phobert_1 + visobert_2 + visobert_3` code triple had a positive median delta but reached 60% non-decreasing fold-runs, below the required 70%. Stronger code triples lacked paired canonical-test source probabilities and were retained as excluded evidence, not inferred or fabricated.

All dataset hashes remained unchanged. Protected-label predictions were never recalibrated or changed. Detailed TP/TN/FP/FN corrections, fold variance, seed variance, and disagreement are retained in `repeated_oof_summary.json`.

NOT_PROMOTED
