# ViPragSent Fair-Framework Cycle (Corrected Protocol)

**NOT_PROMOTED**. One frozen canonical-test evaluation was performed after all target choices were selected using same-split nested development OOF only.

## Development selection

| Target | Selected method | OOF delta vs incumbent |
| --- | --- | ---: |
| irony | incumbent_unchanged | +0.0000000000 |
| idiom_figurative | incumbent_unchanged | +0.0000000000 |
| code_switching | incumbent_unchanged | +0.0000000000 |

Protected-label binary predictions were copied unchanged from the incumbent. The probability bank tested individual sources, all source pairs/triples, target experts, a preserving rescue, a separately reported non-negative logistic stacker, and a true softmax mixture-of-experts gate. Scalers were fit within each training fold; threshold plateaus used the prescribed 0.02 F1-point window.

No target candidate improved the incumbent in at least 3 of 5 outer folds, so the selected target predictions have zero TP/FP/FN changes and zero selection-induced fold/seed variance. Detailed fold scores, candidate confusion deltas, and all negative results are retained in `rejected/best_rejected_candidate.json`.

## Frozen test gate

| Metric | Candidate | Baseline max | Margin |
| --- | ---: | ---: | ---: |
| implicit_sentiment | 67.9828834091 | 60.8469599667 | +7.1359234423 |
| sarcasm | 80.2780593410 | 80.0317999369 | +0.2462594041 |
| irony | 97.0177342786 | 97.4131500651 | -0.3954157865 |
| idiom_figurative | 97.1819960861 | 97.2957617951 | -0.1137657090 |
| code_switching | 81.8164481314 | 81.9458209143 | -0.1293727829 |
| mocking | 82.0528332325 | 81.9802025151 | +0.0726307174 |
| macro_pragmatic_f1 | 84.3883257464 | 82.8250207520 | +1.5633049945 |

Raw baseline predictions were recomputed and verified against four-decimal display registry values before use. Dataset hashes are identical before and after.

NOT_PROMOTED
