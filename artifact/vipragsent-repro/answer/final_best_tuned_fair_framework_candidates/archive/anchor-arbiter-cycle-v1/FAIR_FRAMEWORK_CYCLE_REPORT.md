# ViPragSent anchor-arbiter cycle

**NOT_PROMOTED** — the canonical test was not accessed.

## Actual nested development-OOF screens

| Target | Candidate | Median Δ F1 | Rescued FN | Introduced FP | Status |
| --- | --- | ---: | ---: | ---: | --- |
| irony | conservative_logistic | +0.000000 | 0 | 0 | rejected |
| irony | conservative_gradient_boosted | +0.013438 | 1 | 1 | rejected |
| irony | nonnegative_probability_stacker | +0.000000 | 0 | 0 | rejected |
| irony | selective_topk_1 | -0.426870 | 1 | 4 | rejected |
| irony | selective_topk_2 | -1.146216 | 1 | 9 | rejected |
| irony | selective_topk_3 | -1.848082 | 1 | 14 | rejected |
| idiom_figurative | conservative_logistic | -0.148039 | 0 | 1 | rejected |
| idiom_figurative | conservative_gradient_boosted | +0.000000 | 0 | 0 | rejected |
| idiom_figurative | nonnegative_probability_stacker | +0.000000 | 0 | 0 | rejected |
| idiom_figurative | selective_topk_1 | -0.422770 | 1 | 4 | rejected |
| idiom_figurative | selective_topk_2 | -0.835035 | 2 | 8 | rejected |
| idiom_figurative | selective_topk_3 | -0.935004 | 4 | 11 | rejected |
| code_switching | conservative_logistic | +0.104472 | 1 | 1 | rejected |
| code_switching | conservative_gradient_boosted | +0.101120 | 2 | 3 | rejected |
| code_switching | nonnegative_probability_stacker | -0.107055 | 0 | 1 | rejected |
| code_switching | selective_topk_1 | +0.416680 | 3 | 2 | rejected |
| code_switching | selective_topk_2 | +0.198869 | 4 | 6 | rejected |
| code_switching | selective_topk_3 | -0.014175 | 5 | 10 | rejected |

The requested Sailor irony anchor, Vistral code anchor, and XLM-R idiom anchor cannot be completed reproducibly from the present checkout: their adapter/checkpoint manifests lack paired canonical-train OOF and label-free development probability artifacts. Existing canonical-test predictions were deliberately not read. The restored `phobert_3_reproduced+visobert_2+visobert_3` code candidate is preserved.

NOT_PROMOTED
