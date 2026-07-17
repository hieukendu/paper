# Main Encoder Learning Curves

Status: `complete` · metric: development macro pragmatic F1.

| System | Seed | Completed epochs | Peak dev F1 | Final dev F1 |
| --- | ---: | ---: | ---: | ---: |
| PhoBERT fine-tune | 20260520 | 5 | 81.47 | 81.07 |
| PhoBERT fine-tune | 20260521 | 9 | 81.74 | 81.58 |
| PhoBERT fine-tune | 20260522 | 9 | 81.77 | 81.55 |
| ViPragSent (ours) | 20260520 | 5 | 73.45 | 73.20 |
| ViPragSent (ours) | 20260521 | 6 | 73.35 | 73.27 |
| ViPragSent (ours) | 20260522 | 8 | 73.54 | 73.46 |
| XLM-R-large | 20260520 | 10 | 79.93 | 79.93 |
| XLM-R-large | 20260521 | 10 | 79.64 | 79.64 |
| XLM-R-large | 20260522 | 10 | 81.22 | 81.11 |

`fig6_learning_curves.svg` plots the mean available development F1 at each epoch for these three main encoder systems. The number of contributing seeds decreases after a run stops early. Sailor-7B and Vistral-7B histories record per-epoch training loss but not a comparable per-epoch development macro-F1, so they are not plotted here.

Source: `results/learning_curves.json`.
