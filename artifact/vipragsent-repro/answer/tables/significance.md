# Paired Bootstrap Significance Against ViPragSent

Status: `complete` · reference: ViPragSent (ours) · 1,000 paired non-parametric bootstrap resamples over 2,000 test records.

`Δ F1` is challenger minus ViPragSent in percentage points. Reported p-values are raw two-sided values; apply Holm correction when selecting a final hypothesis family. A displayed `0.000` means no bootstrap draw crossed zero at the recorded precision, not a literal probability of zero.

| Challenger | Seed | Δ F1 | 95% paired bootstrap CI | Raw p |
| --- | ---: | ---: | ---: | ---: |
| GPT-4.1-mini 8-shot | 20260520 | -4.4923 | [-5.6920, -3.3581] | 0.000 |
| GPT-4.1-mini zero-shot | 20260520 | -13.0504 | [-14.3446, -11.7727] | 0.000 |
| PhoBERT fine-tune | 20260520 | 7.9995 | [7.1033, 8.8370] | 0.000 |
| PhoBERT fine-tune | 20260521 | 7.7394 | [6.7638, 8.6954] | 0.000 |
| PhoBERT fine-tune | 20260522 | 8.1594 | [7.1736, 9.1800] | 0.000 |
| Sailor-7B SFT | 20260520 | 8.6266 | [7.7742, 9.5324] | 0.000 |
| Sailor-7B SFT | 20260521 | 8.1732 | [7.3222, 9.1033] | 0.000 |
| Sailor-7B SFT | 20260522 | 8.5349 | [7.6300, 9.4059] | 0.000 |
| ViPragSent without emotion auxiliary | 20260520 | 1.0950 | [0.4974, 1.7138] | 0.000 |
| ViPragSent without polarity auxiliary | 20260520 | -0.3603 | [-0.6682, -0.0781] | 0.008 |
| ViPragSent without rationale auxiliary | 20260520 | -0.0747 | [-0.2784, 0.1238] | 0.450 |
| ViPragSent without uncertainty weighting | 20260520 | -0.0963 | [-0.2962, 0.0875] | 0.332 |
| Vistral-7B SFT | 20260520 | 9.2280 | [8.3030, 10.2217] | 0.000 |
| Vistral-7B SFT | 20260521 | 9.1111 | [8.2749, 9.9917] | 0.000 |
| Vistral-7B SFT | 20260522 | 8.8952 | [7.8931, 9.8578] | 0.000 |
| XLM-R-large | 20260520 | 6.2695 | [5.2123, 7.1802] | 0.000 |
| XLM-R-large | 20260521 | 6.5058 | [5.4954, 7.5367] | 0.000 |
| XLM-R-large | 20260522 | 8.4333 | [7.5688, 9.2795] | 0.000 |

The local heuristic smoke baseline is omitted because its prediction file is incomplete (1,992 gold IDs missing).

Source: `results/significance.json`.
