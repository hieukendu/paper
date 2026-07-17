# Calibration of Pragmatic-polarity Confidence

Status: `complete` · target: pragmatic polarity · 10 equal-width confidence bins.

| System | Test records | ECE ↓ | Missing confidence | status |
| --- | ---: | ---: | ---: | --- |
| PhoBERT fine-tune | 2,000 | 0.089697 | 0 | complete |
| ViPragSent (ours) | 2,000 | **0.067116** | 0 | complete |
| XLM-R-large | 2,000 | 0.145464 | 0 | complete |

The following systems are excluded rather than assigned an ECE of zero because their prediction records do not contain pragmatic-polarity confidence scores: GPT-4.1-mini zero-shot, GPT-4.1-mini 8-shot, Sailor-7B SFT, and Vistral-7B SFT (2,000 missing confidence values each). See `../figures/fig7_calibration.svg` for reliability curves.

Source: `results/calibration.json`.
