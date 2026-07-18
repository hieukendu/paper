# Title and Abstract Draft

## Title candidates

1. **ViPragSent: A Comparative Evaluation of Vietnamese Social-Media Pragmatics and Multi-Task Trade-offs**
2. **Evaluating Vietnamese Social-Media Pragmatics Across Encoders, 7B SFT Models, and Prompted APIs**
3. **Vietnamese Social-Media Pragmatics: Comparative Evaluation and Multi-Task Trade-off Analysis**

## Recommended title

**ViPragSent: A Comparative Evaluation of Vietnamese Social-Media Pragmatics and Multi-Task Trade-offs**

The recommended title identifies the setting, the paper's evaluation orientation, and the qualified trade-off analysis without suggesting that ViPragSent is the leading system.

## Abstract

Pragmatic interpretation in Vietnamese social-media text can depend on signals that extend beyond surface polarity in everyday online interaction. Such phenomena can shift apparent polarity or communicative intent, making a single polarity label insufficient for the narrowly defined evaluation considered here. We present ViPragSent, an adjudicated evaluation setting for six pragmatic phenomena: implicit sentiment, sarcasm, irony, idiom/figurative language, code-switching, and mocking. The setting makes these distinctions measurable without claiming comprehensive coverage of Vietnamese pragmatics. We compare Vietnamese and multilingual encoders, 7B QLoRA SFT models, prompted API baselines, and a multi-task encoder under a common protocol. Vistral-7B SFT is the strongest recorded evaluated system, with macro pragmatic F1 of 82.83. ViPragSent is not the leading recorded system; instead, its single-seed ablation records an observed trade-off between pragmatic detection and average ordinary-task performance relative to its no-multitask reference. Reported comparisons are traceable to prediction files, run manifests, result hashes, and pinned external model archives. Source text remains private, non-commercial research material: the manuscript reports aggregate results and provenance, not a public raw-text release.

## Keywords

Vietnamese NLP; social-media pragmatics; pragmatic sentiment analysis; comparative evaluation; multi-task learning; reproducibility

## Abstract-to-paper consistency audit

| Sentence | Supporting manuscript section(s) | Evidence status |
| --- | --- | --- |
| Pragmatic interpretation in Vietnamese social-media text can depend on signals beyond surface polarity. | Section 1, paragraphs 1--2 | Framing statement bounded to the paper's setting; no prevalence or universal claim. |
| Such phenomena can shift apparent polarity or communicative intent, making a single polarity label insufficient for the narrowly defined evaluation. | Section 1, paragraphs 1--2; Section 3.1 | Framing statement for the defined task only. |
| We present ViPragSent as an adjudicated evaluation setting for the six named phenomena. | Section 3.1; Table 1 | VERIFIED: label definitions and adjudicated task inventory. |
| The setting makes these distinctions measurable without claiming comprehensive coverage of Vietnamese pragmatics. | Sections 1--3 | Bounded task-construction statement; no general-coverage claim. |
| We compare the four named system families under a common protocol. | Sections 1 and 4.1--4.2 | VERIFIED: approved comparison boundary and recorded protocol. |
| Vistral-7B SFT is the strongest recorded evaluated system with macro pragmatic F1 of 82.83. | Section 5.1; Table 2 | VERIFIED: `answer/results/main_pragmatic.json`; rounded display value. |
| ViPragSent is not the leading recorded system; its ablation records a single-seed observed trade-off relative to the no-multitask reference. | Sections 5.1--5.2 | VERIFIED, QUALIFIED: no causal, robust, or universal interpretation. |
| Reported comparisons are traceable to predictions, manifests, result hashes, and pinned external archives. | Section 4.3 | VERIFIED: artifact-level traceability only, not an independent rerun. |
| Source text remains private and non-commercial; only aggregate results and provenance are reported. | Sections 3.2, 8, and 9 | VERIFIED: no public raw-text release claim. |

Audit result: **PASS**. The abstract contains no low-resource result, citation, unsupported superiority claim, causal claim, or public-data-release claim.
