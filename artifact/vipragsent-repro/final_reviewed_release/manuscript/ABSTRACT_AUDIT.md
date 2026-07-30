# Title and Abstract Draft

## Title candidates

1. **ViPragSent: A Comparative Evaluation of Vietnamese Social-Media Pragmatics and Multi-Task Trade-offs**
2. **Evaluating Vietnamese Social-Media Pragmatics Across Encoders, 7B SFT Models, and Prompted APIs**
3. **Vietnamese Social-Media Pragmatics: Comparative Evaluation and Multi-Task Trade-off Analysis**

## Recommended title

**ViPragSent: Comparative Evaluation of Vietnamese Pragmatic Sentiment with a Social-Media Source Component**

The recommended title identifies the setting, the paper's evaluation orientation, and the qualified trade-off analysis without suggesting that ViPragSent is the leading system.

## Abstract

Pragmatic interpretation in Vietnamese-language text can depend on signals that extend beyond surface polarity. Such phenomena can shift apparent polarity or communicative intent, making a single polarity label insufficient for the defined evaluation considered here. We present ViPragSent, an adjudicated evaluation setting combining a documented social-media source component with author-created, context-augmented figurative-language derivatives. It covers six pragmatic phenomena without claiming comprehensive coverage of Vietnamese pragmatics. We compare Vietnamese, multilingual, and Vietnamese-social-media encoders, 7B QLoRA SFT models, prompted API baselines, and a multi-task encoder under a common protocol. Vistral-7B SFT is the strongest recorded evaluated system, with macro pragmatic F1 of 82.83. ViPragSent is not the leading recorded system; its three-seed ablation instead records configuration-level pragmatic-performance differences. Source-stratified and low-resource analyses remain descriptive or exploratory. Reported comparisons are traceable to prediction files, run manifests, result hashes, and pinned external model archives. Source text remains private: the manuscript reports aggregate results and provenance, not a public raw-text release.

## Keywords

Vietnamese NLP; social-media pragmatics; pragmatic sentiment analysis; comparative evaluation; multi-task learning; reproducibility

## Abstract-to-paper consistency audit

| Sentence | Supporting manuscript section(s) | Evidence status |
| --- | --- | --- |
| Pragmatic interpretation in Vietnamese social-media text can depend on signals beyond surface polarity. | Section 1, paragraphs 1--2 | Framing statement bounded to the paper's setting; no prevalence or universal claim. |
| Such phenomena can shift apparent polarity or communicative intent, making a single polarity label insufficient for the narrowly defined evaluation. | Section 1, paragraphs 1--2; Section 3.1 | Framing statement for the defined task only. |
| We present ViPragSent as an adjudicated evaluation setting for the six named phenomena. | Section 3.1; Table 1 | VERIFIED: label definitions and adjudicated task inventory. |
| The setting makes these distinctions measurable without claiming comprehensive coverage of Vietnamese pragmatics. | Sections 1--3 | Bounded task-construction statement; no general-coverage claim. |
| We compare encoder families including ViSoBERT, 7B SFT models, prompted APIs, and a multi-task encoder under a common protocol. | Sections 1 and 4.1--4.2 | VERIFIED: recorded comparison boundary and protocol. |
| Vistral-7B SFT is the strongest recorded evaluated system with macro pragmatic F1 of 82.83. | Section 5.1; Table 2 | VERIFIED: `answer/results/main_pragmatic.json`; rounded display value. |
| ViPragSent is not the leading recorded system; P0 records three-seed configuration-level pragmatic differences. | Sections 5.1--5.2 | VERIFIED, QUALIFIED: no causal, robust, or universal interpretation. |
| Reported comparisons are traceable to predictions, manifests, result hashes, and pinned external archives. | Section 4.3 | VERIFIED: artifact-level traceability only, not an independent rerun. |
| Source text remains private and non-commercial; only aggregate results and provenance are reported. | Sections 3.2, 8, and 9 | VERIFIED: no public raw-text release claim. |

Audit result: **PASS**. The abstract contains no unsupported superiority, causal, data-efficiency, independent-reproduction, or public-data-release claim.
