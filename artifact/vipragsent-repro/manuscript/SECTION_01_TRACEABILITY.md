# Section 1 Traceability Record

Status: **VERIFIED FOR DRAFTING AND RESULTS-CONSISTENT**.  The Introduction contains the exact locked research question and only C1--C5.

| Claim ID | Introduction claim | Evidence | Permitted scope | Status |
| --- | --- | --- | --- | --- |
| I-C01 | Vietnamese social-media model context motivates a bounded evaluation setting. | `references.bib`: `nguyen-etal-2023-visobert`; Section 2 | Context only; not evidence that ViSoBERT was evaluated. | VERIFIED |
| I-C02 | The task operationalizes six pragmatic targets while polarity and emotion remain auxiliaries. | `configs/labels.yaml`; `answer/data_provenance/gold_build_report.json`; Section 3 | Defined label construction only; not general coverage of pragmatics. | VERIFIED |
| C1 | The study evaluates six phenomena in 12,000 adjudicated Vietnamese-language records: 10,000 retained local social-media export records and 2,000 VIVID-identified replacement candidates, with polarity and emotion annotations. | `answer/data_provenance/gold_build_report.json`; `answer/data_provenance/source_registry.json`; `answer/results/annotation_agreement.json`; Section 3 | Do not call all records social-media or claim a public raw-text release. | VERIFIED, QUALIFIED |
| C2 | The comparison covers the four approved system groups with qualified seed/run scope. | `answer/results/main_pragmatic.json`; `answer/run_manifests/`; Section 4 | Comparative evaluation only; prompted APIs are single recorded runs. | VERIFIED |
| C3 | Vistral-7B SFT is the strongest recorded evaluated system at 82.83 macro pragmatic F1; ViPragSent records 73.75. | `answer/results/main_pragmatic.json`; `answer/tables/main_pragmatic.md`; Section 5 | Rounded display values only; no universal superiority claim. | VERIFIED |
| C4 | The ablation is a single-seed observed pragmatic-versus-external-ordinary-task trade-off. | `answer/results/multitask_ablation.json`; `answer/tables/multitask_ablation.md`; Section 5 | No causal, robust, or general effect claim. | VERIFIED |
| C5 | Reported comparisons link to predictions, manifests, hashes, and pinned archives. | `answer/results/claim_ledger.csv`; `answer/reproducibility/verification_manifest.json`; `answer/reproducibility/artifact_registry.json`; Section 4 | Artifact-level traceability only; not a fresh rerun. | VERIFIED |
