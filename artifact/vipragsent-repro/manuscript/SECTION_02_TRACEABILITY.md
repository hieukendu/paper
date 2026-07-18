# Section 2 Traceability Record

Status: **VERIFIED FOR DRAFTING**.  This record distinguishes the verified scholarly AIVIVN description from unresolved organizer provenance and local-mirror licensing.

| Claim ID | Section claim | Evidence | Permitted scope | Status |
| --- | --- | --- | --- | --- |
| S2-C01 | PhoBERT, XLM-R, and ViSoBERT provide Vietnamese, multilingual, and Vietnamese-social-media model context, respectively. | `references.bib`: `nguyen-tuan-nguyen-2020-phobert`, `conneau-etal-2020-unsupervised`, `nguyen-etal-2023-visobert` | Positioning and baseline context only; not a claim that ViSoBERT was evaluated. | VERIFIED |
| S2-C02 | UIT-VSFC and UIT-VSMEC are external ordinary-task benchmarks, not sources of ViPragSent records. | `references.bib`: `nguyen-etal-2018-uit-vsfc`, `ho-etal-2020-uit-vsmec`; `configs/data_governance.yaml`; `README.md` | External sentiment/emotion evaluation only. | VERIFIED |
| S2-C03 | Nguyen et al. (2020) is a scholarly description of a binary AIVIVN-2019 Vietnamese e-commerce-review setting and records the historical challenge URL. | `references.bib`: `nguyen-etal-2020-efficient`; publisher PDF, DOI `10.3233/FAIA200579` | Scholarly description only; not canonical organizer attribution. | VERIFIED |
| S2-C03a | The local evaluated AIVIVN artifact is a hash-identified Kaggle mirror; its canonical-release identity and license remain unresolved. | `answer/data_provenance/source_registry.json`; `answer/data_provenance/checksums.json` | Mirror provenance and artifact identity only; do not call it an organizer release or canonical split. | VERIFIED_LOCAL_HASH / UNRESOLVED_LICENSE |
| S2-C04 | Auxiliary-task learning is a design family whose effect requires empirical evaluation. | `references.bib`: `moore-barnes-2021-multi`; `answer/results/multitask_ablation.json` | Positioning only; no general benefit or causal claim. | VERIFIED |
