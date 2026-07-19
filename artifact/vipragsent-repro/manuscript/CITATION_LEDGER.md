# Citation Ledger

Verification date: 2026-07-18.  This ledger records source metadata and citation readiness only; it does not authorize new empirical claims.

## Required-citation coverage

| Required citation in `DETAILED_OUTLINE.md` | Citation key | Planned sections | Status | Basis / remaining condition |
| --- | --- | --- | --- | --- |
| Nguyen et al. (2023), ViSoBERT | `nguyen-etal-2023-visobert` | 1, 2 | VERIFIED | Existing ACL Anthology record retained. |
| ReML-AI (2026), VIVID repository | `rem-lab-2026-vivid` | 3 | VERIFIED | Official repository; data-source provenance only. |
| Ghosh et al. (2019), verbal irony | `ghosh-etal-2019-verbal-irony` | 3 | VERIFIED | Conceptual context for contextualization; not a data-source claim. |
| Nguyen and Tuan Nguyen (2020), PhoBERT | `nguyen-tuan-nguyen-2020-phobert` | 2, 4 | VERIFIED | Existing ACL Anthology record retained. |
| Conneau et al. (2020), XLM-R | `conneau-etal-2020-unsupervised` | 2, 4 | VERIFIED | Existing ACL Anthology record retained. |
| UIT-VSFC primary paper | `nguyen-etal-2018-uit-vsfc` | 2, 5 | VERIFIED | Primary IEEE conference paper. |
| UIT-VSMEC primary paper | `ho-etal-2020-uit-vsmec` | 2, 5 | VERIFIED | Primary Springer proceedings chapter. |
| AIVIVN scholarly dataset description | `nguyen-etal-2020-efficient` | 2, 5 | VERIFIED | Peer-reviewed proceedings paper verified at the publisher/DOI. It supports only the published AIVIVN-2019 description and must not be treated as an organizer-authored canonical release record. |
| Sailor-7B-Chat source/model card | `dou-etal-2024-sailor` | 4 | PARTIAL | The primary Sailor paper is verified. The configured Chat ID and revision `19066fae0a8a3ba029c190d8e3dacccf4d1234b8` are documented as a user-attested 2026-07-14 pre-fine-tuning download; archived runs retain local paths rather than an independently logged remote revision. |
| Vistral-7B-Chat source/model card | `nguyen-etal-2023-vistral` | 4 | PARTIAL | The official exact model card is verified. Revision `d331b64e61b935cc43c2b3010ae9fb4fde599b45` is documented as a user-attested 2026-07-14 pre-fine-tuning download; the archived run manifest does not independently log it, and no archival paper/DOI is supplied. |
| GPT-4.1-mini documentation/version source | `openai-2025-gpt-4-1-mini` | 4 | VERIFIED | Official documentation names snapshot `gpt-4.1-mini-2025-04-14`, matching recorded API responses. |
| Multi-task sentiment/pragmatics primary source | `moore-barnes-2021-multi` | 2 | VERIFIED | Primary NAACL paper directly studies auxiliary linguistic/pragmatic tasks for targeted sentiment classification. |

## Verified and partial records

| Citation key | Source type | Official URL | DOI | Sections where it may be used | Bounded claim supported | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `nguyen-etal-2023-visobert` | Primary peer-reviewed conference paper | https://aclanthology.org/2023.emnlp-main.315/ | `10.18653/v1/2023.emnlp-main.315` | 1, 2 | Vietnamese social-media language-model baseline/context. | VERIFIED |
| `nguyen-tuan-nguyen-2020-phobert` | Primary peer-reviewed conference paper | https://aclanthology.org/2020.findings-emnlp.92/ | `10.18653/v1/2020.findings-emnlp.92` | 2, 4 | Vietnamese pretrained encoder used as an evaluated baseline/backbone family. | VERIFIED |
| `conneau-etal-2020-unsupervised` | Primary peer-reviewed conference paper | https://aclanthology.org/2020.acl-main.747/ | `10.18653/v1/2020.acl-main.747` | 2, 4 | XLM-R multilingual pretrained encoder baseline family. | VERIFIED |
| `nguyen-etal-2018-uit-vsfc` | Primary peer-reviewed IEEE conference paper | https://doi.org/10.1109/KSE.2018.8573337 | `10.1109/KSE.2018.8573337` | 2, 5 | Provenance and task context for the ordinary Vietnamese sentiment benchmark. | VERIFIED |
| `ho-etal-2020-uit-vsmec` | Primary peer-reviewed Springer proceedings chapter | https://doi.org/10.1007/978-981-15-6168-9_27 | `10.1007/978-981-15-6168-9_27` | 2, 5 | Provenance and task context for the Vietnamese social-media emotion benchmark. | VERIFIED |
| `nguyen-etal-2020-efficient` | Peer-reviewed conference-proceedings paper | https://journals.sagepub.com/doi/full/10.3233/FAIA200579 | `10.3233/FAIA200579` | 2, 5 | Published description of a binary Vietnamese e-commerce-review dataset used in AIVIVN 2019, including the historical challenge URL recorded in the paper. It does not establish the identity, license, or canonical split of the local Kaggle mirror. | VERIFIED |
| `dou-etal-2024-sailor` | Primary peer-reviewed system-demonstration paper; official model card corroborates Chat ID | https://aclanthology.org/2024.emnlp-demo.45/ | `10.18653/v1/2024.emnlp-demo.45` | 4 | Sailor model family and configured Chat identifier `sail/Sailor-7B-Chat`; revision `19066fae0a8a3ba029c190d8e3dacccf4d1234b8` is a user-attested 2026-07-14 download-time pin, not an independently logged run-manifest field. | PARTIAL |
| `nguyen-etal-2023-vistral` | Official model card | https://huggingface.co/Viet-Mistral/Vistral-7B-Chat | N/A (no DOI supplied) | 4 | Configured identifier `Viet-Mistral/Vistral-7B-Chat`; revision `d331b64e61b935cc43c2b3010ae9fb4fde599b45` is a user-attested 2026-07-14 download-time pin, not an independently logged run-manifest field. | PARTIAL |
| `openai-2025-gpt-4-1-mini` | Official API documentation | https://developers.openai.com/api/docs/models/gpt-4.1-mini | N/A (documentation; no DOI) | 4 | API baseline model snapshot `gpt-4.1-mini-2025-04-14`, matching recorded raw responses. | VERIFIED |
| `moore-barnes-2021-multi` | Primary peer-reviewed conference paper | https://aclanthology.org/2021.naacl-main.227/ | `10.18653/v1/2021.naacl-main.227` | 2 | Auxiliary linguistic/pragmatic tasks can be jointly modeled with targeted sentiment and evaluated as a design choice; it does not establish a ViPragSent improvement. | VERIFIED |

## Unresolved record

| Required source | Source type sought | Official URL | DOI | Planned sections | What it would support | Citation status / resolution path |
| --- | --- | --- | --- | --- | --- | --- |
| AIVIVN 2019 organizer-authored challenge/dataset record | Organizer-authored challenge or dataset record | Historical URL recorded by Nguyen et al. (2020): https://www.aivivn.com/contests/1 | Unknown | Provenance only; not required for Section 2 prose | Canonical organizer attribution, canonical split, and license only. | UNRESOLVED — obtain an organizer page export, a stable archive showing organizer/date, or organizer-provided archival documentation; then create a record. |

## AIVIVN mirror provenance (separate from citation availability)

| Field | Status | Evidence / permitted interpretation |
| --- | --- | --- |
| `scholarly_citation_status` | VERIFIED | Nguyen et al. (2020), DOI `10.3233/FAIA200579`, is a publisher-hosted proceedings record. It may be cited for the paper's own AIVIVN-2019 dataset description, not as an organizer record. |
| `provenance_status` | VERIFIED_BY_SECONDARY_SOURCES | Repository source registry identifies the used artifact as `Kaggle mcocoz/aivivn-2019`; it is described as an AIVIVN-2019 mirror, not an organizer-authored canonical release. |
| `historical_organizer_url_status` | VERIFIED_IN_SCHOLARLY_SOURCE | The proceedings PDF records `https://www.aivivn.com/contests/1` as a historical URL. This verifies that the URL was cited by the paper, not that the page is live or organizer-controlled today. |
| `original_page_status` | UNAVAILABLE_DURING_VERIFICATION | The reported organizer challenge page was not retrievable during verification; this observation is date-bound rather than a claim that the page never existed. |
| `exact_artifact_status` | VERIFIED_BY_LOCAL_HASH | `data/raw/aivivn_2019/train.csv` is `sha256:1f77af867fc62eb87981cf5f671ed68deb43693d6712c593fd61a19742ebe6c3`; `test.csv` is `sha256:b073a16dbe9f4fa25b08e7c03d43b236da5e9d217ef28ec385e75b63b0053215`. |
| `task_and_split_status` | VERIFIED_BY_LOCAL_MANIFEST | The evaluated mirror is binary negative/positive sentiment with source train/test; dev is deterministically sampled from train. The scholarly paper also describes a binary setting, but the local mirror must not be described as the paper's or challenge's canonical split without further evidence. |
| `official_test_labels_status` | UNVERIFIED_FOR_LOCAL_MIRROR | Nguyen et al. (2020) state that their source test labels were unavailable and that their team labeled its test data. This does not establish the origin or status of labels in the local mirror. |
| `license_status` | UNRESOLVED | Local registry records `manual_review_required`. |
| `redistribution_status` | DISABLED | It is evaluation-only; repository notices prohibit relicensing or redistributing third-party raw data. |

## Archived QLoRA run provenance (separate from model citation availability)

| System | Archive-level status | Verified evidence | Historical gaps | Paper-safe interpretation |
| --- | --- | --- | --- | --- |
| Sailor-7B-Chat QLoRA | VERIFIED_AT_PINNED_ARCHIVE_REVISION | Pinned archive revision; PEFT configuration; manifests for seeds 20260520/21/22; three adapter LFS SHA-256 identifiers; user-attested base revision `19066fae0a8a3ba029c190d8e3dacccf4d1234b8`. | The archived run manifest does not independently log the remote revision, `transformers` version, or frozen runtime. | The adapter weights and per-seed run records are traceable; do not claim exact historical environment reproducibility. |
| Vistral-7B-Chat QLoRA | VERIFIED_AT_PINNED_ARCHIVE_REVISION | Pinned archive revision; PEFT configuration; manifests for seeds 20260520/21/22; three adapter LFS SHA-256 identifiers; user-attested base revision `d331b64e61b935cc43c2b3010ae9fb4fde599b45`. | The archived run manifest does not independently log the remote revision, `transformers` version, or frozen runtime. | The adapter weights and per-seed run records are traceable; do not claim exact historical environment reproducibility. |

## Provenance and anonymity notes

- The repository records user-attested Sailor/Vistral Chat download-time pins for 2026-07-14. The three Hugging Face archives independently verify the historical adapters and run manifests, while status remains PARTIAL because the archived manifests do not independently log those remote revisions or a frozen software environment.
- The adapter archive URLs in `configs/artifact_registry.json` identify the account owner and must remain out of an anonymous manuscript. They can be used in an internal reproducibility record or replaced by an anonymous archive after a de-anonymization decision.
- Missing venue, pages, and DOI for the Vistral card and the OpenAI documentation are recorded as not applicable rather than inferred.
