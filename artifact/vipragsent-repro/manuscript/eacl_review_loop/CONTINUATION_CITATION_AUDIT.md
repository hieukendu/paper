# Citation-by-Citation Audit: EACL Continuation

Audit date: 2026-07-30
Scope: all in-text citations in `latex/sections/` and both synchronized bibliography files.
Method: enumerate every cited key, compare metadata with the primary archival record, then read each citing sentence in context. `references.bib` and `latex/references.bib` are required to be byte-identical.

## Results

All 13 cited keys occur in both the source and bibliography. No duplicate key, undefined citation, or unused bibliography item was found after the corrected build. The two bibliography files match byte-for-byte after this audit. The key `ghosh-etal-2019-verbal-irony` was renamed to `ghosh-etal-2020-interpreting` to match its verified 2020 ACL Anthology record. The Toukmaji--Flanigan record now includes the ACL DOI.

| Key | Primary verification record | Metadata | In-text use and support | Verdict |
| --- | --- | --- | --- | --- |
| `nguyen-tuan-nguyen-2020-phobert` | ACL Anthology 2020.findings-emnlp.92 | title, two authors, Findings 2020, pp. 1037--1042, DOI verified | Vietnamese monolingual pretrained encoder | PASS |
| `conneau-etal-2020-unsupervised` | ACL Anthology 2020.acl-main.747 | title, ten authors, ACL 2020, pp. 8440--8451, DOI verified | multilingual XLM-R reference | PASS |
| `nguyen-etal-2023-visobert` | ACL Anthology 2023.emnlp-main.315 | title, four authors, EMNLP 2023, pp. 5191--5207, DOI verified | Vietnamese social-media pretrained model | PASS |
| `nguyen-etal-2018-uit-vsfc` | IEEE KSE 2018 DOI 10.1109/KSE.2018.8573337 | title, five authors, KSE 2018, pp. 19--24, DOI verified | student-feedback sentiment resource | PASS |
| `ho-etal-2020-uit-vsmec` | Springer PACLING 2019 proceedings DOI 10.1007/978-981-15-6168-9_27 | title, seven authors, 2020 book chapter, pp. 319--333, DOI verified | Vietnamese social-media emotion resource | PASS |
| `nguyen-etal-2020-efficient` | IOS Press DOI 10.3233/FAIA200579 | title, four authors, FAIA 327, pp. 343--354, DOI verified | published binary Vietnamese review setting only | PASS |
| `dou-etal-2024-sailor` | ACL Anthology 2024.emnlp-demo.45 | title, nine authors, EMNLP Demo 2024, pp. 424--435, DOI verified | open South-East Asian language-model family | PASS |
| `nguyen-etal-2023-vistral` | official Hugging Face `Viet-Mistral/Vistral-7B-Chat` card | model-card title, authors, year, model ID; no venue/pages/DOI supplied | configured Vietnamese instruction-tuned baseline family | PASS WITH MODEL-CARD SCOPE |
| `moore-barnes-2021-multi` | ACL Anthology 2021.naacl-main.227 | title, two authors, NAACL 2021, pp. 2838--2869, DOI verified | auxiliary objectives for targeted sentiment; no causal transfer claim | PASS |
| `rem-lab-2026-vivid` | official ReML-AI repository at pinned commit `471def...` | repository title, organization, year and revision verified; no archival venue/DOI supplied | Vietnamese figurative-language benchmark and provenance source only | PASS WITH REPOSITORY SCOPE |
| `ghosh-etal-2020-interpreting` | ACL Anthology 2020.scil-1.10 | title, three authors, SCiL 2020, pp. 82--93, official URL verified | semantic incongruity/contextual strategies for verbal irony; explicitly not reproduced | PASS |
| `tran-etal-2026-vigoemotions` | ACL Anthology 2026.eacl-long.129 | title, four authors, EACL 2026, pp. 2805--2831, DOI verified | Vietnamese fine-grained emotion benchmark | PASS |
| `toukmaji-flanigan-2025-adapting` | ACL Anthology 2025.gem-1.61 | title, two authors, GEM 2025, pp. 670--704, DOI 10.18653/v1/2025.gem-1.61 verified | comparative low-resource LLM adaptation; no blanket superiority claim | PASS |

## Context check

The citations support positioning and task-family statements, not the paper's 84.3883 result, dataset counts, or experimental claims; those claims instead trace to repository artifacts. The Related Work section explicitly limits all external citations to their published task/model scope. No sentence relies on a citation to establish a claimed final-system result or independent reproduction.

## Sources consulted

- ACL Anthology records for PhoBERT, XLM-R, ViSoBERT, Sailor, Moore--Barnes, Ghosh--Musi--Muresan, ViGoEmotions, and Toukmaji--Flanigan.
- Official DOI landing records for UIT-VSFC, UIT-VSMEC, and the AIVIVN-related proceedings record.
- Official model/repository cards for Vistral and VIVID, used only where no archival paper is supplied in the bibliography.

## Build verification

The post-audit XeLaTeX/BibTeX build completes with no undefined citations, no undefined references, no duplicate labels, and no duplicate BibTeX keys. Final machine-level citation verdict: **PASS**, subject to the stated model-card and repository-source boundaries.
