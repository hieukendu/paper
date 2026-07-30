# ViPragSent Provenance Consistency Audit

## Authoritative composition

The final gold corpus is 12,000 records: 10,000 retained local ViSoBERT-export records plus 2,000 author-created, context-augmented VIVID-derived replacement records. The latter replaced earlier ViSoBERT rows, which are not in the final partitions.

| Split | ViSoBERT-export | VIVID-derived | Total |
| --- | ---: | ---: | ---: |
| Train | 6,667 | 1,333 | 8,000 |
| Development | 1,667 | 333 | 2,000 |
| Test | 1,666 | 334 | 2,000 |
| Total | 10,000 | 2,000 | 12,000 |

The arithmetic is verified: 6,667+1,667+1,666=10,000; 1,333+333+334=2,000; and 8,000+2,000+2,000=12,000.

## Correction and governance verdict

`answer/THIRD_PARTY_NOTICES.md` was corrected from the obsolete sole-ViSoBERT statement. Repository controls, provenance reports, and manuscript Section 3 already state the authoritative composition. External UIT-VSFC, UIT-VSMEC, and AIVIVN resources remain evaluation-only; raw and processed text remain private; no public redistribution, relicensing, canonical-Kaggle, or resolved-source-authorization claim is made.

**Verdict: PASS WITH GOVERNANCE LIMITATIONS DISCLOSED.**
