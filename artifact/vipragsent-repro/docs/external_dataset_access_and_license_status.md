# External dataset access and license status

**Review date:** 2026-07-18  
**Scope:** UIT-VSFC, UIT-VSMEC, and the local AIVIVN-2019 Kaggle mirror used
only for the external ordinary-task diagnostic. This note does not grant any
license and does not replace a source-rights review.

## Decision

The repository must not redistribute raw or processed copies of any of these
external datasets. A release, controlled-access arrangement, or new download
workflow requires an archived source permission or a documented, reviewed
license applicable to the exact artifact.

## Source review

| Dataset | Local artifact / source record | License finding | Status | Repository action |
| --- | --- | --- | --- | --- |
| UIT-VSFC | `data/raw/uit_vsfc`; source registry records a Hugging Face dataset script / Google Drive source. | The SEACrowd record identifies the source Google Drive and reports its dataset license as `unknown`. No primary license document for the exact local files was found. | `UNRESOLVED_PRIMARY_LICENSE` | Evaluation-only; do not redistribute. |
| UIT-VSMEC | `data/raw/uit_vsmec`; source registry records `tridm/UIT-VSMEC`. | The recorded source card does not establish a source license. A separate Hugging Face mirror advertises CC BY-NC-SA 4.0, but that mirror is not evidence that the original corpus carries that license. | `UNRESOLVED_PRIMARY_LICENSE` | Evaluation-only; do not redistribute or relicense from mirror metadata. |
| AIVIVN-2019 | `data/raw/aivivn_2019`; source registry records `mcocoz/aivivn-2019` as an unofficial mirror. | The scholarly description identifies the historical contest but does not grant a data license. The organizer page is not available for terms verification, and the local mirror is not asserted canonical. | `UNRESOLVED_PRIMARY_LICENSE` | Evaluation-only; do not redistribute, call canonical, or rely on its labels as organizer-issued. |

## Evidence anchors

- UIT-VSFC: the SEACrowd dataset record identifies the original Google Drive
  source and reports the license as unknown.
- UIT-VSMEC: the project-recorded `tridm/UIT-VSMEC` source provides corpus
  metadata; the CC BY-NC-SA 4.0 value observed on another mirror is retained
  only as a non-authoritative lead for future rights review.
- AIVIVN-2019: `nguyen-etal-2020-efficient` (DOI
  `10.3233/FAIA200579`) supports the historical dataset description, not a
  license, canonical-file identity, or redistribution right.

## Required before any change in access status

1. Identify the exact source URL and version or commit for each artifact.
2. Archive the source license or written permission with a retrieval date.
3. Confirm that the terms cover the intended action (internal use,
   peer-review access, controlled access, or public redistribution).
4. Update `answer/data_provenance/source_registry.json` and this ledger with
   the evidence, scope, and reviewer of the decision.
