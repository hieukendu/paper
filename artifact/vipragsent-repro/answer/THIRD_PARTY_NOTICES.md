# Third-party materials and notices

ViPragSent does not relicense third-party datasets or base models.

## Dataset source

- The 12,000-record ViPragSent collection is derived solely from a local
  ViSoBERT export. Raw text is restricted to private research and is not
  licensed for redistribution by this repository.
- UIT-VSFC, UIT-VSMEC, and AIVIVN are used only as external evaluation
  benchmarks. They are not incorporated into ViPragSent and remain subject to
  their original terms and citation requirements.

## Base models

- PhoBERT: `vinai/phobert-base`
- XLM-R: `FacebookAI/xlm-roberta-large`
- Sailor: `sail/Sailor-7B-Chat` at revision
  `19066fae0a8a3ba029c190d8e3dacccf4d1234b8` (Apache-2.0). The researcher
  attests that this base was downloaded on 2026-07-14 before QLoRA fine-tuning.
- Vistral: `Viet-Mistral/Vistral-7B-Chat` at revision
  `d331b64e61b935cc43c2b3010ae9fb4fde599b45` (AFL-3.0; gated access terms).
  The researcher attests that this base was downloaded on 2026-07-14 before
  QLoRA fine-tuning.

Each checkpoint or adapter must be used with the applicable base-model license
and access terms. Base weights are not stored in this Git repository or in the
ViPragSent adapter repositories. The archived run manifests retain local cache
paths, not remote revision fields; the two pins above are documented
download-time provenance, not independently logged runtime snapshots.

## Project policy

Original project materials are governed by the repository's
`ViPragSent Private Research Use Terms`. Those terms do not grant permission to
redistribute third-party text or weights. Before any public dataset release,
the project owner must obtain and archive the applicable source permission.
