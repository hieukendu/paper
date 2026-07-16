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
- Sailor: `sail/Sailor-7B`
- Vistral: `Viet-Mistral/Vistral-7B-Chat`

Each checkpoint or adapter must be used with the applicable base-model license
and access terms. Base weights are not stored in this Git repository or in the
ViPragSent adapter repositories.

## Project policy

Original project materials are governed by the repository's
`ViPragSent Private Research Use Terms`. Those terms do not grant permission to
redistribute third-party text or weights. Before any public dataset release,
the project owner must obtain and archive the applicable source permission.
