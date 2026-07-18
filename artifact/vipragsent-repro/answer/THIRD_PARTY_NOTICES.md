# Third-party materials and notices

ViPragSent does not relicense third-party datasets or base models.

## Dataset source

- The final 12,000-record ViPragSent collection contains 10,000 retained rows
  from a local ViSoBERT export and 2,000 adjudicated rows identified in their
  stored record metadata as `VIVID_seed_and_irony_generation` candidates. The
  latter replaced earlier ViSoBERT rows before the fixed gold partitions used
  by the reported experiments were formed. The upstream VIVID source/version
  and license have not yet been verified. Raw text is restricted to private
  research and is not licensed for redistribution by this repository.
- UIT-VSFC, UIT-VSMEC, and AIVIVN are used only as external evaluation
  benchmarks. They are not incorporated into ViPragSent and remain subject to
  their original terms and citation requirements. No primary license document
  or research-use authorization for the exact external artifacts was verified
  on 2026-07-18; mirrors must not be treated as authority to redistribute the
  original datasets or as evidence of authorization to use them.

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
the project owner must obtain and archive applicable source permission or
document and review an applicable source license.
