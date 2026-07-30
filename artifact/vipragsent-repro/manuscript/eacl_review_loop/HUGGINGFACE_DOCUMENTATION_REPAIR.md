# Proposed Hugging Face Documentation Repair

Status: internal proposed update; no remote Hugging Face page was edited by this workflow.

## Scope

The public checkpoint and adapter archives are verified as publicly accessible, but public availability is not the same as a fully specified independent training reproduction. The following additions should be made to the three corresponding public cards before a de-anonymized artifact release. They are derived only from the local run manifests, pinned archive revisions, and artifact registry.

## Shared card block

> **Research and reproducibility scope.** This archive provides the recorded checkpoint or adapter artifacts and their manifest/history files for artifact inspection and evaluation reuse. It does not, by itself, establish an independent reproduction of the historical training run. Raw ViPragSent source text is private and is not distributed. Reproduction of reported evaluation requires the authorized canonical labels and the exact preprocessing/evaluation environment described in the repository.

> **Use and governance.** The archive is for research use. Do not infer rights to redistribute, reconstruct, or release private source text or third-party derived materials. Consult the project data-governance record before any data access or redistribution request.

## Encoder checkpoint card: `vipragsent-experiment-checkpoints`

Add:

- Archive revision: `ec3ea6a832b832d2f69eac70dc8ebddbf662a2fa`.
- Contents: `best.pt`, per-seed manifest/history records, and hash-listed artifact metadata.
- Recorded seed identifiers: `20260520`, `20260521`, `20260522`; these are run identifiers, not claims about experiment dates.
- Evaluation scope: supports recorded-checkpoint evaluation where the authorized private labels and matching preprocessing are available.
- Limitation: no public raw corpus; no claim of an independently rerunnable historical training environment.

## Vistral and Sailor QLoRA adapter cards

Add the same shared block plus:

- Contents: PEFT adapter weights/configuration and per-seed manifest/history records for seeds `20260520`, `20260521`, and `20260522`.
- Archive revisions verified during the EACL loop: Vistral `d7d8750cc73f39a841b29cf9ee377234509cebb6`; Sailor `6b8af60242827d45ec3e32a6a666daa090c1ce20`.
- Base model requirement: users must separately obtain the applicable base model under its own terms.
- Reproduction boundary: manifests support inspection of recorded adapter runs but do not independently freeze every historical remote revision or software dependency.
- Evaluation boundary: do not describe a public test set, a public reproduction, or permission to redistribute underlying text.

## Author action before publication

1. Check that each public card's license field accurately reflects the archive and each required base model.
2. Confirm the displayed model identifiers and immutable revision hashes against the live Hub pages.
3. Add a contact/access route for controlled research requests only if the authors and data-rights holders approve it.
4. Do not add a direct repository or account link to the anonymous manuscript; these instructions are for post-decision or controlled artifact material.
