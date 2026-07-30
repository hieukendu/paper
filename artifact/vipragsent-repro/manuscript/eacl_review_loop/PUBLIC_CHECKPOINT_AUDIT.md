# Public Checkpoint Audit

Anonymous Git access and remote revision checks succeeded on 2026-07-30. LFS-tracked model/adaptor paths are visible in each public tree; no large weights were copied into this repository or workspace.

| Repository | Revision | Public evidence | Contents | Classification |
|---|---|---|---|---|
| `Thundergod2007/vipragsent-experiment-checkpoints` | `ec3ea6a832b832d2f69eac70dc8ebddbf662a2fa` | `git ls-remote` and public tree | `best.pt`, histories, and manifests for encoder, ablation, and low-resource runs | PUBLIC_AND_EVALUATION_READY |
| `Thundergod2007/vipragsent-vistral-7b-qlora` | `d7d8750cc73f39a841b29cf9ee377234509cebb6` | `git ls-remote` and public tree | adapter safetensors, adapter config, histories, manifests, three seed directories | PUBLIC_BUT_DOCUMENTATION_INCOMPLETE |
| `Thundergod2007/vipragsent-sailor-7b-qlora` | `6b8af60242827d45ec3e32a6a666daa090c1ce20` | `git ls-remote` and public tree | adapter safetensors, adapter config, histories, manifests, three seed directories | PUBLIC_BUT_DOCUMENTATION_INCOMPLETE |

Public availability supports artifact access and reuse. It does not demonstrate an independent rerun or retraining reproduction. The two QLoRA repositories require fully explicit base-model revision, environment, and end-to-end evaluation commands before being classified training-reproduction-ready.
