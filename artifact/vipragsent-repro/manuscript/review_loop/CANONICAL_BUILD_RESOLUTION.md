# Canonical Build Resolution

## Discovery

The clean worktree at commit `f4016e92d6d6e97c3c41955733c68eb7bb0a425c` contains one compiled manuscript PDF: `latex/main.pdf`. No `final_build/` or `build_final/` directory exists in this committed tree. The other discovered PDFs are embedded figure assets, not manuscript builds.

| Object | Commit introducing/latest changing object | SHA-256 / pages | Result-text check | Decision |
|---|---|---|---|---|
| `latex/main.pdf` | `e580d527bc56a8b157f6181f5df0fb0c39d7211b` | `5459eef749152bc8c33c322b770be2a6087f5720db475d4b4b2817f476014755`; 14 pages | Contains `ViPragSent-Final`, `84.39`, `1.56`, and the strict-per-label-dominance qualification | **CANONICAL** |
| `latex/figures/*.pdf` | historical figure commits | not manuscript PDFs | not applicable | Figure assets only |

## Source relationship

The canonical source tree is `artifact/vipragsent-repro/manuscript/latex/`, headed by `main.tex` and its included `sections/`, `tables/`, appendices, and figures. `main.pdf` was rebuilt from this tree in Phase 8 and its text matches the current promoted-result framing.

## Path discrepancy

The previously observed untracked `final_build/` directory existed only in another dirty working tree. It is not present in the latest committed `main`, has no commit provenance here, and is therefore not authoritative. The canonical build is `artifact/vipragsent-repro/manuscript/latex/main.pdf`.
