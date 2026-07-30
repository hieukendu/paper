# Loop 01 Post-review Repair Log

| Issue | Repair | Verification |
| --- | --- | --- |
| RER-01 | Replaced the composite appendix float with normal appendix sections/tables in a `\twocolumn[...]` full-width continuation block. | Rebuilt PDF p.14; headings and tables are in normal document flow, and the document remains two-column. |
| RER-02 | Removed direct Vistral Hugging Face and VIVID GitHub URLs from both anonymous bibliography files; retained bounded source-type notes. | Bibliography files hash-identical; targeted URL search returns no GitHub/Hugging Face URL in the anonymous bibliography. |
| RER-03 | Removed the minipage width interaction that produced the appendix overfull box. | Final `main.log` contains no `Overfull \hbox`, undefined citation, undefined reference, or changed-label warning. |
| RER-04 | Replaced date-shaped ViSoBERT source IDs in Section 4 prose with role-based wording; IDs stay in traceability artifacts. | Targeted section inspection and final PDF check. |
| Citation completion | Corrected Ghosh cite key to the verified 2020 record and added the Toukmaji--Flanigan ACL DOI in both bibliographies. | 13-key citation audit; byte-identical bibliography SHA-256 `17C68631733CBC82A5C76CFC8E221E3C92541A415D6646D719E5F80F48E446D0`. |

## Build record

- Command: `xelatex -interaction=nonstopmode -halt-on-error main.tex`, `bibtex main`, then XeLaTeX until stable.
- Final current PDF: `latex/main.pdf`.
- Pages: 14.
- SHA-256: `FE97029DA8FBD5185F4950CF36AEF78839EF997BA338BEA722C65299DCE6C2F1`.
- Text extraction confirms `ViPragSent-Final`, `84.39`, `1.56`, and the three per-label qualifications.
