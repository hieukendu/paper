# LaTeX setup (do this only after framing approval)

Use the official ACL style repository, not a copied template from another conference:

- Source: <https://github.com/acl-org/acl-style-files>
- Template entry point: `acl_latex.tex`
- Official policy reference: <https://aclrollingreview.org/cfp>

For an anonymous ARR submission:

1. Copy the current `acl.sty`, `acl_latex.tex`, `acl_natbib.bst`, and any required bibliography support files into this directory.
2. Rename the working TeX file only after the title and venue cycle are confirmed.
3. Keep author names, affiliations, identifiable acknowledgements, and identifying repository links out of the anonymous source and supplement.
4. Do not modify the ACL style files.
5. Before submission, re-check the future venue’s call for page limits, checklist requirements, and its exact anonymity policy.

The anonymous manuscript entry point is `main.tex`. Compile it with XeLaTeX so
that the Times New Roman body font and its true bold heading hierarchy are
available:

```text
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```
