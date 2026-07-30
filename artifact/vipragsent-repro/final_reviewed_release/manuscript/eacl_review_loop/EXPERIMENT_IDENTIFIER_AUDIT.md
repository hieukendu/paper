# Experiment Identifier Audit

Status: PASS with a bounded author confirmation.

## Evidence inspected

- `answer/final_best_tuned/final_selection.json`
- `answer/final_best_tuned/configs/final_config.yaml`
- `answer/final_best_tuned/artifact_index.json`
- `answer/final_best_tuned/archive/38fa5fc/final_selection.json`
- `answer/final_best_tuned/archive/38fa5fc/configs/final_config.yaml`

## Finding

The identifiers `20260901`, `20260902`, `20260903`, `20260833`, `20260834`, and `20260835` occur consistently as suffixes of retained source/run labels, checkpoint paths, artifact-index paths, source assignments, and corresponding SHA-256 entries. Each maps to a concrete stored prediction or checkpoint record; they are not result values and are not used as calendar claims in the final-system selection logic.

The records establish their operational role as opaque run/seed labels. They do not document the creator's original naming convention, so this audit does not infer that the digit strings are dates or malformed dates.

## Repair

Section 4 now calls these values opaque run identifiers and keeps them in the traceability artifacts rather than displaying date-like names in the manuscript. The only retained `seed-20260520` occurrence is explicitly scoped to a historical single-seed diagnostic.

## Residual author action

Before any public artifact release, an author should confirm the original naming convention. This is a provenance clarification only; it does not alter the promoted result, the test boundary, or the source assignments documented above.
