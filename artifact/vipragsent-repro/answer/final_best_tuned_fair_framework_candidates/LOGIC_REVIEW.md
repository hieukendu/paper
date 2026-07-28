# Logic review — fair-framework optimization cycle v3

## A. Before modifying code

- Assumptions: canonical splits are `data/processed/vipragsent_{train,dev,test}.jsonl`; selection may use development labels only; canonical-test labels may only be used at the final promotion gate.
- Immutable inputs: canonical splits, `answer/final_best_tuned`, and all baseline prediction artifacts. Candidate outputs are confined to this directory and `outputs/final_best_tuned_fair_framework`.
- Candidate rule: targeted labels are irony, idiom_figurative, and code_switching. Protected label binaries (implicit_sentiment, sarcasm, mocking) are copied byte-for-value from the incumbent.
- Expected I/O: saved development probability JSONL files with exactly the 2,000 development IDs; paired test sources require the exact registered checkpoint plus a label-free inference command.
- Invariants checked: clean split hashes are captured before work; every candidate must be registered; every rejection must have a reason; no canonical test is opened for model selection.
- Discovered issue: v2 excluded the strongest code candidate because its test inference path had not been materialized, and used a rigid 70% fold rule despite the protocol saying that rule is not sufficient by itself.
- Correction: v3 records checkpoint availability separately from development eligibility, uses median delta + bootstrap + utility + seed stability + precision constraints, and never treats absent inference as a development rejection.
- Tests: pending after implementation.
- Safe to continue: yes; only development artifacts will be loaded during screening.

## B. After probability/artifact discovery

- Sources found: all six development seed probability artifacts are present and ID-aligned. Existing paired test probabilities are available only for `phobert_1`, `visobert_2`, and `visobert_3`.
- Exact registered checkpoint paths and hashes for `phobert_2`, `phobert_3`, and `visobert_1` are listed in `answer/final_best_tuned/final_selection.json`, but their checkpoint files are absent. Legacy test probability files for the missing sources exist under `answer/final_best_tuned_candidates/frozen_test_components`, but their producing checkpoint cannot be hash-verified.
- Invariant result: legacy files are evidence-only and are not eligible for paired inference. This is a reproducibility block, not a candidate-quality rejection.
- Correction: the probability registry records both legacy artifacts and missing-checkpoint status; v3 refuses to freeze/evaluate a test candidate that would rely on either.
- Tests: path, hash, ID-alignment, range, and schema checks are run by v3 before selection.
- Safe to continue: yes, for development-only selection and reporting.

## C. After cheap screening

Pending v3 execution.

## D. After repeated OOF

Pending v3 execution.

## E. Before freezing a candidate

Pending v3 execution.

## F. Before canonical-test inference

Pending v3 execution. The runner must stop here if an exact checkpoint is unavailable.

## G. Before modifying final_best_tuned

Pending v3 execution. `final_best_tuned` must remain untouched unless one frozen candidate passes all seven strict full-precision gates.

## C. After cheap screening

All six development source files passed strict probability loading, range, and ID checks. Cheap search used the mandated code triple and simplex weights (0.05 grid); only the top ten code configurations advance to repeated OOF. No test candidate probabilities or test labels were used.

## C. After cheap screening

All six development source files passed strict probability loading, range, and ID checks. Cheap search used the mandated code triple and simplex weights (0.05 grid); only the top ten code configurations advance to repeated OOF. No test candidate probabilities or test labels were used.

## C. After cheap screening

All six development source files passed strict probability loading, range, and ID checks. Cheap search used the mandated code triple and simplex weights (0.05 grid); only the top ten code configurations advance to repeated OOF. No test candidate probabilities or test labels were used.

## C. After cheap screening

All six development source files passed strict probability loading, range, and ID checks. Cheap search used the mandated code triple and simplex weights (0.05 grid); only the top ten code configurations advance to repeated OOF. No test candidate probabilities or test labels were used.

## D. After repeated OOF

Repeated evaluation completed for every top-ten cheap candidate per target. Eligibility used positive median delta, bootstrap P(delta>0) >= 0.80, positive paired utility, at least 60% positive split-seed runs, and a zero-introduced-FP rule for irony/idiom. No rigid fold-percentage rule was used. Group holdout summaries were written.

## E. Before freezing a candidate

The selected code candidate differs from the incumbent and passed development eligibility, but its `phobert_3` registered checkpoint is absent. Freeze aborted: a paired label-free inference path cannot be verified. No test probabilities or test labels were loaded after selection.

## F. Before canonical-test inference

Stopped safely: no frozen candidate exists because an exact checkpoint hash cannot be checked. Canonical test evaluation count remains zero.

## G. Before modifying final_best_tuned

Promotion gate was not reached. `final_best_tuned` remains unchanged; candidate/report/registry/status consistency checks passed.
