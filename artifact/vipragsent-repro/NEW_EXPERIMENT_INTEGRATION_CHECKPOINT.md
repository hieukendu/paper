# New Experiment Integration Checkpoint

Date: 2026-07-19
Mode: direct-mode; automatic continuation authorized by the requested repository-wide integration workflow.

## Validation verdict

**PASS: VERIFIED AT ARTIFACT LEVEL.** The validator recomputed all 93 core per-seed reports and 45 external per-seed metrics from prediction JSONL, confirmed 90/90 new (96/96 wider-scope) exact gold-ID sets, found no duplicate IDs, split leakage, reused seed predictions, metric mismatch, aggregate mismatch, unresolved ViSoBERT revision, or broken manifest link. The 46 stored aggregates comprise 31 core P0/P1/P2 aggregates and 15 external aggregates; the expected count of 32 is a stale scope count. This is not an independent training reproduction.

## Newly verified experiments

- P0: three seeds for `vipragsent_full`, `phobert_finetune`, and the four recorded ablations (`vipragsent_no_rationale`, `vipragsent_no_emotion`, `vipragsent_no_polarity`, and `vipragsent_no_uncertainty`).
- P0: three-seed `visobert_finetune`, pinned to `uitnlp/visobert` revision `196a62afad9cbe4f52a54aabad828b13f0eec59a`.
- P1: descriptive source-stratified sensitivity over 1,666 `visobert_local` and 334 `VIVID_seed_and_irony_generation` test records.
- P2: three seeds for two systems at positive budgets 64, 128, 256, 512, and 1,024 (total training sizes 256, 512, 1,024, 2,048, and 4,096 under the verified 1:3 positive-to-other design).
- New three-seed external evaluations for the four ablations and ViSoBERT.

## Obsolete active claims

- ViSoBERT is an unevaluated/future baseline.
- The ablation is single-seed or its three-seed extension is future work.
- Low-resource sarcasm has one seed per budget or requires multi-seed reruns.
- External benchmark evidence has no multi-seed expansion.

## Updated evidence-bounded claim set

- C2: retain the comparative-evaluation claim, adding the evaluated ViSoBERT baseline and recorded external three-seed evaluations while preserving API single-run qualifications.
- C4: retain the trade-off claim, now based on three-seed ablation evidence; describe a recorded association, not an auxiliary-task cause, universal effect, or robustness guarantee.
- C6: retain low-resource sarcasm as a three-seed, exploratory analysis. Report mixed monotonicity across systems and uncertainty; do not assert data-efficiency superiority.
- C7 (analysis claim, not a headline contribution): P1 is a descriptive source-stratified sensitivity analysis with unequal stratum sizes. It must not support causal source, domain-superiority, or general-transfer claims.

## Proposed manuscript/table plan and page impact

- Main Table 2: add ViSoBERT to the pragmatic comparison.
- Main Table 3: replace the former single-seed ablation panel with the three-seed P0 ablation and retain the external diagnostic as a qualified, separate panel.
- Analysis: introduce P1 in the main text as a compact descriptive table/pointer; describe P2 briefly in Analysis and place the full budget-by-system table in the appendix.
- Appendix: add P1 source-stratified and P2 low-resource tables, with seed counts and interval definitions in captions.
- Expected impact: approximately 0.4--0.7 additional main-content pages. The eight-page target remains an internal target and must be rechecked after compilation; secondary P2 detail stays in the appendix to control main-text space.

## Remaining governance issues

- VIVID authorization and license/research-use review remain unresolved; P1 does not change this.
- Raw/processed social-media text remains private; no public-release or unrestricted-use claim is permitted.
- No independent full retraining reproduction was performed.

## Repository metadata updated before manuscript prose

- `results/artifact_index.json` and `answer/results/artifact_index.json`
- `results/claim_ledger.csv` and `answer/results/claim_ledger.csv`
- `configs/artifact_registry.json` and `answer/reproducibility/artifact_registry.json`
- `answer/reproducibility/verification_manifest.json`
- `data/manifest/source_registry.json` and `answer/data_provenance/source_registry.json`
- `answer/README.md` and `answer/run_manifests/README.md`
