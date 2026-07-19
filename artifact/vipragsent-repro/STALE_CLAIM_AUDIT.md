# Stale-Claim Audit

Date: 2026-07-19
Scope: active repository documentation and manuscript sources; generated LaTeX build files and `main(5).pdf` excluded.

## Result

No active document retains an obsolete claim that ViSoBERT is unevaluated, that the P0 ablation is single-seed, that P2 has one seed per budget, or that the new external diagnostics lack three-seed results.

| Search concept | Classification | Disposition |
|---|---|---|
| Missing / unevaluated ViSoBERT; future ViSoBERT | CURRENT (no matches) | The active plan and manuscript describe the verified three-seed baseline. |
| Single-seed ablation | HISTORICAL | The only retained match is the integration checkpoint's description of what was replaced. Active drafting state and manuscript say three-seed. |
| One seed per budget; future low-resource reruns | CURRENT (no matches) | P2 is documented as three-seed for all five budgets. |
| No external-benchmark multi-seed results | CURRENT (no matches) | The manuscript and plan qualify the recorded three-seed external diagnostics. |
| Independent reproduction | CURRENT | Every match correctly denies that artifact-level validation is an independent training rerun. |
| Public raw-data release | CURRENT | Matches state the continuing private-data / governance boundary or list the absent release protocol as a gap. |
| Data-efficiency superiority | CURRENT | Matches deny such a conclusion for the P2 mixed-monotonicity and uncertainty results. |
| Causal source effect | CURRENT | Matches explicitly restrict P1 to descriptive, observational source-stratified sensitivity. |

The active `manuscript/DRAFTING_STATE.md` stale P0 and P2 descriptions were updated during this audit. Historical wording was retained only in `NEW_EXPERIMENT_INTEGRATION_CHECKPOINT.md`, where it is explicitly labeled as obsolete prior state.
