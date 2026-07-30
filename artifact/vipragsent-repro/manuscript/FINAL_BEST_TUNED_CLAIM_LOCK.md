# Final Best-Tuned Claim Lock

## Canonical System and Composition

**Canonical manuscript name:** `ViPragSent-Final`.

ViPragSent-Final is a frozen, label-wise prediction ensemble, not one ordinary trained encoder. It uses three-seed ViSoBERT for implicit sentiment and sarcasm, a 0.6 ViSoBERT / 0.4 PhoBERT probability blend for irony and code-switching, uniform three-seed ViSoBERT for idiom/figurative language, and a targeted uniform three-seed mocking component. Label thresholds are selected/refit on development data; the frozen configuration is evaluated once on the canonical test partition.

## Locked Results

| Metric | ViPragSent-Final | Best-tuned baseline leader | Difference |
|---|---:|---:|---:|
| Implicit sentiment | 67.9829 | 60.8470 | +7.1359 |
| Sarcasm | 80.2781 | 80.0318 | +0.2463 |
| Irony | 97.0177 | 97.4132 | -0.3955 |
| Idiom/figurative language | 97.1820 | 97.2958 | -0.1138 |
| Code-switching | 81.8164 | 81.9458 | -0.1294 |
| Mocking | 82.0528 | 81.9802 | +0.0726 |
| Macro pragmatic F1 | 84.3883 | 82.8250 (Vistral-7B best-tuned) | +1.5633 |

## Permitted Claim

ViPragSent-Final achieves the highest macro pragmatic F1 among the evaluated systems, reaching 84.39 and exceeding the strongest aggregate best-tuned baseline, Vistral-7B, by 1.56 points.

## Mandatory Qualifications

- The aggregate lead does not establish strict per-label dominance: irony, idiom/figurative language, and code-switching remain 0.40, 0.11, and 0.13 points below their respective best-tuned baseline leaders.
- The final system is development-selected and more complex than a single model; no causal account of its gain is established.
- The final configuration was selected with five-fold development OOF evidence, frozen, and evaluated once on canonical test labels.
- Artifact inspection is `ANALYZED`, not independently reproduced; locally absent checkpoints prevent a full training rerun.
- Existing data access, licensing, privacy, and governance limitations remain unchanged.

## Prohibited Claims

- Strict all-label superiority, universal dominance, or state-of-the-art status.
- Statistical equivalence, irrelevance, or negligibility of the three deficits without a supporting analysis.
- A claim that the ensemble architecture caused the gain, that every auxiliary task improves performance, or that the result is an independent reproduction.
- A public release or authorization claim for raw dataset text.

## Superseded Statements

The manuscript's former 73.7469 ViPragSent result and 82.8250 Vistral-7B leadership claim are historical baseline/progression evidence only. They cannot describe the promoted final system or its final aggregate comparison.
