# Engineering Plan: FreshBench-RAG

Project: `EMNLP-2026-freshbench-rag`
Title: *FreshBench-RAG: Rolling Time-Stamped Evaluation for Retrieval-Augmented QA*

> Scope of this document: a concrete, file-level plan to convert this auto-generated
> manuscript package into a real, submission-ready paper. It does **not** run experiments
> or modify any file other than creating this `PLAN.md`.

---

## 1. Paper summary & current state

FreshBench-RAG proposes a *rolling, time-stamped* evaluation protocol for retrieval-augmented
QA: monthly corpus construction from Reuters/BBC/AP news feeds plus Wikipedia recent edits,
GPT-4o question generation with grounding verification (BERTScore), a three-tier temporal
**contamination detection** filter (C4 8-gram overlap, closed-book parametric-hit tagging, 5%
human spot-check), and metrics that track answer F1/EM, citation precision/recall, a composite
**freshness score** `F`, temporal-degradation slope `β`, and Static-vs-Fresh rank instability
(Kendall τ). The headline claims are that Fresh F1 drops 8–14 points below Static, rankings
disagree 38% of the time, disabling contamination detection inflates Fresh F1 by ~+4.9 points
(61% memorized), and `F` correlates with human "answer currency" at ρ=0.81 vs 0.63 for raw F1.

**Every quantitative result in this package is SYNTHETIC and must be replaced before submission.**
All table numbers are hardcoded inline in [`sections/*.tex`](sections/) and
[`appendix/*.tex`](appendix/); all five in-text figures are hardcoded TikZ/pgfplots data (not
generated from any file); the four PDFs in [`figures/`](figures/) are fixed-seed simulator output
([`scripts/make_artifacts.py`](scripts/make_artifacts.py),
[`scripts/generate_figures.py`](scripts/generate_figures.py)) and are **never `\includegraphics`'d
into the paper**. The [`results/*.json`](results/) files carry a *generic placeholder schema*
(`quality/risk/cost`, methods named "Control"/"Cheap baseline"/"Strong API") that does **not match
any table in the paper** — there is currently no results file backing any reported number. There is
**no `configs/` directory** (the proposal's datasets/models live only in prose). Multiple internal
inconsistencies already exist (see §7). Treat the entire numeric layer as unbuilt.

---

## 2. Complete TABLE inventory

All tables are hand-authored inline LaTeX with synthetic numbers. None is generated from a results
file today. "Must be produced by" lists the real experiment + the results file that should drive it
after the rebuild (see §4–§6).

| # | `\label` | Source file:line | Metrics / columns reported | Real experiment + results file that must produce it |
|---|----------|------------------|----------------------------|------------------------------------------------------|
| T1 | `tab:dataset-stats` | [`sections/05_results.tex:12`](sections/05_results.tex) | Per-split Raw / Retained / Contam. rate / IAA(κ); Static 4,082→3,847, Fresh 2,467→2,291, quarterly rows, human-verified 510→467 | Benchmark construction + contamination filtering counts → new `results/dataset_stats.json` |
| T2 | `tab:main` | [`sections/05_results.tex:41`](sections/05_results.tex) | Static & Fresh **Token F1**, **P_cite** per retriever (BM25/DPR/ColBERT/SPLADE + GPT-4o), parametric baselines (GPT-4o-ZS, GPT-4o-CoT), Oracle; ΔF1; ±std over 3 seeds; † significance | Main RAG evaluation → `results/main_results.json` (re-schema'd) |
| T3 | `tab:per-condition` | [`sections/05_results.tex:109`](sections/05_results.tex) | Fresh F1 & P_cite per system × 4 change-types (Entity/Number/Relationship/Negation) + per-category counts (867/611/524/289); 3 seeds; † | Change-type stratified eval → `results/per_condition.json` |
| T4 | `tab:time-decay` | [`sections/05_results.tex:177`](sections/05_results.tex) | Fresh F1 at 1/3/6/12 mo post-cutoff for BM25/ColBERT/SPLADE + GPT-4o-ZS; OLS slope `β̂`; 3 seeds | Temporal binning + OLS fit (Eq. `eq:degradation`) → `results/time_decay.json` |
| T5 | `tab:ablation` | [`sections/06_analysis.tex:12`](sections/06_analysis.tex) | Fresh F1 & Δ vs Full for w/o TCD/CFT/GV/MRU + 2-component ablations; ColBERT+GPT-4o; 3 seeds | Component ablation on 2,291 Fresh items → `results/ablations.json` (re-schema'd) |
| T6 | `tab:error-analysis` | [`sections/06_analysis.tex:45`](sections/06_analysis.tex) | Error taxonomy (Retrieval: indexing latency/topic drift/vocab mismatch; Generation: stale override/unsupported citation/temporal ambiguity); Count + Share%; κ=0.74 | Manual error coding of 400 all-system failures → `results/error_taxonomy.csv` (re-schema'd) |
| T7 | `tab:efficiency` | [`sections/06_analysis.tex:79`](sections/06_analysis.tex) | Per-item time(s), peak RAM(GB), items/hr per system + pipeline overhead | Wall-clock profiling over 500 items → new `results/efficiency.json` |
| T8 | `tab:bertscore-threshold` | [`sections/06_analysis.tex:195`](sections/06_analysis.tex) | Precision/Recall/F1/n_labeled across θ∈{0.60,…,0.80} for grounding verification | Threshold sweep vs human grounding labels → new `results/bertscore_threshold.json` |
| T9 | `tab:leakage` | [`sections/06_analysis.tex:221`](sections/06_analysis.tex) | Per-split leakage count/rate/F1-drop on 412 human-verified items | Manual leakage audit + decontaminated re-scoring → new `results/leakage.json` |
| T10 | *(unlabeled)* Extended ablation | [`appendix/d_extended_tables.tex:4`](appendix/d_extended_tables.tex) | Component vs Score (Full 58.2 / No protocol control / No uncertainty / No validation filter) | **Placeholder schema; mislabeled vocabulary** — replace with real ablation or delete; backed by `results/ablations.json` |
| T11 | *(unlabeled)* Extended robustness | [`appendix/d_extended_tables.tex:20`](appendix/d_extended_tables.tex) | Slice vs Score (Seen-style/Hard-language/Noisy/Fresh-shifted) | **Placeholder schema** — replace with real robustness slices or delete; backed by `results/robustness.json` |
| T12 | `tab:qual-success` | [`appendix/e_qualitative_examples.tex:31`](appendix/e_qualitative_examples.tex) | Qualitative success cases (Input / FreshBench output / Baseline output) | Curated real transcripts from replay logs |
| T13 | `tab:qual-comparison` | [`appendix/e_qualitative_examples.tex:61`](appendix/e_qualitative_examples.tex) | Qualitative comparison vs Hybrid+GPT-4o no-filter | Curated real transcripts from replay logs |

Note: `tab:error-analysis` (T6) is **referenced as `tab:error-taxonomy`** in
[`appendix/e_qualitative_examples.tex:4`](appendix/e_qualitative_examples.tex) — a broken cross-reference (see §7).

---

## 3. Complete FIGURE inventory

**Important architecture note:** the five figures the paper actually renders are **inline TikZ/pgfplots
with data literals embedded in the `.tex`** — they are NOT produced by any script and do NOT read any
file. The four PDFs in [`figures/`](figures/) generated by the two scripts are **orphaned** (never
`\includegraphics`'d). [`scripts/generate_figures.py`](scripts/generate_figures.py) even describes a
different figure set (a 2-panel F1/citation decay, an ablation bar, a Kendall-τ line, a calibration
plot) than what the paper shows. The rebuild must pick ONE pipeline (recommended: real matplotlib
PDFs read from `results/*.json`, swapped in via `\includegraphics`) and make scripts, figures, and
prose agree.

| # | `\label` (or PDF) | Producing function / location | What it plots | Real data source needed |
|---|-------------------|-------------------------------|---------------|--------------------------|
| F1 | `fig:main-bar` | Inline TikZ, [`sections/05_results.tex:69–98`](sections/05_results.tex) | Static vs Fresh Token F1 bar chart, 4 systems | Same data as T2 → `results/main_results.json` |
| F2 | `fig:per-category` | Inline TikZ, [`sections/05_results.tex:133–166`](sections/05_results.tex) | Per-change-type Fresh F1, 4 systems × 4 categories | Same data as T3 → `results/per_condition.json` |
| F3 | `fig:time-decay` | Inline TikZ, [`sections/05_results.tex:195–224`](sections/05_results.tex) | Fresh F1 vs months post-cutoff, 3 systems; decay curves | Same data as T4 → `results/time_decay.json` |
| F4 | `fig:scaling` | Inline TikZ, [`sections/06_analysis.tex:105–132`](sections/06_analysis.tex) | Fresh F1 vs retrieval budget k∈{1,3,5,10,15}, ColBERT & BM25 | New retrieval-budget sweep → new `results/scaling.json` |
| F5 | `fig:cross-domain` | Inline TikZ, [`sections/06_analysis.tex:142–175`](sections/06_analysis.tex) | Fresh F1 across News/Wikipedia/MedQA-Fresh/LegalBench-Rolling, 4 systems | Cross-domain eval (2 held-out domains) → new `results/cross_domain.json` |
| P1 | `figures/fig1_tradeoff.pdf` *(orphaned)* | `write_pdf(...fig1...)` [`scripts/make_artifacts.py:79`](scripts/make_artifacts.py); also matplotlib decay panels [`scripts/generate_figures.py:55–107`](scripts/generate_figures.py) | Quality bars (make_artifacts) / F1+citation decay (generate_figures) | Decommission or wire to real decay data |
| P2 | `figures/fig2_ablation.pdf` *(orphaned)* | `write_pdf(...fig2...)` [`scripts/make_artifacts.py:85`](scripts/make_artifacts.py); matplotlib bars [`scripts/generate_figures.py:110–156`](scripts/generate_figures.py) | Ablation bar chart | Wire to real `results/ablations.json` or decommission |
| P3 | `figures/fig3_robustness.pdf` *(orphaned)* | `write_pdf(...fig3...)` [`scripts/make_artifacts.py:91`](scripts/make_artifacts.py); matplotlib τ-line [`scripts/generate_figures.py:159–195`](scripts/generate_figures.py) | Robustness bars / Kendall-τ over 12 months | Wire to real rank-instability data or decommission |
| P4 | `figures/fig4_calibration.pdf` *(orphaned)* | `write_pdf(...fig4...)` [`scripts/make_artifacts.py:97`](scripts/make_artifacts.py); matplotlib reliability [`scripts/generate_figures.py:198–244`](scripts/generate_figures.py) | Citation-confidence calibration | New calibration data (`results/calibration.json` re-schema'd) |

---

## 4. Data → artifact dependency map

```
EXPERIMENTS (to run)                 RESULTS FILE (schema to define)        FEEDS TABLES / FIGURES
─────────────────────────────────────────────────────────────────────────────────────────────────
Benchmark build + 3-tier filter ──▶  results/dataset_stats.json        ──▶  T1 (tab:dataset-stats)
Main RAG eval (4 retr × GPT-4o,      results/main_results.json (re-      ──▶  T2 (tab:main), F1 (fig:main-bar)
  +ZS/+CoT, +Oracle), 3 seeds          schema'd to real metrics)
Change-type stratified eval     ──▶  results/per_condition.json        ──▶  T3 (tab:per-condition), F2 (fig:per-category)
Temporal binning + OLS slope    ──▶  results/time_decay.json           ──▶  T4 (tab:time-decay), F3 (fig:time-decay)
Component ablation (ColBERT)    ──▶  results/ablations.json (re-       ──▶  T5 (tab:ablation), T10, P2
                                       schema'd)
Manual error coding (400)       ──▶  results/error_taxonomy.csv (re-   ──▶  T6 (tab:error-analysis)
                                       schema'd)
Wall-clock profiling            ──▶  results/efficiency.json           ──▶  T7 (tab:efficiency)
BERTScore θ sweep               ──▶  results/bertscore_threshold.json  ──▶  T8 (tab:bertscore-threshold)
Leakage audit + re-score        ──▶  results/leakage.json             ──▶  T9 (tab:leakage)
Retrieval-budget k sweep        ──▶  results/scaling.json             ──▶  F4 (fig:scaling)
Cross-domain eval (4 domains)   ──▶  results/cross_domain.json        ──▶  F5 (fig:cross-domain), T11
Freshness-score vs human study  ──▶  results/freshness_human.json     ──▶  prose §5.5/§6.7 (ρ=0.81 etc.)
Calibration buckets             ──▶  results/calibration.json (re-     ──▶  P4 (fig4_calibration)
                                       schema'd)
Per-claim provenance            ──▶  results/claim_ledger.csv         ──▶  every numeric claim (Prop. 1, appendix C)
```

Current reality: every arrow above is **broken** — tables read literals from `.tex`, figures read
literals from TikZ, and the JSON files hold an unrelated placeholder schema. The rebuild must
physically connect each arrow.

---

## 5. Experiments to run

There is **no `configs/` directory**; the datasets/models/protocol live only in
[`sections/04_experimental_setup.tex`](sections/04_experimental_setup.tex) and
[`notes/source_note.md`](notes/source_note.md). **First deliverable: create `configs/*.yaml`** that
declare every dataset, model, retriever, and threshold below so the run is reproducible and auditable
(recommended: `configs/datasets.yaml`, `configs/models.yaml`, `configs/retrieval.yaml`,
`configs/eval.yaml`).

### 5.1 Datasets to obtain / build
- **Primary FreshBench corpus**: monthly harvest from **Reuters, BBC World Service, AP** RSS feeds +
  **Wikipedia** recent edits (diff ≥50 tokens), plus BEIR subsets with publication metadata. Dedup via
  **MinHash LSH** (Jaccard 0.8). Reconcile the window: prose currently disagrees (setup says **Jan–Dec
  2024, 12 months**; results say **Jan 2023–Mar 2025, 26 months, 3,847 Static + 2,291 Fresh**) — pick one
  and make all tables/figures match.
- **Static control split** (pre-cutoff Wikipedia + BEIR) — required for the Static-vs-Fresh contrast (T2)
  and the rank-instability reference vector `r_static` (Eq. `eq:rank-tau`).
- **Human-verified subset** (~412–467 items; reconcile the two numbers and the κ=0.73 vs 0.76 conflict).
- **Robustness slice** (~528 adversarially perturbed items) and **Ablation slice** (~891 items).
- **Held-out domains** for F5: **MedQA-Fresh** (PubMed) and **LegalBench-Rolling** (US court opinions).
- **C4 / Common Crawl 5% sample** ([`raffel2020t5`]) for 8-gram contamination overlap.
- Record source licenses, URLs, checksums, and a `MANIFEST` (per appendix C timestamp invariant
  `t_s(i) ≤ t_c(i) ≤ t_e(i)`).

### 5.2 Models / retrievers / baselines (use the paper's exact names)
- **Retrievers** (top-k=5): `BM25` (rank-bm25 / PyLucene, k1=1.2 b=0.75), `DPR`
  (`facebook/dpr-question_encoder-multiset-base` + FAISS flat-L2), `ColBERT v2`
  (`colbert-ir/colbertv2.0`, PLAID), `SPLADE` (`naver/splade-cocondenser-ensembledistil`), and
  **Hybrid** (BM25 top-100 + DPR rerank via RRF k=60). Dense index uses `all-MiniLM-L6-v2` +
  `faiss-cpu`. *(Note: the abstract/T2 evaluate four retrievers without Hybrid, but Hybrid is defined
  in setup and used in appendix E — decide whether Hybrid is in main results and make all sections agree.)*
- **Generators**: `GPT-4o` (`gpt-4o-2024-08-06`), `Claude 3 Haiku` (`claude-3-haiku-20240307`),
  `Llama-3-8B-Instruct` (llama.cpp, 4-bit GGUF). *(Setup promises 5×3 combos but every table reports
  only the GPT-4o pairings + one BM25+Llama figure line in the orphaned script — either run the full
  grid or trim the claim.)*
- **Non-retrieval baselines**: `GPT-4o-ZS`, `GPT-4o-CoT`; **Oracle** (gold passages) upper bound.

### 5.3 Protocol
- Months 1–6 = dev (prompt tuning, threshold selection: BERTScore τ=0.72/0.70 — reconcile the two
  values in method vs analysis; contamination overlap θ=0.4). Months 7–12 = held-out test; all main
  results from test window.
- Three-tier contamination detection: C4 8-gram overlap ≥0.4 flag, closed-book parametric-hit tagging,
  5% human spot-check (3 annotators, majority vote).
- Question generation via GPT-4o structured prompt (Appendix B — currently a stub, must be finalized);
  grounding via BERTScore F1 ≥ threshold.

### 5.4 Seeds, metrics, significance
- **Seeds**: ≥3 (paper reports std over 3 question-sampling orderings). Recommend ≥5 for stable CIs.
- **Metrics**: EM, Token F1, P_cite, R_cite, temporal-degradation slope β (OLS, Eq. `eq:degradation`),
  Knowledge-Cutoff-Gap, composite freshness score `F = F1 · P_cite · e^(−λ·age)` (λ=0.041), Kendall τ
  rank correlation (Eq. `eq:rank-tau`).
- **Significance / CIs**: paired bootstrap (n=1,000) for the † markers in T2/T3; 95% bootstrap CIs on
  all aggregates; one-sided t-test on OLS residuals with **Benjamini–Hochberg** correction for the
  slopes; binomial test for the 38% rank-reversal rate; Cohen's d effect sizes (claimed 0.62–1.14);
  bootstrap test on the ρ-difference for the freshness-score-vs-human claim.
- **Human studies**: inter-annotator κ for verification (0.73/0.76), error-taxonomy κ=0.74,
  change-type-classifier validation κ=0.71, and the 24-annotator "answer currency" rating study
  (ρ=0.81 vs F1 0.63, P_cite 0.71) — all currently synthetic; must be really collected.

### 5.5 Theory / formal-claim evidence (appendix C)
The proofs in [`appendix/c_protocol_proofs.tex`](appendix/c_protocol_proofs.tex) are **protocol-level**,
not empirical, so they don't need experiments to be *true* — but they impose obligations the rebuild
must satisfy:
- **Prop. 1 (ledger completeness)**: every aggregate claim must name a table/figure id, and every
  table/figure cell must name a replay-tuple set. → `results/claim_ledger.csv` must be expanded to map
  *each* numeric claim to its source set (today it lists only 12 generic placeholder rows).
- **Prop. 2 (cascade cost bound `c_l + q·c_h`)**: only meaningful if a cascade/selective-escalation
  variant is actually run; otherwise remove or mark clearly as a general protocol statement.
- **Timestamp invariant `t_s ≤ t_c ≤ t_e`**: must be enforced and verified per item in the build.

---

## 6. Replacement procedure (step-by-step)

### (a) Run experiments, dump to `results/*.json`
Run §5 experiments and write one JSON per logical result (schema names in §4). Define a real schema —
the current `results/main_results.json` schema (`rows: [{method, quality, risk, cost}]`) is a generic
placeholder and must be replaced. Suggested `main_results.json` schema:
```json
{"metric_window": "test_m7_12", "seeds": [0,1,2],
 "rows": [{"retriever":"ColBERT","generator":"GPT-4o",
           "static_f1":{"mean":80.1,"std":0.7,"ci95":[...]},
           "fresh_f1":{"mean":71.3,"std":1.2,"ci95":[...]},
           "static_pcite":{...},"fresh_pcite":{...},
           "delta_f1":-8.8,"sig_vs_all":true}]}
```
Store per-seed and per-example replay tuples `(x_i, y_is, z_is, c_is)` (appendix C Assumption 1).

### (b) Refactor the scripts to READ `results/*.json` instead of simulating
- [`scripts/make_artifacts.py`](scripts/make_artifacts.py): **delete the hardcoded `DATA` dict
  (line 13)** and the entire fixed-seed-simulator path. This file currently *writes* the synthetic
  JSON/CSV and orphaned PDFs; repurpose it (or replace it) to *aggregate real per-seed run logs* into
  the schemas in §4, compute means/std/CIs/significance, and write `results/*.json` +
  `results/claim_ledger.csv`. Remove every use of the literal
  `'Synthetic example result - replace before submission'` (the `synthetic_label` key).
- [`scripts/generate_figures.py`](scripts/generate_figures.py): **replace every hardcoded numpy array**
  (`hybrid_f1`, `colbert_f1`, `bm25_llama_f1`, `ablation_scores` line 121, `tau_values` line 163,
  `observed_*` lines 207–209, etc.) with `json.load()` of the corresponding `results/*.json`. Also
  reconcile the figure set: this script's figures (decay panels, τ-line, calibration) do **not** match
  the paper's inline TikZ figures.

### (c) Regenerate figures — and decide the figure pipeline
Two viable paths; pick one and apply consistently:
- **Path A (lowest drift):** keep the five inline TikZ figures but auto-generate their coordinate blocks
  from `results/*.json` via a small templating step, so F1–F5 can never disagree with T2–T5.
- **Path B:** convert F1–F5 to `\includegraphics` of real matplotlib PDFs produced by
  `generate_figures.py`, and delete/repurpose the four orphaned `figures/*.pdf`. Either way, ensure the
  PDFs that ship are the real ones and the four placeholder PDFs are removed.

### (d) Replace hardcoded inline numbers in each `.tex` table (relist)
For each table below, overwrite the literal numbers with values pulled from the matching results file.
**Strongly recommended: auto-generate these tables** (e.g., `pgfplotstable` reading a CSV, or a build
step emitting `.tex` table bodies from JSON) to eliminate the drift that already exists between prose,
tables, and the `results/` files.
- T1 `tab:dataset-stats` — [`sections/05_results.tex:13–27`](sections/05_results.tex)
- T2 `tab:main` — [`sections/05_results.tex:42–60`](sections/05_results.tex)
- T3 `tab:per-condition` — [`sections/05_results.tex:110–124`](sections/05_results.tex)
- T4 `tab:time-decay` — [`sections/05_results.tex:178–188`](sections/05_results.tex)
- T5 `tab:ablation` — [`sections/06_analysis.tex:13–28`](sections/06_analysis.tex)
- T6 `tab:error-analysis` — [`sections/06_analysis.tex:46–62`](sections/06_analysis.tex)
- T7 `tab:efficiency` — [`sections/06_analysis.tex:80–94`](sections/06_analysis.tex)
- T8 `tab:bertscore-threshold` — [`sections/06_analysis.tex:196–206`](sections/06_analysis.tex)
- T9 `tab:leakage` — [`sections/06_analysis.tex:222–229`](sections/06_analysis.tex)
- T10/T11 extended tables — [`appendix/d_extended_tables.tex`](appendix/d_extended_tables.tex)
  (replace placeholder vocabulary or delete)
- T12/T13 qualitative — [`appendix/e_qualitative_examples.tex`](appendix/e_qualitative_examples.tex)
  (swap fabricated transcripts for real replay logs)
- Inline TikZ figure data: F1–F5 coordinate blocks (locations in §3).

Also fix every **in-prose number** that restates a table value: abstract (3,847/2,291; 71.3/80.1;
−13.4; 53–61; +4.9; 38%; ρ=0.81/0.63), §1 contributions (lines 4, 11, 12), §5 narrative (lines 63–67,
191, 193, 226, 234, 236, 240), §6 narrative (lines 31–35, 65–69, 99, 134–136, 177–179, 183, 209, 232),
§9 conclusion (note: conclusion says **4.8** F1 inflation vs abstract's **4.9**, and "2.3× faster"
citation decay that appears only in §2/§9 — must be produced and reconciled).

### (e) Update claim_ledger.csv and re-verify every numeric claim
[`results/claim_ledger.csv`](results/claim_ledger.csv) currently has 12 generic placeholder rows
(`Control 47.6`, `Cheap baseline`, etc.) unrelated to the paper. Rebuild it as: `claim_id, value, CI,
table_or_figure_label, replay_tuple_set, status` covering **every** numeric claim in prose, tables,
figures, and in the review-response notes ([`reviews/author_response.md`](reviews/author_response.md),
[`reviews/round2_author_response.md`](reviews/round2_author_response.md),
[`notes/revision_execution_report.md`](notes/revision_execution_report.md)) — those response docs cite
numbers that must match the final results. Status must change from the synthetic label to a real
provenance pointer (satisfying appendix C Prop. 1).

### (f) Recompile
Rebuild `main.pdf` (latexmk). Fix the broken refs found in §7 first, then confirm no `??` references,
no undefined citations (e.g., `golchin2024time`), and no remaining `\syntheticlabel{}` / `\todoexp{}`
markers.

---

## 7. Consistency & rigor checks

**Existing inconsistencies that MUST be resolved (found during analysis):**
1. **Evaluation window**: setup ([`04_experimental_setup.tex:5`](sections/04_experimental_setup.tex))
   says "Jan–Dec 2024, 12-month window, 1,847 primary, 412 human-verified, κ=0.73, 1,731 retained";
   results ([`05_results.tex:6`](sections/05_results.tex)) say "26 months Jan 2023–Mar 2025, 3,847
   Static + 2,291 Fresh, 467 human-verified, κ=0.76". These describe two different benchmarks.
2. **Contamination-inflation value**: abstract & §5 say **+4.9** F1 (CI 3.9–5.5); conclusion
   ([`09_conclusion.tex:4`](sections/09_conclusion.tex)) says **4.8**. The orphaned figure script uses
   **77.1 vs 72.3 → +4.8**; T5 uses **76.2 vs 71.3 → +4.9**.
3. **Broken cross-references**: `\ref{tab:dataset-matrix}` used in
   [`03_method.tex:13`](sections/03_method.tex) and [`04_experimental_setup.tex:5`](sections/04_experimental_setup.tex)
   — no such label exists (actual: `tab:dataset-stats`). `\ref{tab:error-taxonomy}` in
   [`e_qualitative_examples.tex:4`](appendix/e_qualitative_examples.tex) — actual label is
   `tab:error-analysis`. Will render as `??`.
4. **Undefined citation**: `\citep{golchin2024time}` ([`06_analysis.tex:214`](sections/06_analysis.tex))
   — `references.bib` defines only `golchin2023time`.
5. **BERTScore threshold**: method uses **τ=0.72** ([`03_method.tex:10`](sections/03_method.tex)) and
   setup says 0.72; analysis §6.7 / T8 use **θ=0.70** as the default. Reconcile.
6. **System-set mismatch**: T2 evaluates 4 retrievers (no Hybrid); setup defines 5 (incl. Hybrid);
   appendix E "best baseline" is Hybrid+GPT-4o; orphaned fig1 plots BM25+**Llama-3-8B**. Decide the
   canonical system set.
7. **Sample-size mismatch**: §6 error analysis says **400** failed predictions
   ([`06_analysis.tex:39`](sections/06_analysis.tex)) but appendix E says "**300**-sample error
   analysis" ([`e_qualitative_examples.tex:36`](appendix/e_qualitative_examples.tex)).
8. **`results/dataset_matrix.csv`** (1,862 / 386 / 513 / 913) disagrees with both the setup prose and T1.

**Standard rigor gate (apply after rebuild):**
- paper == results == figures: every number in `.tex` equals the value in its `results/*.json`
  (ideally enforced by auto-generation).
- CIs + ≥3 (preferably ≥5) seeds on every aggregate; significance tests actually computed (paired
  bootstrap, BH-corrected t-tests, binomial, Cohen's d) rather than asserted.
- Dataset licenses, checksums, and a `MANIFEST` recorded; the timestamp invariant
  `t_s ≤ t_c ≤ t_e` verified per item; no Fresh item leaks into the Static control.
- Remove every "Synthetic example result - replace before submission" marker: the `\syntheticlabel`
  macro ([`main.tex:26`](main.tex)) and its uses in [`appendix/a_reproducibility.tex`](appendix/a_reproducibility.tex),
  [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex),
  [`appendix/c_protocol_proofs.tex`](appendix/c_protocol_proofs.tex),
  [`appendix/d_extended_tables.tex`](appendix/d_extended_tables.tex); the `synthetic_label` keys in all
  `results/*.json`; and the `status` column in `results/claim_ledger.csv`.
- Finalize the stub prompt/annotation templates in
  [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex) (currently generic
  scaffolding, not the actual FreshBench prompts).

---

## 8. Resource estimate

This is a deliberately **CPU-only + API-only** ("cheap") design — no GPU required
([`04_experimental_setup.tex:20–21`](sections/04_experimental_setup.tex), confirmed by
[`notes/source_note.md`](notes/source_note.md): 32 GB RAM preferred, 100–200 GB storage, moderate
API/annotation budget, 16–20 week timeline).

- **Compute**: single 16-core CPU workstation, 32–64 GB RAM, ~100–200 GB storage for monthly corpora
  + FAISS/ColBERT indices. One monthly eval cycle ≈ 4.5 h wall-clock (claimed); a 2,291-item Fresh
  eval ≈ 2.2 h.
- **API**: GPT-4o ≈ $0.027/item; full 12-month window claimed at **$214** (GPT-4o $183 + Claude Haiku
  $31); 26-month rolling ≈ **$1,612**. Add question-generation + judge + change-type classifier +
  freshness-score human-study scaffolding calls — budget **~$300–$2,000** depending on the chosen window
  and whether the full 5×3 model grid is run. Local Llama-3-8B via llama.cpp adds CPU time, no API cost.
- **Annotation effort**: 5% spot-check on each monthly slice (3 annotators), 412–467-item human
  verification, 400-item error taxonomy (2 annotators), change-type validation, and a **24-annotator
  answer-currency rating study**. At $18/hr (per ethics statement), estimate **~$1.5–4k** in
  annotation, plus recruiting/IRB overhead. This is the dominant cost/risk, not compute.
- **Overall**: **Low–Medium** — feasible on a laptop/workstation; the binding constraints are API
  budget and human annotation throughput, not hardware.

---

## 9. Risks & mitigations (ordered by severity)

1. **Entire numeric layer is synthetic and disconnected.** Tables read `.tex` literals, figures read
   TikZ literals, `results/*.json` hold an unrelated placeholder schema, and four PDFs are orphaned.
   *Mitigation:* §6 rebuild — wire experiments→JSON→tables/figures and auto-generate to prevent
   re-drift.
2. **Human-study claims are fabricated** (κ values, ρ=0.81 answer-currency, 24-annotator study,
   error-taxonomy κ=0.74). These are central to the paper's contribution and cannot be cheaply faked.
   *Mitigation:* budget and run real annotation (§8); if infeasible, soften/remove the freshness-vs-human
   claim.
3. **Self-internal inconsistencies (8 listed in §7)** will be caught by any careful reviewer and signal
   fabrication. *Mitigation:* resolve all before recompile; add a CI check that prose numbers match
   `results/`.
4. **Dataset licensing / redistribution** (Reuters/AP/BBC). Paper claims academic licenses + no raw-text
   redistribution. *Mitigation:* verify each outlet's terms, release only QA pairs + URLs + metadata +
   checksums; favor CC-BY-SA Wikipedia and public-domain sources where possible.
5. **Contamination-detection validity** depends on a 5% C4 sample approximating closed-model
   pretraining data — an approximation the paper itself flags. *Mitigation:* report it as a proxy with
   sensitivity analysis; don't overclaim "genuine retrieval" attribution without the closed-book +
   human checks actually run.
6. **Question-generator/evaluator overlap** (GPT-4o generates and is evaluated). *Mitigation:* keep the
   parametric-hit exclusion + human spot-check; consider a non-GPT-4o question generator for an unbiased
   subset.
7. **Indexing-latency / temporal-ambiguity noise** on very fresh items can dominate error bars.
   *Mitigation:* the explicit indexing-lag flag and 72-hour cutoff discussed in appendix E; report CIs.

---

## 10. Execution checklist

- [ ] Create `configs/*.yaml` (datasets, models, retrieval, eval, thresholds) — currently absent.
- [ ] **Resolve the 8 internal inconsistencies in §7** and fix broken refs (`tab:dataset-matrix`,
      `tab:error-taxonomy`) and the undefined `golchin2024time` citation.
- [ ] Decide canonical evaluation window (12-mo vs 26-mo) and system set (4 vs 5 retrievers, GPT-4o-only
      vs full 5×3 grid).
- [ ] Build monthly corpora (Reuters/BBC/AP + Wikipedia + BEIR), dedup (MinHash 0.8), record
      licenses/checksums/MANIFEST, enforce `t_s ≤ t_c ≤ t_e`.
- [ ] Generate + ground-verify questions (finalize Appendix B prompts); run 3-tier contamination filter.
- [ ] Run main RAG eval (T2/F1), per-condition (T3/F2), time-decay+OLS (T4/F3), ablation (T5), efficiency
      (T7), BERTScore sweep (T8), leakage audit (T9), k-sweep (F4), cross-domain (F5), calibration (P4) —
      all with ≥3 (target ≥5) seeds.
- [ ] Run human studies: verification (κ), error taxonomy (400 items, κ), change-type validation (κ),
      24-annotator answer-currency rating (ρ).
- [ ] Compute CIs + significance (paired bootstrap, BH-corrected t-tests, binomial, Cohen's d).
- [ ] Define real `results/*.json` schemas; dump per-seed + replay tuples.
- [ ] Refactor `scripts/make_artifacts.py` (delete `DATA` dict / simulator; aggregate real logs) and
      `scripts/generate_figures.py` (replace all hardcoded arrays with `json.load`).
- [ ] Regenerate figures (pick Path A inline-from-JSON or Path B includegraphics); delete orphaned PDFs.
- [ ] Replace every inline table number (T1–T13) and every in-prose number; auto-generate tables.
- [ ] Rebuild `results/claim_ledger.csv` with real provenance (satisfy appendix C Prop. 1); reconcile
      numbers cited in `reviews/` response notes.
- [ ] Remove all `\syntheticlabel{}` uses, `synthetic_label` JSON keys, and the synthetic `status` column.
- [ ] Recompile `main.pdf`; verify no `??`, no undefined citations, no synthetic markers remain.
- [ ] Final consistency pass: paper == results == figures == claim_ledger == review responses.
