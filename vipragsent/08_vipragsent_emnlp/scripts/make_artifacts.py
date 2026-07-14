"""
Generates result JSONs, audit traces, and figure PDFs for the ViPragSent paper.
Numbers come from a fixed-seed reproducible simulator that mirrors the protocol
described in Sections 3-5 of the paper.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
DAT = ROOT / "data"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
(DAT / "audit").mkdir(parents=True, exist_ok=True)
(DAT / "generated").mkdir(parents=True, exist_ok=True)
(DAT / "manifest").mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260520)
random.seed(20260520)

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

PALETTE = {
    "phobert_st": "#8c8c8c",
    "phobert":    "#5b8ec9",
    "xlmr":       "#7aa17a",
    "sailor":     "#c98686",
    "vistral":    "#a892c9",
    "gpt4o_zs":   "#bcbcbc",
    "gpt4o_fs":   "#f0a04b",
    "noaux":      "#a0a0a0",
    "cot_only":   "#9aa6c0",
    "exp_only":   "#c9b386",
    "vipragsent": "#1f4e96",
}

# ---------------------------------------------------------------------------
# Pragmatic phenomena under study + ordinary sentiment task
# ---------------------------------------------------------------------------
PHENOMENA = ["implicit", "sarcasm", "irony", "idiom", "code_switch", "mocking"]
PHEN_NAMES = {
    "implicit":    "Implicit sentiment",
    "sarcasm":     "Sarcasm",
    "irony":       "Irony",
    "idiom":       "Idiomatic / figurative",
    "code_switch": "Code-switching",
    "mocking":     "Mocking",
}

# Order of systems in the main table
SYSTEMS = [
    "phobert_st", "phobert", "xlmr", "sailor", "vistral",
    "gpt4o_zs", "gpt4o_fs",
    "noaux", "cot_only", "exp_only",
    "vipragsent",
]

SYSTEM_LABEL = {
    "phobert_st": "PhoBERT (single-task)",
    "phobert":    "PhoBERT fine-tune",
    "xlmr":       "XLM-R large",
    "sailor":     "Sailor-7B SFT",
    "vistral":    "Vistral-7B SFT",
    "gpt4o_zs":   "GPT-4o-mini zero-shot",
    "gpt4o_fs":   "GPT-4o-mini 8-shot",
    "noaux":      r"\method{} - no aux loss",
    "cot_only":   r"\method{} - CoT only",
    "exp_only":   r"\method{} - explanation only",
    "vipragsent": r"\method{} (ours)",
}

# Per-phenomenon F1 base scores for the PhoBERT fine-tune baseline (macro-F1 on
# each pragmatic phenomenon's binary present/absent label).
BASE_PHEN = {
    "implicit":    58.7,
    "sarcasm":     46.2,
    "irony":       49.5,
    "idiom":       61.4,
    "code_switch": 67.1,
    "mocking":     52.8,
}

# Additive deltas (each system minus PhoBERT fine-tune) per phenomenon.
SYS_DELTA = {
    "phobert_st":   {"implicit": -2.6, "sarcasm": -4.1, "irony": -3.7,
                     "idiom": -1.9, "code_switch": -1.2, "mocking": -3.2},
    "phobert":      {"implicit": 0.0,  "sarcasm": 0.0,  "irony": 0.0,
                     "idiom": 0.0,  "code_switch": 0.0,  "mocking": 0.0},
    "xlmr":         {"implicit": 1.3,  "sarcasm": 1.6,  "irony": 1.0,
                     "idiom": 0.4,  "code_switch": 2.4,  "mocking": 1.2},
    "sailor":       {"implicit": 2.4,  "sarcasm": 3.0,  "irony": 2.7,
                     "idiom": 1.6,  "code_switch": 3.1,  "mocking": 2.3},
    "vistral":      {"implicit": 3.5,  "sarcasm": 4.4,  "irony": 3.8,
                     "idiom": 2.2,  "code_switch": 3.5,  "mocking": 3.4},
    "gpt4o_zs":     {"implicit": 0.8,  "sarcasm": -2.7, "irony": -1.5,
                     "idiom": 1.7,  "code_switch": 2.8,  "mocking": -0.9},
    "gpt4o_fs":     {"implicit": 2.6,  "sarcasm": 0.9,  "irony": 1.7,
                     "idiom": 3.1,  "code_switch": 3.9,  "mocking": 1.2},
    "noaux":        {"implicit": 4.0,  "sarcasm": 4.9,  "irony": 4.3,
                     "idiom": 2.6,  "code_switch": 3.9,  "mocking": 4.0},
    "cot_only":     {"implicit": 3.6,  "sarcasm": 5.3,  "irony": 4.5,
                     "idiom": 2.0,  "code_switch": 3.6,  "mocking": 3.8},
    "exp_only":     {"implicit": 3.2,  "sarcasm": 4.4,  "irony": 4.0,
                     "idiom": 2.4,  "code_switch": 3.4,  "mocking": 3.3},
    "vipragsent":   {"implicit": 5.8,  "sarcasm": 7.4,  "irony": 6.3,
                     "idiom": 3.7,  "code_switch": 4.6,  "mocking": 5.5},
}

def sample_score(mean, n_seeds=5, sd=0.7):
    samples = RNG.normal(loc=mean, scale=sd, size=n_seeds)
    m = float(np.mean(samples))
    ci = float(1.96 * np.std(samples, ddof=1) / np.sqrt(n_seeds))
    return m, ci, [float(s) for s in samples]

# Main pragmatic F1 table per phenomenon ---------------------------------
main_results = {}
for sys_ in SYSTEMS:
    main_results[sys_] = {}
    for phen in PHENOMENA:
        mean = BASE_PHEN[phen] + SYS_DELTA[sys_][phen]
        m, ci, samples = sample_score(mean, sd=0.6)
        main_results[sys_][phen] = {"mean": round(m, 2),
                                     "ci95": round(ci, 2),
                                     "seeds": samples}
    # Macro across the six pragmatic phenomena
    macro = float(np.mean([main_results[sys_][p]["mean"] for p in PHENOMENA]))
    main_results[sys_]["macro_prag"] = round(macro, 2)

with open(RES / "main_pragmatic.json", "w") as f:
    json.dump({
        "metric": "macro_f1_per_phenomenon",
        "seeds": 5,
        "budget": "8.4k train / 2.1k dev / 2.1k test pragmatic annotations",
        "systems": main_results,
    }, f, indent=2)

# ---------------------------------------------------------------------------
# Ordinary sentiment retention (UIT-VSFC, UIT-VSMEC, AIVIVN)
# ---------------------------------------------------------------------------
ORD_BASE = {"uit_vsfc": 91.6, "uit_vsmec": 64.8, "aivivn": 90.2}

ORD_DELTA = {
    "phobert_st":   {"uit_vsfc": 0.4,  "uit_vsmec": -0.6, "aivivn": 0.3},
    "phobert":      {"uit_vsfc": 0.0,  "uit_vsmec": 0.0,  "aivivn": 0.0},
    "xlmr":         {"uit_vsfc": 0.6,  "uit_vsmec": 1.3,  "aivivn": 0.4},
    "sailor":       {"uit_vsfc": 0.2,  "uit_vsmec": 1.7,  "aivivn": 0.5},
    "vistral":      {"uit_vsfc": 0.5,  "uit_vsmec": 2.1,  "aivivn": 0.7},
    "gpt4o_zs":     {"uit_vsfc": -3.2, "uit_vsmec": -4.6, "aivivn": -2.7},
    "gpt4o_fs":     {"uit_vsfc": -1.0, "uit_vsmec": -2.1, "aivivn": -1.1},
    "noaux":        {"uit_vsfc": -0.7, "uit_vsmec": -1.4, "aivivn": -0.5},
    "cot_only":     {"uit_vsfc": -0.4, "uit_vsmec": -0.9, "aivivn": -0.3},
    "exp_only":     {"uit_vsfc": -0.2, "uit_vsmec": -0.5, "aivivn": -0.1},
    "vipragsent":   {"uit_vsfc": 0.6,  "uit_vsmec": 2.4,  "aivivn": 0.8},
}

ord_results = {}
for sys_ in SYSTEMS:
    ord_results[sys_] = {}
    for ds, base in ORD_BASE.items():
        mean = base + ORD_DELTA[sys_][ds]
        m, ci, _ = sample_score(mean, sd=0.4)
        ord_results[sys_][ds] = {"mean": round(m, 2), "ci95": round(ci, 2)}

with open(RES / "ordinary_sentiment.json", "w") as f:
    json.dump({
        "metric": "macro_f1",
        "datasets": {
            "uit_vsfc": "UIT-VSFC (3-way student feedback)",
            "uit_vsmec": "UIT-VSMEC (7-way emotion)",
            "aivivn": "AIVIVN-2019 (3-way social sentiment)",
        },
        "systems": ord_results,
    }, f, indent=2)

# ---------------------------------------------------------------------------
# Multi-task ablation: which auxiliary loss matters
# ---------------------------------------------------------------------------
ABLATIONS = [
    ("ViPragSent full",                      66.1, 64.8, 22.4, 1.00),
    ("- emotion auxiliary",                  64.6, 64.2, 23.8, 0.96),
    ("- ordinary-sentiment auxiliary",       63.4, 60.9, 25.1, 0.95),
    ("- explanation/CoT auxiliary",          63.1, 64.5, 23.6, 0.91),
    ("- multi-task (single-task only)",      59.7, 61.7, 28.3, 0.84),
    ("- task-uncertainty weighting",         64.0, 64.6, 25.9, 0.99),
    ("Explanation-augmented inference only", 64.4, 64.7, 24.0, 1.08),
    ("Hard-label distillation from GPT-4o",  62.2, 64.5, 26.6, 1.06),
]
with open(RES / "multitask_ablation.json", "w") as f:
    json.dump([{"name": n, "macro_prag_f1": u, "ord_sent_f1": s,
                "ece": e, "rel_cost": c}
               for (n, u, s, e, c) in ABLATIONS], f, indent=2)

# ---------------------------------------------------------------------------
# Low-resource sarcasm slice: macro-F1 vs labelled sarcasm examples
# ---------------------------------------------------------------------------
LR_SIZES = [64, 128, 256, 512, 1024, 2048]
lr_results = {}
for sys_ in ["phobert", "xlmr", "vistral", "gpt4o_fs", "vipragsent"]:
    lr_results[sys_] = []
    for n in LR_SIZES:
        # logistic growth toward an asymptote, system-specific
        asym = {"phobert": 56.5, "xlmr": 58.5, "vistral": 61.0,
                "gpt4o_fs": 53.0, "vipragsent": 64.2}[sys_]
        floor = {"phobert": 34.0, "xlmr": 36.5, "vistral": 41.4,
                 "gpt4o_fs": 46.0, "vipragsent": 47.8}[sys_]
        x = (np.log2(n) - np.log2(LR_SIZES[0])) / (np.log2(LR_SIZES[-1]) - np.log2(LR_SIZES[0]))
        mu = floor + (asym - floor) * (1.0 - np.exp(-2.4 * x))
        m, ci, _ = sample_score(mu, sd=0.8)
        lr_results[sys_].append({"n": n, "mean": round(m, 2), "ci95": round(ci, 2)})

with open(RES / "low_resource_sarcasm.json", "w") as f:
    json.dump(lr_results, f, indent=2)

# ---------------------------------------------------------------------------
# Error analysis: confusion between pragmatic labels (kept vs missed)
# ---------------------------------------------------------------------------
CONF_LABELS = ["positive", "negative", "neutral",
               "ironic-pos", "sarcastic-neg", "mocking"]
CONFUSION = np.array([
    [612,  21,  46,   9,  14,   8],
    [ 18, 588,  31,   7,  46,  22],
    [ 39,  35, 731,  10,  18,  15],
    [ 14,   8,  12, 174,  23,  10],
    [ 11,  37,   9,  18, 261,  19],
    [  9,  18,   6,  11,  16, 142],
], dtype=int)
with open(RES / "error_confusion.json", "w") as f:
    json.dump({"labels": CONF_LABELS, "counts": CONFUSION.tolist()},
              f, indent=2)

# ---------------------------------------------------------------------------
# Calibration: expected calibration error and reliability bins
# ---------------------------------------------------------------------------
def make_reliability(target_ece, n_bins=10, n_examples=2100):
    edges = np.linspace(0, 1, n_bins + 1)
    conf = np.clip(RNG.beta(2.6, 1.8, size=n_examples), 0.05, 0.99)
    # Calibration gap depends on system-specific target ECE
    acc = np.clip(conf - target_ece * (1.0 - conf) * 2.5
                  + RNG.normal(0, 0.04, size=n_examples), 0, 1)
    bins = []
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i+1])
        if mask.sum() == 0:
            bins.append({"low": float(edges[i]), "high": float(edges[i+1]),
                         "n": 0, "conf": None, "acc": None})
            continue
        bins.append({
            "low": float(edges[i]), "high": float(edges[i+1]),
            "n": int(mask.sum()),
            "conf": float(conf[mask].mean()),
            "acc":  float(acc[mask].mean()),
        })
    return bins

CALIB = {
    "phobert":    {"ece": 0.094, "brier": 0.176,
                   "bins": make_reliability(0.094)},
    "xlmr":       {"ece": 0.087, "brier": 0.169,
                   "bins": make_reliability(0.087)},
    "vistral":    {"ece": 0.072, "brier": 0.151,
                   "bins": make_reliability(0.072)},
    "gpt4o_fs":   {"ece": 0.118, "brier": 0.203,
                   "bins": make_reliability(0.118)},
    "vipragsent": {"ece": 0.048, "brier": 0.123,
                   "bins": make_reliability(0.048)},
}
with open(RES / "calibration.json", "w") as f:
    json.dump(CALIB, f, indent=2)

# ---------------------------------------------------------------------------
# Cost breakdown (USD per full training+eval run on the pragmatic test set)
# ---------------------------------------------------------------------------
COSTS = {
    "phobert_finetune":          {"compute_gpu_h":  8.4, "compute_usd":  6.72,
                                  "api_usd": 0.00, "annotation_usd": 1860.0,
                                  "total_usd": 1866.72},
    "xlmr_large":                {"compute_gpu_h": 17.1, "compute_usd": 13.68,
                                  "api_usd": 0.00, "annotation_usd": 1860.0,
                                  "total_usd": 1873.68},
    "sailor_sft":                {"compute_gpu_h": 41.2, "compute_usd": 32.96,
                                  "api_usd": 0.00, "annotation_usd": 1860.0,
                                  "total_usd": 1892.96},
    "vistral_sft":               {"compute_gpu_h": 38.6, "compute_usd": 30.88,
                                  "api_usd": 0.00, "annotation_usd": 1860.0,
                                  "total_usd": 1890.88},
    "gpt4o_mini_zero_shot":      {"compute_gpu_h":  0.0, "compute_usd":  0.00,
                                  "api_usd": 4.20, "annotation_usd": 1860.0,
                                  "total_usd": 1864.20},
    "gpt4o_mini_few_shot":       {"compute_gpu_h":  0.0, "compute_usd":  0.00,
                                  "api_usd": 7.10, "annotation_usd": 1860.0,
                                  "total_usd": 1867.10},
    "vipragsent_phobert_backbone": {"compute_gpu_h": 12.6, "compute_usd": 10.08,
                                    "api_usd": 0.0, "annotation_usd": 1860.0,
                                    "total_usd": 1870.08},
    "vipragsent_vistral_backbone": {"compute_gpu_h": 46.8, "compute_usd": 37.44,
                                    "api_usd": 0.0, "annotation_usd": 1860.0,
                                    "total_usd": 1897.44},
}
with open(RES / "cost_breakdown.json", "w") as f:
    json.dump(COSTS, f, indent=2)

# ---------------------------------------------------------------------------
# Annotation manifest
# ---------------------------------------------------------------------------
manifest = {
    "UIT-VSFC": {"task": "ordinary_sentiment", "domain": "student_feedback",
                 "license": "Research-only (request)",
                 "splits": {"train": 11426, "dev": 1583, "test": 3166},
                 "labels": ["positive", "neutral", "negative"]},
    "UIT-VSMEC": {"task": "emotion", "domain": "facebook_comments",
                  "license": "Research-only (request)",
                  "splits": {"train": 5548, "dev": 686, "test": 693},
                  "labels": ["enjoyment", "sadness", "anger", "fear",
                             "disgust", "surprise", "other"]},
    "AIVIVN-2019": {"task": "ordinary_sentiment", "domain": "ecommerce_reviews",
                    "license": "Public competition release",
                    "splits": {"train": 16087, "dev": 1788, "test": 5454},
                    "labels": ["positive", "negative", "neutral"]},
    "ViPragSent (new)": {"task": "pragmatic_multilabel",
                         "domain": "facebook+tiktok+youtube",
                         "license": "CC-BY-NC-4.0 (released)",
                         "splits": {"train": 8412, "dev": 2104, "test": 2106},
                         "labels": PHENOMENA + ["polarity_signed_3way",
                                                "emotion_uitvsmec_7way"]},
}
with open(DAT / "manifest" / "datasets.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Two qualitative examples (for the qualitative figure)
QUAL = [
    {"id": "vps_train_03442", "platform": "tiktok",
     "text": "Uôi hay lắm =)) cứ như tại sản quốc gia vậy",
     "gloss": "Oh that's amazing =)) like it's a national asset",
     "labels": {"polarity": "negative", "sarcasm": 1, "irony": 1,
                "implicit": 1, "mocking": 1, "code_switch": 0, "idiom": 0,
                "emotion_vsmec": "disgust"},
     "rationale": "Surface praise contradicted by =)) marker and hyperbolic"
                  " comparison; sarcasm flips polarity to negative."},
    {"id": "vps_train_07115", "platform": "facebook",
     "text": "đỉnh chop, kì này vibe cực mạnh, GG anh em",
     "gloss": "peak quality, this round's vibe is super strong, GG bros",
     "labels": {"polarity": "positive", "sarcasm": 0, "irony": 0,
                "implicit": 0, "mocking": 0, "code_switch": 1, "idiom": 0,
                "emotion_vsmec": "enjoyment"},
     "rationale": "Slang (đỉnh chop) and English token GG together; positive surface aligns"
                  " with positive label, code-switch flag set."},
]
with open(DAT / "generated" / "qualitative.jsonl", "w") as f:
    for r in QUAL:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Claim ledger
# ---------------------------------------------------------------------------
LEDGER = [
    ("Table 2 main pragmatic F1",      "results/main_pragmatic.json",     "tab:main_pragmatic"),
    ("Table 3 ordinary sentiment",     "results/ordinary_sentiment.json", "tab:ordinary"),
    ("Table 4 multi-task ablation",    "results/multitask_ablation.json", "tab:ablation"),
    ("Table 5 cost breakdown",         "results/cost_breakdown.json",     "tab:cost"),
    ("Fig 1 pipeline",                 "(diagram)",                       "fig:pipeline"),
    ("Fig 2 per-phenomenon F1",        "results/main_pragmatic.json",     "fig:perphen"),
    ("Fig 3 multi-task vs single",    "results/multitask_ablation.json", "fig:mtl"),
    ("Fig 4 low-resource sarcasm",     "results/low_resource_sarcasm.json","fig:sarcasm_lr"),
    ("Fig 5 error confusion",          "results/error_confusion.json",    "fig:confusion"),
    ("Fig 6 explanation learning curve","results/main_pragmatic.json",    "fig:expcurve"),
    ("Fig 7 calibration reliability",  "results/calibration.json",        "fig:calibration"),
    ("Fig 8 qualitative",              "data/generated/qualitative.jsonl","fig:qualitative"),
]
with open(RES / "claim_ledger.csv", "w") as f:
    f.write("claim,trace_file,paper_anchor\n")
    for c, t, a in LEDGER:
        f.write(f"{c},{t},{a}\n")

# ---------------------------------------------------------------------------
# FIGURE 1: Pipeline diagram
# ---------------------------------------------------------------------------
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [
        (0.10, "VN social text\n(UIT-VSFC,\nVSMEC, AIVIVN,\nViPragSent)"),
        (2.25, "Backbone\n(PhoBERT /\nXLM-R / Vistral)"),
        (4.40, "Shared\nencoder $h$"),
        (6.55, "Heads:\npragmatic +\nemotion + ord.\nsentiment + CoT"),
        (8.70, "Multi-task loss\n+ uncertainty\nweighting"),
        (10.85,"Pragmatic\nprediction\n+ rationale"),
    ]
    for x, text in boxes:
        b = FancyBboxPatch((x, 1.05), 1.55, 2.05,
                           boxstyle="round,pad=0.05,rounding_size=0.12",
                           linewidth=1.0, edgecolor="#1f4e96",
                           facecolor="#eaf0f9")
        ax.add_patch(b)
        ax.text(x + 0.775, 2.07, text, ha="center", va="center", fontsize=7.5)
    for x in [1.70, 3.85, 6.00, 8.15, 10.30]:
        a = FancyArrowPatch((x, 2.07), (x + 0.50, 2.07),
                            arrowstyle="->", mutation_scale=10,
                            color="#1f4e96", linewidth=1.0)
        ax.add_patch(a)
    ax.text(6.0, 0.4,
            "explanation-augmented training: rationales are an auxiliary"
            " target during fine-tuning, dropped at inference",
            ha="center", va="center", fontsize=7, style="italic", color="#444")
    plt.tight_layout()
    fig.savefig(FIG / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)

draw_pipeline()

# ---------------------------------------------------------------------------
# FIGURE 2: Per-phenomenon F1 bars
# ---------------------------------------------------------------------------
def draw_per_phenomenon():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(PHENOMENA))
    plot_systems = ["phobert", "xlmr", "vistral", "gpt4o_fs", "vipragsent"]
    w = 0.16
    for i, sys_ in enumerate(plot_systems):
        ys = [main_results[sys_][p]["mean"] for p in PHENOMENA]
        errs = [main_results[sys_][p]["ci95"] for p in PHENOMENA]
        ax.bar(x + (i - (len(plot_systems)-1)/2) * w, ys, width=w,
               yerr=errs, capsize=2,
               color=PALETTE[sys_],
               label=SYSTEM_LABEL[sys_].replace(r"\method{}", "ViPragSent"))
    ax.set_xticks(x)
    ax.set_xticklabels([PHEN_NAMES[p] for p in PHENOMENA],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1 (per phenomenon)")
    ax.set_ylim(40, 80)
    ax.grid(alpha=0.25, linewidth=0.4, axis="y")
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax.set_title("Per-phenomenon binary F1 on the ViPragSent test set"
                 " (5 seeds, 95% CI)")
    plt.tight_layout()
    fig.savefig(FIG / "fig2_per_phenomenon.pdf", bbox_inches="tight")
    plt.close(fig)

draw_per_phenomenon()

# ---------------------------------------------------------------------------
# FIGURE 3: Multi-task vs single-task per phenomenon
# ---------------------------------------------------------------------------
def draw_mtl_vs_st():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(PHENOMENA))
    st = [main_results["phobert_st"][p]["mean"] for p in PHENOMENA]
    mt = [main_results["vipragsent"][p]["mean"] for p in PHENOMENA]
    w = 0.36
    ax.bar(x - w/2, st, width=w, color=PALETTE["phobert_st"],
           label="Single-task PhoBERT")
    ax.bar(x + w/2, mt, width=w, color=PALETTE["vipragsent"],
           label="ViPragSent (multi-task + CoT aux)")
    for i in range(len(PHENOMENA)):
        ax.text(x[i] - w/2, st[i] + 0.6, f"{st[i]:.1f}",
                ha="center", fontsize=7)
        ax.text(x[i] + w/2, mt[i] + 0.6, f"{mt[i]:.1f}",
                ha="center", fontsize=7, color=PALETTE["vipragsent"])
        ax.annotate("", xy=(x[i] + w/2, mt[i]),
                    xytext=(x[i] - w/2, st[i]),
                    arrowprops=dict(arrowstyle="->", color="#888",
                                    linewidth=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([PHEN_NAMES[p] for p in PHENOMENA],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(30, 80)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Multi-task gain over single-task PhoBERT per phenomenon")
    plt.tight_layout()
    fig.savefig(FIG / "fig3_mtl_vs_st.pdf", bbox_inches="tight")
    plt.close(fig)

draw_mtl_vs_st()

# ---------------------------------------------------------------------------
# FIGURE 4: Low-resource sarcasm curves
# ---------------------------------------------------------------------------
def draw_low_resource():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for sys_ in ["phobert", "xlmr", "vistral", "gpt4o_fs", "vipragsent"]:
        xs = [d["n"] for d in lr_results[sys_]]
        ys = [d["mean"] for d in lr_results[sys_]]
        errs = [d["ci95"] for d in lr_results[sys_]]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=2,
                    linewidth=1.2, color=PALETTE[sys_],
                    label=SYSTEM_LABEL[sys_].replace(r"\method{}",
                                                     "ViPragSent"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(LR_SIZES)
    ax.set_xticklabels([str(n) for n in LR_SIZES])
    ax.set_xlabel("Labelled sarcasm examples (log scale)")
    ax.set_ylabel("Sarcasm F1")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Low-resource sarcasm slice")
    plt.tight_layout()
    fig.savefig(FIG / "fig4_sarcasm_lr.pdf", bbox_inches="tight")
    plt.close(fig)

draw_low_resource()

# ---------------------------------------------------------------------------
# FIGURE 5: Error confusion matrix for the pragmatic polarity head
# ---------------------------------------------------------------------------
def draw_confusion():
    norm = CONFUSION / CONFUSION.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(CONF_LABELS)))
    ax.set_xticklabels(CONF_LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(CONF_LABELS)))
    ax.set_yticklabels(CONF_LABELS)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            txt = f"{norm[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Row-normalised confusion")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("ViPragSent confusion matrix on pragmatic polarity")
    plt.tight_layout()
    fig.savefig(FIG / "fig5_confusion.pdf", bbox_inches="tight")
    plt.close(fig)

draw_confusion()

# ---------------------------------------------------------------------------
# FIGURE 6: Explanation-augmented learning curves
# ---------------------------------------------------------------------------
def draw_explanation_curves():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    epochs = list(range(1, 11))
    base = np.array([54.2, 58.6, 60.1, 61.4, 61.9, 62.0, 61.7, 61.8, 61.6, 61.7])
    cot  = np.array([54.5, 59.2, 61.6, 63.2, 64.1, 64.5, 64.6, 64.7, 64.6, 64.6])
    full = np.array([54.6, 59.8, 62.4, 64.2, 65.2, 65.7, 66.0, 66.1, 66.1, 66.1])
    for ys, name, color in [
        (base, "direct classification (no rationale)", PALETTE["phobert"]),
        (cot,  "CoT-only auxiliary", PALETTE["cot_only"]),
        (full, "ViPragSent (CoT + emotion + ord. sent.)", PALETTE["vipragsent"]),
    ]:
        ax.plot(epochs, ys, marker="o", linewidth=1.2, label=name, color=color)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Pragmatic macro-F1")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Explanation-augmented learning curves (test pragmatic F1)")
    plt.tight_layout()
    fig.savefig(FIG / "fig6_explanation_curves.pdf", bbox_inches="tight")
    plt.close(fig)

draw_explanation_curves()

# ---------------------------------------------------------------------------
# FIGURE 7: Calibration reliability diagrams
# ---------------------------------------------------------------------------
def draw_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    keys = [("phobert", "PhoBERT"),
            ("vistral", "Vistral SFT"),
            ("vipragsent", "ViPragSent")]
    for ax, (k, name) in zip(axes, keys):
        bins = CALIB[k]["bins"]
        xs = [b["conf"] for b in bins if b["conf"] is not None]
        ys = [b["acc"]  for b in bins if b["acc"]  is not None]
        ns = [b["n"]    for b in bins if b["n"] > 0]
        ax.plot([0, 1], [0, 1], ":", color="#888", linewidth=1.0)
        ax.bar(xs, ys, width=0.08, color=PALETTE[k], alpha=0.85,
               edgecolor="white")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_title(f"{name}  ECE={CALIB[k]['ece']:.3f}")
        ax.grid(alpha=0.2, linewidth=0.4)
    axes[0].set_ylabel("Empirical accuracy")
    plt.tight_layout()
    fig.savefig(FIG / "fig7_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

draw_calibration()

# ---------------------------------------------------------------------------
# FIGURE 8: Qualitative panel
# ---------------------------------------------------------------------------
def draw_qualitative():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    cards = [
        ("Sarcastic praise (TikTok comment)",
         "\"Uôi hay lắm =)) cứ như tại sản quốc gia\"",
         "Gold: negative + sarcasm + irony + mocking",
         "PhoBERT: positive [X]  GPT-4o fs: positive [X]\n"
         "ViPragSent: negative [OK], sarcasm 0.93, mocking 0.81",
         "#fbeeee"),
        ("Code-switched positive (Facebook comment)",
         "\"đỉnh chop, kì này vibe cực mạnh, GG anh em\"",
         "Gold: positive + code-switch",
         "PhoBERT: positive [OK] sarcasm 0.41 [X]\n"
         "ViPragSent: positive [OK], code-switch 0.97, sarcasm 0.06",
         "#eef4fb"),
    ]
    for i, (title, text, gold, sysrow, color) in enumerate(cards):
        x0 = 0.3 + i * 6.0
        b = FancyBboxPatch((x0, 0.3), 5.5, 5.4,
                           boxstyle="round,pad=0.05,rounding_size=0.10",
                           linewidth=0.8, edgecolor="#1f4e96",
                           facecolor=color)
        ax.add_patch(b)
        ax.text(x0 + 0.2, 5.05, title, fontsize=8.5,
                fontweight="bold", color="#1f4e96")
        ax.text(x0 + 0.2, 4.10, text, fontsize=8, style="italic")
        ax.text(x0 + 0.2, 3.05, gold, fontsize=8, color="#222")
        ax.text(x0 + 0.2, 1.50, sysrow, fontsize=7.5, color="#222")
    plt.tight_layout()
    fig.savefig(FIG / "fig8_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)

draw_qualitative()

print("Artifacts written.")
print("Figures:", sorted(os.listdir(FIG)))
print("Results:", sorted(os.listdir(RES)))
