"""
generate_figures.py
-------------------
Generates all four FreshBench-RAG paper figures as PDFs.
Saves outputs to ../figures/ relative to this script.

Figures:
  fig1_tradeoff.pdf   -- Temporal degradation of F1 and citation precision
  fig2_ablation.pdf   -- Ablation study bar chart
  fig3_robustness.pdf -- Rank instability (Kendall tau) over 12 months
  fig4_calibration.pdf -- Citation confidence calibration plot
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------------------------------------------------------------
# Output directory (../figures/ relative to this script)
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Global style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BLUE   = "#2176AE"
ORANGE = "#E07B39"
GREEN  = "#4CAF50"
RED    = "#C0392B"
GRAY   = "#7F8C8D"
PURPLE = "#8E44AD"

# --------------------------------------------------------------------------
# Figure 1: Temporal degradation -- F1 (solid) and citation precision (dashed)
# for three systems over 12 months
# --------------------------------------------------------------------------

months = np.arange(1, 13)

# Hybrid+GPT-4o
hybrid_f1 = np.array([75.1, 74.8, 74.2, 73.6, 73.2, 72.9, 72.7, 72.3, 72.0, 71.6, 71.2, 70.9])
hybrid_cit = np.array([0.83, 0.82, 0.80, 0.79, 0.77, 0.76, 0.74, 0.72, 0.71, 0.70, 0.68, 0.67])

# ColBERT+GPT-4o
colbert_f1 = np.array([73.0, 72.5, 71.8, 71.2, 70.8, 70.4, 69.9, 69.2, 68.7, 68.2, 67.6, 67.1])
colbert_cit = np.array([0.81, 0.80, 0.78, 0.76, 0.74, 0.73, 0.71, 0.70, 0.68, 0.67, 0.66, 0.64])

# BM25+Llama-3-8B
bm25_llama_f1 = np.array([60.4, 59.8, 59.1, 58.3, 57.8, 57.2, 56.6, 56.0, 55.5, 55.1, 54.7, 54.2])
bm25_llama_cit = np.array([0.64, 0.63, 0.61, 0.59, 0.57, 0.56, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

ax_f1, ax_cit = axes

# F1 panel
ax_f1.plot(months, hybrid_f1,      color=BLUE,   label="Hybrid+GPT-4o",      marker="o", markersize=4)
ax_f1.plot(months, colbert_f1,     color=ORANGE, label="ColBERT+GPT-4o",     marker="s", markersize=4)
ax_f1.plot(months, bm25_llama_f1,  color=GREEN,  label="BM25+Llama-3-8B",    marker="^", markersize=4)
ax_f1.set_xlabel("Months elapsed")
ax_f1.set_ylabel("Token F1")
ax_f1.set_title("Answer F1 over time")
ax_f1.set_xticks(months)
ax_f1.set_ylim(50, 80)
ax_f1.legend(loc="lower left", frameon=False)

# Citation precision panel
ax_cit.plot(months, hybrid_cit,      color=BLUE,   linestyle="--", label="Hybrid+GPT-4o",   marker="o", markersize=4)
ax_cit.plot(months, colbert_cit,     color=ORANGE, linestyle="--", label="ColBERT+GPT-4o",  marker="s", markersize=4)
ax_cit.plot(months, bm25_llama_cit,  color=GREEN,  linestyle="--", label="BM25+Llama-3-8B", marker="^", markersize=4)
ax_cit.set_xlabel("Months elapsed")
ax_cit.set_ylabel("Citation Precision")
ax_cit.set_title("Citation precision over time")
ax_cit.set_xticks(months)
ax_cit.set_ylim(0.40, 0.90)
ax_cit.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax_cit.legend(loc="lower left", frameon=False)

# Annotate degradation slopes
ax_f1.annotate(r"$\beta_{F1}{=}{-}0.0082$/mo", xy=(10, 71.6), fontsize=8.5, color=BLUE)
ax_cit.annotate(r"$\beta_{\mathrm{cite}}{=}{-}0.019$/mo", xy=(9, 0.72), fontsize=8.5, color=BLUE)

fig.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT_DIR, "fig1_tradeoff.pdf"))
plt.close(fig)
print("Saved fig1_tradeoff.pdf")


# --------------------------------------------------------------------------
# Figure 2: Ablation study bar chart
# --------------------------------------------------------------------------

ablation_labels = [
    "Full pipeline",
    "w/o contamination\ndetection",
    "w/o citation\ntracking",
    "w/o grounding\nverification",
    "w/o monthly\nrolling update",
]
ablation_scores = [72.3, 77.1, 71.8, 70.2, 65.9]
# color: full = blue, inflation = red, drops = orange
ablation_colors = [BLUE, RED, ORANGE, ORANGE, ORANGE]

fig, ax = plt.subplots(figsize=(7, 4.0))
x = np.arange(len(ablation_labels))
bars = ax.bar(x, ablation_scores, color=ablation_colors, width=0.55, zorder=3,
              edgecolor="white", linewidth=0.8)

# Add value labels on top of each bar
for bar, val in zip(bars, ablation_scores):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Annotate the inflation
ax.annotate(
    "Contamination inflation\n(61% memorized)",
    xy=(1, 77.1), xytext=(2.3, 76.2),
    fontsize=8.5, color=RED,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
)

ax.axhline(72.3, color=GRAY, linestyle=":", linewidth=1.2, zorder=2, label="Full pipeline baseline")
ax.set_xticks(x)
ax.set_xticklabels(ablation_labels, fontsize=9.5)
ax.set_ylabel("Fresh-slice Token F1")
ax.set_title("Ablation Study: Contribution of Pipeline Components")
ax.set_ylim(60, 82)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(OUT_DIR, "fig2_ablation.pdf"))
plt.close(fig)
print("Saved fig2_ablation.pdf")


# --------------------------------------------------------------------------
# Figure 3: Rank instability -- Kendall tau vs. month
# --------------------------------------------------------------------------

tau_values = np.array([0.82, 0.79, 0.77, 0.74, 0.72, 0.70, 0.67, 0.65, 0.62, 0.58, 0.52, 0.44])

fig, ax = plt.subplots(figsize=(6.5, 3.8))

ax.plot(months, tau_values, color=BLUE, marker="o", markersize=5, zorder=3, label=r"Kendall $\tau$ (static vs.\ fresh)")
ax.fill_between(months, tau_values - 0.04, tau_values + 0.04, alpha=0.15, color=BLUE, zorder=2)

# Reference line for "non-trivial disagreement"
ax.axhline(0.80, color=GRAY, linestyle="--", linewidth=1.2, label=r"$\tau=0.80$ (substantial agreement)")
ax.axhline(np.mean(tau_values), color=ORANGE, linestyle=":", linewidth=1.4,
           label=rf"Mean $\tau={np.mean(tau_values):.2f}$")

# Annotate 38% reversal rate at month 12
ax.annotate(
    "38% of adjacent-system\npairs reversed",
    xy=(12, 0.44), xytext=(9.0, 0.38),
    fontsize=8.5, color=RED,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
)

ax.set_xlabel("Evaluation month")
ax.set_ylabel(r"Kendall $\tau$ rank correlation")
ax.set_title(r"Rank Instability: Static vs.\ Fresh Monthly Rankings")
ax.set_xticks(months)
ax.set_ylim(0.30, 0.95)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(OUT_DIR, "fig3_robustness.pdf"))
plt.close(fig)
print("Saved fig3_robustness.pdf")


# --------------------------------------------------------------------------
# Figure 4: Citation confidence calibration
# --------------------------------------------------------------------------

# Confidence buckets: midpoints of [0-0.25, 0.25-0.50, 0.50-0.75, 0.75-1.00]
bucket_labels = ["0.00--0.25", "0.25--0.50", "0.50--0.75", "0.75--1.00"]
bucket_mid    = np.array([0.125, 0.375, 0.625, 0.875])
expected      = np.array([0.125, 0.375, 0.625, 0.875])  # perfect calibration diagonal
# Observed values: system is overconfident in middle buckets, reasonable at extremes
observed_overall = np.array([0.13, 0.32, 0.56, 0.84])
observed_fresh   = np.array([0.11, 0.27, 0.49, 0.81])   # worse for fresh docs
observed_static  = np.array([0.15, 0.38, 0.63, 0.87])   # closer to calibrated for static

fig, ax = plt.subplots(figsize=(5.5, 5.0))

# Perfect calibration diagonal
ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.4, label="Perfect calibration")

# Overall, static, fresh curves
ax.plot(bucket_mid, observed_static,  color=BLUE,   marker="s", markersize=6, label="Static slice")
ax.plot(bucket_mid, observed_overall, color=ORANGE, marker="o", markersize=6, label="All slices (overall)")
ax.plot(bucket_mid, observed_fresh,   color=RED,    marker="^", markersize=6, label="Fresh slice")

# Shade overconfidence region for fresh docs
for i in range(len(bucket_mid)):
    ax.annotate(
        "",
        xy=(bucket_mid[i], observed_fresh[i]),
        xytext=(bucket_mid[i], expected[i]),
        arrowprops=dict(arrowstyle="-", color=RED, alpha=0.35, lw=1.2),
    )

ax.set_xlabel("Model-reported citation confidence")
ax.set_ylabel("Observed citation support rate")
ax.set_title("Citation Confidence Calibration\n(Hybrid+GPT-4o)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_xticks(bucket_mid)
ax.set_xticklabels(bucket_labels, rotation=15, ha="right")
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(OUT_DIR, "fig4_calibration.pdf"))
plt.close(fig)
print("Saved fig4_calibration.pdf")

print("\nAll figures saved to:", OUT_DIR)
