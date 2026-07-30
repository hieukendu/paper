"""Generate final-system manuscript figures from promoted comparison JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "answer" / "final_best_tuned" / "final_comparison.json"
OUTPUT = ROOT / "manuscript" / "latex" / "figures"
LABELS = [
    ("implicit_sentiment", "Implicit"),
    ("sarcasm", "Sarcasm"),
    ("irony", "Irony"),
    ("idiom_figurative", "Idiom/fig."),
    ("code_switching", "Code-switch."),
    ("mocking", "Mocking"),
]


def load() -> list[dict]:
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def macro_comparison(rows: list[dict], final: dict) -> None:
    systems = [row for row in rows if row["regime"] == "best_tuned"] + [final]
    names = ["PhoBERT", "XLM-R", "Sailor", "Vistral", "Final"]
    values = [float(row["macro_pragmatic_f1"]) for row in systems]
    colors = ["#8da0cb"] * (len(systems) - 1) + ["#1b9e77"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("Macro pragmatic F1")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=0, labelsize=6)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    save(fig, "final_macro_comparison")


def label_gaps(rows: list[dict], final: dict) -> None:
    baselines = [row for row in rows if row["regime"] == "best_tuned"]
    # Compact category names and angled labels keep the six categories legible
    # at single-column width in the ACL review format.
    names = ["Implicit", "Sarcasm", "Irony", "Idiom", "Code", "Mocking"]
    gaps = [float(final[key]) - max(float(row[key]) for row in baselines) for key, _ in LABELS]
    colors = ["#1b9e77" if value >= 0 else "#d95f02" for value in gaps]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    bars = ax.bar(names, gaps, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("F1 difference (points)")
    ax.set_ylim(-1, 8)
    ax.tick_params(axis="x", labelrotation=25, labelsize=5.5)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.subplots_adjust(bottom=0.27)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, gaps):
        y = value + (0.18 if value >= 0 else -0.35)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    save(fig, "final_label_gaps")


def final_vs_leaders(rows: list[dict], final: dict) -> None:
    baselines = [row for row in rows if row["regime"] == "best_tuned"]
    names = [display for _, display in LABELS]
    leaders = [max(float(row[key]) for row in baselines) for key, _ in LABELS]
    values = [float(final[key]) for key, _ in LABELS]
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.bar([value - 0.19 for value in x], leaders, width=0.38, label="Best-tuned label leader", color="#8da0cb")
    ax.bar([value + 0.19 for value in x], values, width=0.38, label="ViPragSent-Final", color="#1b9e77")
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Binary macro-F1")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, "final_vs_label_leaders")


def main() -> None:
    rows = load()
    final = next(row for row in rows if row["system"] == "ViPragSent final continuation")
    macro_comparison(rows, final)
    label_gaps(rows, final)
    final_vs_leaders(rows, final)


if __name__ == "__main__":
    main()
