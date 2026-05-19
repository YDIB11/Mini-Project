"""Generate the macro-F1 bar chart for the poster's Validation box.

Reads _multiseed_summary.json from ../results/ and writes bar_chart.png.
Shows all 7 models (3 unimodal + 3 bimodal + 1 fusion) with ±1 std error bars.

Run:
    python bar_chart.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "results" / "_multiseed_summary.json"
OUT = HERE / "bar_chart.png"

# Display order: unimodal → bimodal → fusion (ascending complexity).
ORDER = [
    ("audio",           "Audio",     "uni"),
    ("video",           "Video",     "uni"),
    ("text",            "Text",      "uni"),
    ("video+audio",     "V+A",       "bi"),
    ("audio+text",      "A+T",       "bi"),
    ("video+text",      "V+T",       "bi"),
    ("video+audio+text","V+A+T\n(Fusion)", "tri"),
]

COLORS = {
    "uni":  "#9CA3AF",   # neutral gray — unimodal
    "bi":   "#6B7280",   # darker gray — bimodal
    "tri":  "#C41E1A",   # EPFL red   — full fusion
}
EDGE_COLORS = {
    "uni":  "#6B7280",
    "bi":   "#374151",
    "tri":  "#7a1110",
}


def main() -> None:
    with SUMMARY.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels, f1s, errs, colors, edges = [], [], [], [], []
    for key, label, tier in ORDER:
        m = data[key]["metrics"]["macro_f1"]
        labels.append(label)
        f1s.append(m["mean"])
        errs.append(m["std"])
        colors.append(COLORS[tier])
        edges.append(EDGE_COLORS[tier])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    bars = ax.bar(
        x, f1s,
        color=colors,
        edgecolor=edges,
        linewidth=0.8,
        yerr=errs,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "#374151"},
        width=0.65,
    )

    for bar, val, err in zip(bars, f1s, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.007,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Test macro-F1", fontsize=13)
    ax.set_ylim(0.58, 0.90)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax.set_title(
        "Modality ablation — HateMM test split (n = 109)\n"
        "mean ± std over 3 seeds {42, 1337, 2025}",
        fontsize=12,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLORS["uni"],  edgecolor=EDGE_COLORS["uni"],  label="Unimodal"),
        Patch(facecolor=COLORS["bi"],   edgecolor=EDGE_COLORS["bi"],   label="Bimodal"),
        Patch(facecolor=COLORS["tri"],  edgecolor=EDGE_COLORS["tri"],  label="Fusion (V+A+T)"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
