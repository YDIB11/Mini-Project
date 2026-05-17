"""Bar chart: accuracy and AUROC across the four models, multi-seed mean +/- std.

Outputs:
  paper/figures/ablation_bars.pdf  (for inclusion in main.tex)
  paper/figures/ablation_bars.png  (for poster use)

Numbers below are taken from results/_multiseed_summary.json (seeds 42, 1337, 2025).
If the summary file is present we re-read it; otherwise we use the hardcoded fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path("results/_multiseed_summary.json")
OUT_DIR = Path("paper/figures")

# Source: multi-seed aggregate over seeds {42, 1337, 2025} on HateMM test split (n=109).
FALLBACK = {
    "video":            {"acc_mean": 0.7217, "acc_std": 0.0295, "auroc_mean": 0.8081, "auroc_std": 0.0139},
    "audio":            {"acc_mean": 0.6697, "acc_std": 0.0242, "auroc_mean": 0.7659, "auroc_std": 0.0332},
    "text":             {"acc_mean": 0.7920, "acc_std": 0.0212, "auroc_mean": 0.8271, "auroc_std": 0.0158},
    "video+audio+text": {"acc_mean": 0.7920, "acc_std": 0.0522, "auroc_mean": 0.8759, "auroc_std": 0.0119},
}

DISPLAY_NAME = {
    "video": "Video",
    "audio": "Audio",
    "text": "Text",
    "video+audio+text": "V+A+T",
}

# Visual style. Matches the poster palette: red primary (#c41e1a) for the
# AUROC bar (the robust win), dark grey for the neutral accuracy bar.
ACC_COLOR   = "#404040"   # dark grey
AUROC_COLOR = "#c41e1a"   # EPFL/poster red
EDGE_COLOR  = "#1a1a1a"
GRID_COLOR  = "#999999"


def load_data() -> dict[str, dict[str, float]]:
    if SUMMARY_PATH.exists():
        raw = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        out = {}
        for key, info in raw.items():
            m = info["metrics"]
            out[key] = {
                "acc_mean":   m["accuracy"]["mean"],
                "acc_std":    m["accuracy"]["std"],
                "auroc_mean": m["auroc"]["mean"],
                "auroc_std":  m["auroc"]["std"],
            }
        return out
    return FALLBACK


def main() -> None:
    data = load_data()
    rows = [{"key": k, **v} for k, v in data.items()]
    rows.sort(key=lambda r: r["auroc_mean"])   # weakest -> strongest on AUROC

    labels      = [DISPLAY_NAME.get(r["key"], r["key"]) for r in rows]
    acc_means   = [r["acc_mean"]   for r in rows]
    acc_stds    = [r["acc_std"]    for r in rows]
    auroc_means = [r["auroc_mean"] for r in rows]
    auroc_stds  = [r["auroc_std"]  for r in rows]

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":        9,
        "axes.labelsize":   9,
        "xtick.labelsize":  9,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        "pdf.fonttype":     42,   # embed TrueType so text stays editable / crisp
        "ps.fonttype":      42,
    })

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(rows))
    w = 0.38
    err_kw = {"elinewidth": 0.8, "ecolor": EDGE_COLOR, "capsize": 3, "capthick": 0.8}

    ax.bar(x - w/2, acc_means,   w, yerr=acc_stds,   color=ACC_COLOR,
           edgecolor=EDGE_COLOR, linewidth=0.5, label="Accuracy", error_kw=err_kw)
    ax.bar(x + w/2, auroc_means, w, yerr=auroc_stds, color=AUROC_COLOR,
           edgecolor=EDGE_COLOR, linewidth=0.5, label="AUROC",    error_kw=err_kw)

    ax.set_ylim(0.5, 0.95)
    ax.set_yticks(np.arange(0.5, 0.96, 0.1))
    ax.set_ylabel("Test-set score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.yaxis.grid(True, linestyle="--", color=GRID_COLOR, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.tick_params(axis="both", color=EDGE_COLOR, width=0.8)

    ax.legend(loc="upper left", frameon=False, ncol=1,
              handlelength=1.4, handletextpad=0.5, borderaxespad=0.4)

    plt.tight_layout(pad=0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "ablation_bars.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_DIR / "ablation_bars.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"wrote {OUT_DIR / 'ablation_bars.pdf'} and .png")


if __name__ == "__main__":
    main()
