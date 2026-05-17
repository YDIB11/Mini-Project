"""Generate the macro-F1 bar chart for the poster's Validation box.

Reads the four result JSONs in ../results/ and writes bar_chart.png next to
this file. Single-seed for now; when multi-seed numbers exist we can either
extend the JSONs with a "std" field or pass --std on the command line.

Run:
    python bar_chart.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "bar_chart.png"

# Order chosen to put fusion last so it lands on the right (eye-trailing position).
ORDER = [
    ("audio.json", "Audio"),
    ("video.json", "Video"),
    ("text.json", "Text"),
    ("video+audio+text.json", "Fusion"),
]

# EPFL red for the fusion bar, neutral gray for unimodal baselines.
COLORS = ["#9CA3AF", "#9CA3AF", "#9CA3AF", "#C41E1A"]


def load_f1(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        return float(json.load(f)["overall"]["macro_f1"])


def main() -> None:
    f1s = [load_f1(RESULTS / fname) for fname, _ in ORDER]
    labels = [label for _, label in ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    bars = ax.bar(labels, f1s, color=COLORS, edgecolor="black", linewidth=0.8)

    for bar, value in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    ax.set_ylabel("Test macro-F1", fontsize=14)
    ax.set_ylim(0.6, 0.88)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax.set_title("Modality ablation, HateMM test split (n = 109)", fontsize=14)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
