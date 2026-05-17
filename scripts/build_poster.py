"""Rebuild the two results tables in poster/poster.html from multi-seed JSONs.

Reads:
  results/_multiseed_summary.json       (overall metrics, mean +/- std)
  results/_multiseed_per_community.json (per-community metrics, mean +/- std)

In-place replaces the HTML between markers in poster/poster.html:
  <!-- BEGIN_TABLE_1 --> ... <!-- END_TABLE_1 -->
  <!-- BEGIN_TABLE_2 --> ... <!-- END_TABLE_2 -->

Bolding rules (data-driven, no hardcoding):
  Table 1: a V+A+T cell is bolded iff V+A+T has the highest mean AND the gap
           to the second-best mean exceeds the combined std (i.e. a robust win).
  Table 2: per row, the cell with the highest mean is bolded.

Usage:
    python scripts/build_poster.py

After running, regenerate poster.pdf via headless Chrome if desired:
    chrome --headless --no-pdf-header-footer --print-to-pdf=poster/poster.pdf \\
           --no-margins file:///abs/path/to/poster/poster.html
"""

from __future__ import annotations

import json
import re
from pathlib import Path

POSTER_HTML    = Path("poster/poster.html")
SUMMARY_PATH   = Path("results/_multiseed_summary.json")
COMMUNITY_PATH = Path("results/_multiseed_per_community.json")

DISPLAY = {
    "video+audio+text": "V + A + T",
    "text":             "Text only",
    "video":            "Video only",
    "audio":            "Audio only",
}
T1_ORDER = ["video+audio+text", "text", "video", "audio"]   # rows top to bottom

T2_COMMUNITIES   = ["Blacks", "Others", "Jews"]             # rows in this order if present
T2_MODALITIES    = ["video+audio+text", "video", "audio", "text"]  # column order
T2_COLUMN_HEADERS = ["Fusion", "Video", "Audio", "Text"]

T1_BEGIN, T1_END = "<!-- BEGIN_TABLE_1 -->", "<!-- END_TABLE_1 -->"
T2_BEGIN, T2_END = "<!-- BEGIN_TABLE_2 -->", "<!-- END_TABLE_2 -->"


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} &plusmn; {std:.2f}"


def render_table_1(summary: dict) -> str:
    metrics = ["accuracy", "macro_f1", "precision", "recall", "auroc"]
    headers = ["Acc", "F1", "Prec", "Rec", "AUROC"]

    # Per-metric column statistics for the "robust V+A+T win" check.
    column_stats = {}
    for metric in metrics:
        vals = [(mod,
                 summary[mod]["metrics"][metric]["mean"],
                 summary[mod]["metrics"][metric]["std"])
                for mod in T1_ORDER if mod in summary]
        vals.sort(key=lambda x: -x[1])
        column_stats[metric] = vals  # ranked highest -> lowest

    rows_html = []
    for mod in T1_ORDER:
        if mod not in summary:
            continue
        m = summary[mod]["metrics"]
        cells = []
        for metric in metrics:
            mean, std = m[metric]["mean"], m[metric]["std"]
            ranked = column_stats[metric]
            top_mod, top_mean, top_std = ranked[0]
            sec_mod, sec_mean, sec_std = ranked[1] if len(ranked) > 1 else (None, 0.0, 0.0)
            robust_win = (
                mod == "video+audio+text"
                and mod == top_mod
                and (top_mean - sec_mean) > (top_std + sec_std)
            )
            cls = ' class="winner"' if robust_win else ""
            cells.append(f"<td{cls}>{fmt(mean, std)}</td>")
        row_class = ' class="fusion"' if mod == "video+audio+text" else ""
        cells_html = "\n              ".join(cells)
        rows_html.append(
            f'            <tr{row_class}>\n'
            f'              <td>{DISPLAY[mod]}</td>\n'
            f'              {cells_html}\n'
            f'            </tr>'
        )

    head_cells = "".join(f"<th>{h}</th>" for h in headers)
    return (
        '        <table class="results">\n'
        '          <caption>Table 1. Modality ablation on the test split (n = 109), '
        'mean &plusmn; std over 3 seeds {42, 1337, 2025}.</caption>\n'
        '          <thead>\n'
        f'            <tr><th>Modality</th>{head_cells}</tr>\n'
        '          </thead>\n'
        '          <tbody>\n'
        + "\n".join(rows_html) + "\n"
        '          </tbody>\n'
        '        </table>'
    )


def render_table_2(community: dict) -> str:
    rows_html = []
    for c in T2_COMMUNITIES:
        if c not in community:
            continue
        info = community[c]
        n = int(info["n"])

        # Compute highest-accuracy cell for this row.
        mods_data = [
            (mod,
             info["by_modality"][mod]["accuracy"]["mean"],
             info["by_modality"][mod]["accuracy"]["std"])
            for mod in T2_MODALITIES
            if mod in info["by_modality"]
        ]
        top_idx = max(range(len(mods_data)), key=lambda i: mods_data[i][1])

        cells = []
        for i, (_mod, mean, std) in enumerate(mods_data):
            cls = ' class="winner"' if i == top_idx else ""
            cells.append(f"<td{cls}>{fmt(mean, std)}</td>")
        cells_html = "\n              ".join(cells)
        rows_html.append(
            f'            <tr>\n'
            f'              <td>{c}</td><td>{n}</td>\n'
            f'              {cells_html}\n'
            f'            </tr>'
        )

    head_cells = "".join(f"<th>{h}</th>" for h in T2_COLUMN_HEADERS)
    return (
        '        <table class="results" style="margin-top: 4mm;">\n'
        '          <caption>Table 2. Per-target-community test accuracy (n &ge; 15), '
        'mean &plusmn; std over 3 seeds.</caption>\n'
        '          <thead>\n'
        f'            <tr><th>Community</th><th>n</th>{head_cells}</tr>\n'
        '          </thead>\n'
        '          <tbody>\n'
        + "\n".join(rows_html) + "\n"
        '          </tbody>\n'
        '        </table>'
    )


def replace_between(content: str, begin: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    new_block = f"{begin}\n{replacement}\n        {end}"
    new_content, n = pattern.subn(new_block, content)
    if n == 0:
        raise RuntimeError(f"Marker pair {begin!r}/{end!r} not found in poster.html")
    return new_content


def main() -> None:
    summary   = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    community = json.loads(COMMUNITY_PATH.read_text(encoding="utf-8"))

    html = POSTER_HTML.read_text(encoding="utf-8")
    html = replace_between(html, T1_BEGIN, T1_END, render_table_1(summary))
    html = replace_between(html, T2_BEGIN, T2_END, render_table_2(community))
    POSTER_HTML.write_text(html, encoding="utf-8")

    print(f"Rebuilt {POSTER_HTML} from")
    print(f"  {SUMMARY_PATH}")
    print(f"  {COMMUNITY_PATH}")


if __name__ == "__main__":
    main()
