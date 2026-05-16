"""HateMM annotation loader: yields one VideoRecord per row of the CSV."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class VideoRecord:
    video_id: str                              # filename stem
    path: Path                                 # absolute path to the .mp4
    label: int                                 # 1 = hate, 0 = non-hate
    community: str | None                      # target community (may be None)
    hate_segments: tuple[tuple[float, float], ...]  # (start_s, end_s) pairs; empty for non-hate


# Column-name candidates seen in HateMM annotation CSV variants.
# The first match wins. Edit here if your local CSV differs.
_COLS = {
    "file_name": ("video_file_name", "file_name", "filename"),
    "label":     ("label", "is_hate", "class"),
    "community": ("target_community", "community", "target"),
    "segments":  ("hateful_segments", "segments", "hate_segments"),
}


def load_records(root: Path, annotation_csv: Path) -> list[VideoRecord]:
    df = pd.read_csv(annotation_csv)
    col = {k: _pick(df.columns, opts) for k, opts in _COLS.items()}

    records: list[VideoRecord] = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        file_name = d[col["file_name"]]
        label = _normalize_label(d[col["label"]])
        subdir = "hate_videos" if label == 1 else "non_hate_videos"
        path = root / subdir / file_name

        community = d.get(col["community"]) if col["community"] else None
        if isinstance(community, float) and pd.isna(community):
            community = None

        segments = _parse_segments(d.get(col["segments"]) if col["segments"] else None)

        records.append(VideoRecord(
            video_id=Path(file_name).stem,
            path=path,
            label=label,
            community=community if isinstance(community, str) else None,
            hate_segments=tuple(segments),
        ))
    return records


def _pick(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _normalize_label(value) -> int:
    if isinstance(value, (int, float)):
        return int(bool(value))
    s = str(value).strip().lower()
    if s in {"hate", "1", "true", "yes", "hateful"}:
        return 1
    return 0


def _parse_segments(raw) -> list[tuple[float, float]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    else:
        parsed = raw
    out: list[tuple[float, float]] = []
    for item in parsed or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((float(item[0]), float(item[1])))
    return out
