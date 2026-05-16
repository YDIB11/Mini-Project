"""Compute frozen-encoder features for every video in the preprocess manifest.

One encoder is loaded at a time to keep GPU memory bounded. Output layout:

    features/
      video/<video_id>.pt   -> {"feat": (T_v, D_v)}
      audio/<video_id>.pt   -> {"feat": (T_a, D_a)}
      text/<video_id>.pt    -> {"feat": (T_t, D_t), "mask": (T_t,)}
      index.jsonl           -> one row per video: {video_id, label, community}

Usage:
    python -m src.features.extract --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--modalities", nargs="+",
                        default=["video", "audio", "text"],
                        choices=["video", "audio", "text"])
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = _load_manifest(args.cache_dir / "manifest.jsonl")
    print(f"{len(manifest)} usable records in manifest")

    skip = not args.no_skip_existing
    if "video" in args.modalities:
        _run_video(manifest, args.features_dir, cfg, skip)
    if "audio" in args.modalities:
        _run_audio(manifest, args.features_dir, cfg, skip)
    if "text" in args.modalities:
        _run_text(manifest, args.features_dir, cfg, skip)

    _write_index(manifest, args.features_dir)
    print(f"Done. Features in {args.features_dir}")


def _load_manifest(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("error") is None:
                out.append(entry)
    return out


def _run_video(manifest, features_dir: Path, cfg: dict, skip: bool) -> None:
    from .video import VideoEncoder
    enc = VideoEncoder(cfg["features"]["video"]["encoder"])
    out_dir = features_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in tqdm(manifest, desc="video"):
        out = out_dir / f"{e['video_id']}.pt"
        if skip and out.exists():
            continue
        frames = np.load(e["frames"])
        torch.save({"feat": enc.encode(frames)}, out)


def _run_audio(manifest, features_dir: Path, cfg: dict, skip: bool) -> None:
    from .audio import AudioEncoder
    a_cfg = cfg["features"]["audio"]
    enc = AudioEncoder(a_cfg["encoder"], target_tokens=int(a_cfg.get("target_tokens", 64)))
    out_dir = features_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in tqdm(manifest, desc="audio"):
        out = out_dir / f"{e['video_id']}.pt"
        if skip and out.exists():
            continue
        torch.save({"feat": enc.encode(Path(e["audio"]))}, out)


def _run_text(manifest, features_dir: Path, cfg: dict, skip: bool) -> None:
    from .text import TextEncoder
    t_cfg = cfg["features"]["text"]
    enc = TextEncoder(t_cfg["encoder"], max_length=int(t_cfg.get("max_length", 128)))
    out_dir = features_dir / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in tqdm(manifest, desc="text"):
        out = out_dir / f"{e['video_id']}.pt"
        if skip and out.exists():
            continue
        text = Path(e["transcript"]).read_text(encoding="utf-8")
        feat, mask = enc.encode(text)
        torch.save({"feat": feat, "mask": mask}, out)


def _write_index(manifest, features_dir: Path) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    with (features_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for e in manifest:
            f.write(json.dumps({
                "video_id": e["video_id"],
                "label": e["label"],
                "community": e["community"],
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
