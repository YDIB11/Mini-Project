"""Preprocess every HateMM video: cache frames, audio, transcript + a JSONL manifest.

Usage:
    python -m src.preprocess.run --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .asr import transcribe
from .audio import extract_audio
from .dataset import load_records
from .frames import extract_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N records (smoke test)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="re-do work even if cached output exists")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["data"]["root"])
    annotation_csv = Path(cfg["data"]["annotation_csv"])
    pp = cfg["preprocess"]

    frames_dir = args.cache_dir / "frames"
    audio_dir = args.cache_dir / "audio"
    text_dir = args.cache_dir / "transcripts"
    for d in (frames_dir, audio_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    records = load_records(root, annotation_csv)
    if args.limit:
        records = records[: args.limit]
    print(f"Processing {len(records)} videos -> {args.cache_dir}")

    skip = not args.no_skip_existing
    manifest_path = args.cache_dir / "manifest.jsonl"
    n_fail = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for rec in tqdm(records):
            entry: dict = {
                "video_id": rec.video_id,
                "label": rec.label,
                "community": rec.community,
                "frames": None,
                "audio": None,
                "transcript": None,
                "error": None,
            }
            try:
                frame_path = frames_dir / f"{rec.video_id}.npy"
                if not (skip and frame_path.exists()):
                    arr = extract_frames(
                        rec.path,
                        fps=pp["frames"]["fps"],
                        max_frames=pp["frames"]["max_frames"],
                        size=pp["frames"]["size"],
                    )
                    np.save(frame_path, arr)
                entry["frames"] = str(frame_path)

                audio_path = audio_dir / f"{rec.video_id}.wav"
                if not (skip and audio_path.exists()):
                    extract_audio(
                        rec.path,
                        audio_path,
                        sample_rate=pp["audio"]["sample_rate"],
                        mono=pp["audio"]["mono"],
                    )
                entry["audio"] = str(audio_path)

                text_path = text_dir / f"{rec.video_id}.txt"
                if not (skip and text_path.exists()):
                    text = transcribe(
                        audio_path,
                        model_size=pp["asr"]["model"],
                        language=pp["asr"]["language"],
                    )
                    text_path.write_text(text, encoding="utf-8")
                entry["transcript"] = str(text_path)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                n_fail += 1

            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
            manifest.flush()

    print(f"Done. Manifest: {manifest_path}  failures: {n_fail}/{len(records)}")


if __name__ == "__main__":
    main()
