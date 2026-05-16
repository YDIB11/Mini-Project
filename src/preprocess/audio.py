"""Extract a mono PCM WAV at a target sample rate from a video, via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_audio(
    video_path: Path,
    out_path: Path,
    sample_rate: int = 16_000,
    mono: bool = True,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
