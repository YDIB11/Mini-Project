"""Extract a mono PCM WAV at a target sample rate from a video, via ffmpeg.

If the source video has no audio stream, write a short silent WAV instead so
downstream code (ASR, audio encoder) always has a file to read.
"""

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

    if not _has_audio_stream(video_path):
        _write_silence(out_path, sample_rate=sample_rate, mono=mono, duration_s=0.5)
        return out_path

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


def _has_audio_stream(video_path: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and "audio" in result.stdout


def _write_silence(out_path: Path, sample_rate: int, mono: bool, duration_s: float) -> None:
    layout = "mono" if mono else "stereo"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout={layout}:sample_rate={sample_rate}",
        "-t", str(duration_s),
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
