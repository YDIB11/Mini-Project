"""Transcribe an audio file with faster-whisper. Model is loaded once and cached."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_model(model_size: str):
    from faster_whisper import WhisperModel
    import torch

    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(audio_path: Path, model_size: str = "small", language: str = "en") -> str:
    model = _load_model(model_size)
    segments, _ = model.transcribe(str(audio_path), language=language, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()
