"""Sample a fixed-length, center-cropped RGB frame sequence from a video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_frames(
    video_path: Path,
    fps: float,
    max_frames: int,
    size: int,
) -> np.ndarray:
    """Return a (T, size, size, 3) uint8 array of RGB frames, T <= max_frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(src_fps / fps)), 1)

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(_preprocess(frame, size))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def _preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    rgb = rgb[top:top + s, left:left + s]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
