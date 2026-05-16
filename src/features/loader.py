"""PyTorch Dataset over cached per-modality features."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class HateMMFeatures(Dataset):
    def __init__(
        self,
        features_dir: Path,
        modalities: tuple[str, ...] = ("video", "audio", "text"),
        video_ids: list[str] | None = None,
    ):
        self.features_dir = Path(features_dir)
        self.modalities = tuple(modalities)

        with (self.features_dir / "index.jsonl").open("r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]
        if video_ids is not None:
            keep = set(video_ids)
            entries = [e for e in entries if e["video_id"] in keep]
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        e = self.entries[i]
        item: dict = {
            "video_id": e["video_id"],
            "label": torch.tensor(e["label"], dtype=torch.long),
            "community": e["community"],
        }
        for m in self.modalities:
            payload = torch.load(
                self.features_dir / m / f"{e['video_id']}.pt",
                map_location="cpu",
                weights_only=True,
            )
            item[m] = payload["feat"]
            if "mask" in payload:
                item[f"{m}_mask"] = payload["mask"]
        return item
