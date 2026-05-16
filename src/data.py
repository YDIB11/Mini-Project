"""Train/val/test split, collate function, and DataLoader factory for cached features."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .features.loader import HateMMFeatures


def make_splits(
    features_dir: Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Stratified split by label, cached to features/splits.json."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    split_path = features_dir / "splits.json"
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))

    entries = []
    with (features_dir / "index.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    ids = [e["video_id"] for e in entries]
    labels = [e["label"] for e in entries]

    ids_tv, ids_test = _split(ids, labels, test_frac, seed)
    y_tv = [labels[ids.index(i)] for i in ids_tv]
    val_rel = val_frac / (train_frac + val_frac)
    ids_train, ids_val = _split(ids_tv, y_tv, val_rel, seed)

    splits = {"train": ids_train, "val": ids_val, "test": ids_test}
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def _split(ids: list[str], labels: list[int], frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Stratified split with a non-stratified fallback for tiny datasets."""
    try:
        a, b, _, _ = train_test_split(
            ids, labels, test_size=frac, stratify=labels, random_state=seed,
        )
    except ValueError:
        a, b, _, _ = train_test_split(
            ids, labels, test_size=frac, random_state=seed,
        )
    return a, b


def collate(batch: list[dict]) -> dict:
    """Stack tensors; pad variable-length video sequences."""
    out: dict = {
        "label": torch.stack([b["label"] for b in batch]),
        "video_id": [b["video_id"] for b in batch],
        "community": [b["community"] for b in batch],
    }
    sample = batch[0]
    if "video" in sample:
        max_T = max(b["video"].shape[0] for b in batch)
        D = sample["video"].shape[1]
        videos = torch.zeros(len(batch), max_T, D)
        masks = torch.zeros(len(batch), max_T, dtype=torch.bool)
        for i, b in enumerate(batch):
            T = b["video"].shape[0]
            videos[i, :T] = b["video"]
            masks[i, :T] = True
        out["video"] = videos
        out["video_mask"] = masks
    if "audio" in sample:
        out["audio"] = torch.stack([b["audio"] for b in batch])
    if "text" in sample:
        out["text"] = torch.stack([b["text"] for b in batch])
        out["text_mask"] = torch.stack([b["text_mask"] for b in batch])
    return out


def make_loader(
    features_dir: Path,
    video_ids: list[str],
    modalities: tuple[str, ...],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    ds = HateMMFeatures(features_dir, modalities=modalities, video_ids=video_ids)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


def detect_feature_dims(features_dir: Path, modalities: tuple[str, ...]) -> dict[str, int]:
    """Read one .pt per modality to read off the embedding dim."""
    dims: dict[str, int] = {}
    for m in modalities:
        files = list((features_dir / m).glob("*.pt"))
        if not files:
            raise RuntimeError(f"no features found in {features_dir / m}")
        sample = torch.load(files[0], map_location="cpu", weights_only=True)
        dims[m] = int(sample["feat"].shape[-1])
    return dims
