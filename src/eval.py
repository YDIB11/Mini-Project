"""Evaluate a trained checkpoint and print metrics + per-community breakdown.

Example:
    python -m src.eval --config configs/default.yaml --checkpoint checkpoints/video+audio+text.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import make_loader, make_splits
from .metrics import (
    compute_metrics, compute_metrics_ci,
    per_community_breakdown, per_community_breakdown_ci,
)
from .models.fusion import MultimodalFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    modalities = tuple(ckpt["modalities"])
    dims = ckpt["dims"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalFusion(
        feature_dims=dims,
        d_model=cfg["model"]["fusion"]["d_model"],
        n_heads=cfg["model"]["fusion"]["n_heads"],
        n_layers=cfg["model"]["fusion"]["n_layers"],
        dropout=cfg["model"]["fusion"]["dropout"],
        n_classes=cfg["model"]["classifier"]["n_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    splits = make_splits(
        args.features_dir,
        train_frac=cfg["data"]["splits"]["train_frac"],
        val_frac=cfg["data"]["splits"]["val_frac"],
        test_frac=cfg["data"]["splits"]["test_frac"],
        seed=cfg["seed"],
    )
    loader = make_loader(
        args.features_dir, splits[args.split], modalities,
        batch_size=cfg["train"]["batch_size"], shuffle=False,
    )

    labels, preds, probs, communities = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(dim=-1)
            labels.append(batch["label"].numpy())
            preds.append(pred.cpu().numpy())
            probs.append(prob.cpu().numpy())
            communities.extend(batch["community"])

    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)

    overall = compute_metrics(labels, preds, probs)
    overall_ci = compute_metrics_ci(labels, preds, probs)
    per_comm = per_community_breakdown(labels, preds, communities)
    per_comm_ci = per_community_breakdown_ci(labels, preds, communities)

    report = {
        "modalities": list(modalities),
        "split": args.split,
        "n": int(labels.shape[0]),
        "overall": {k: round(v, 4) for k, v in overall.items()},
        "overall_ci": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in overall_ci.items()},
        "per_community": {
            c: {k: round(v, 4) for k, v in m.items()}
            for c, m in per_comm.items()
        },
        "per_community_ci": {
            c: {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in m.items()}
            for c, m in per_comm_ci.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
