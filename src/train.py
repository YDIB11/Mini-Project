"""Train the fusion model (or any modality subset) on cached HateMM features.

Examples:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --modalities video
    python -m src.train --config configs/default.yaml --modalities video text --epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import detect_feature_dims, make_loader, make_splits
from .metrics import compute_metrics
from .models.fusion import MultimodalFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--modalities", nargs="+",
                        default=["video", "audio", "text"],
                        choices=["video", "audio", "text"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="override cfg.seed for model init / dataloader shuffle; "
                             "splits stay fixed to cfg.seed for fair multi-seed comparison")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_seed = args.seed if args.seed is not None else cfg["seed"]

    import random
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    modalities = tuple(args.modalities)
    run_name = args.run_name
    if run_name is None:
        run_name = "+".join(modalities)
        if args.seed is not None:
            run_name = f"{run_name}_seed{args.seed}"
    epochs = args.epochs or cfg["train"]["epochs"]
    bs = cfg["train"]["batch_size"]

    splits = make_splits(
        args.features_dir,
        train_frac=cfg["data"]["splits"]["train_frac"],
        val_frac=cfg["data"]["splits"]["val_frac"],
        test_frac=cfg["data"]["splits"]["test_frac"],
        seed=cfg["seed"],
    )
    n_train, n_val, n_test = len(splits["train"]), len(splits["val"]), len(splits["test"])
    print(f"splits: train={n_train} val={n_val} test={n_test}")

    train_loader = make_loader(args.features_dir, splits["train"], modalities, bs, shuffle=True)
    val_loader = make_loader(args.features_dir, splits["val"], modalities, bs, shuffle=False) if n_val else None

    dims = detect_feature_dims(args.features_dir, modalities)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  modalities={modalities}  feature_dims={dims}")

    model = MultimodalFusion(
        feature_dims=dims,
        d_model=cfg["model"]["fusion"]["d_model"],
        n_heads=cfg["model"]["fusion"]["n_heads"],
        n_layers=cfg["model"]["fusion"]["n_layers"],
        dropout=cfg["model"]["fusion"]["dropout"],
        n_classes=cfg["model"]["classifier"]["n_classes"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"{run_name}.pt"
    history_path = args.out_dir / f"{run_name}.history.json"

    best_metric = -1.0
    bad_epochs = 0
    patience = cfg["train"]["early_stop_patience"]
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["label"].shape[0]
            n_seen += batch["label"].shape[0]
        train_loss /= max(n_seen, 1)

        record = {"epoch": epoch, "train_loss": round(train_loss, 4)}
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, device)
            record.update({"val_" + k: round(v, 4) for k, v in val_metrics.items()})
            metric = val_metrics["macro_f1"]
        else:
            metric = -train_loss  # rough proxy when there is no val set
        history.append(record)
        print(json.dumps(record))

        if metric > best_metric:
            best_metric = metric
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "modalities": list(modalities),
                "dims": dims,
                "config": cfg,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"checkpoint: {ckpt_path}  best metric: {best_metric:.4f}")


@torch.inference_mode()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    labels, preds, probs = [], [], []
    for batch in loader:
        logits = model(batch)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        labels.append(batch["label"].numpy())
        preds.append(pred.cpu().numpy())
        probs.append(prob.cpu().numpy())
    return compute_metrics(
        np.concatenate(labels),
        np.concatenate(preds),
        np.concatenate(probs),
    )


if __name__ == "__main__":
    main()
