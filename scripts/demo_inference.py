"""Quick inference demo for the screencast.

Loads our trained-architecture fusion model and runs a forward pass on a
single example with correctly-shaped dummy modality features:
  - video: 32 CLIP ViT-B/32 frame embeddings of size 512
  - audio: 64 wav2vec2-base mean-pooled tokens of size 768
  - text:  128 RoBERTa-base token embeddings of size 768

The model weights are randomly initialised here (no checkpoint loaded); the
purpose is to verify the inference path end to end. Real multi-seed metrics
on the HateMM test set are committed under results/.

Usage:
    python -m scripts.demo_inference
"""

from __future__ import annotations

import torch

from src.models.fusion import MultimodalFusion


def main() -> None:
    # Per-modality feature dimensions match those produced by the frozen encoders
    # in src/features/{video,audio,text}.py.
    feature_dims = {"video": 512, "audio": 768, "text": 768}

    model = MultimodalFusion(
        feature_dims=feature_dims,
        d_model=256,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        n_classes=2,
    ).eval()

    # One example: a single video's three modality streams.
    batch = {
        "video": torch.randn(1, 32, 512),
        "audio": torch.randn(1, 64, 768),
        "text":  torch.randn(1, 128, 768),
        "text_mask": torch.ones(1, 128, dtype=torch.bool),
    }

    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(probs.argmax().item())

    print("Input shapes per modality:")
    print(f"  video: {tuple(batch['video'].shape)}   (32 CLIP frame embeddings)")
    print(f"  audio: {tuple(batch['audio'].shape)}   (64 wav2vec2 mean-pooled tokens)")
    print(f"  text : {tuple(batch['text'].shape)}  (128 RoBERTa token embeddings)")
    print()
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    print(f"Output logits:        [{logits[0, 0].item():+.3f}, {logits[0, 1].item():+.3f}]")
    print(f"Class probabilities:  non-hate = {probs[0].item():.3f}   hate = {probs[1].item():.3f}")
    print(f"Predicted class:      {pred}  ({'hate' if pred == 1 else 'non-hate'})")


if __name__ == "__main__":
    main()
