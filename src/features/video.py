"""Frozen CLIP image encoder: produces a per-frame embedding sequence."""

from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPModel


class VideoEncoder:
    def __init__(self, name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPImageProcessor.from_pretrained(name)
        model = CLIPModel.from_pretrained(name)
        self.vision = model.vision_model.to(self.device).eval()
        self.projection = model.visual_projection.to(self.device).eval()
        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.projection.parameters():
            p.requires_grad = False
        self.dim = model.config.projection_dim

    @torch.inference_mode()
    def encode(self, frames_uint8: np.ndarray, batch_size: int = 32) -> torch.Tensor:
        """(T, H, W, 3) uint8 -> (T, D) float32 on CPU."""
        inputs = self.processor(images=list(frames_uint8), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = []
        for i in range(0, pixel_values.shape[0], batch_size):
            pooled = self.vision(pixel_values=pixel_values[i:i + batch_size]).pooler_output
            out.append(self.projection(pooled))
        return torch.cat(out, dim=0).cpu()
