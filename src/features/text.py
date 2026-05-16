"""Frozen text encoder for the ASR transcript (RoBERTa or similar)."""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer


class TextEncoder:
    def __init__(self, name: str, max_length: int = 128, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_length = max_length
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (seq, mask): (max_length, D) float, (max_length,) bool — both on CPU."""
        enc = self.tok(
            text or "",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_dev = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc_dev).last_hidden_state.squeeze(0).cpu()
        mask = enc["attention_mask"].squeeze(0).bool()
        return hidden, mask
