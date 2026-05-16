"""Multimodal fusion transformer.

Works with any non-empty subset of modalities, so the same module covers
the full fusion model and the three unimodal baselines (just pass a
feature_dims dict with one entry).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int],
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.modalities = tuple(feature_dims.keys())
        self.projections = nn.ModuleDict({
            m: nn.Linear(d_in, d_model) for m, d_in in feature_dims.items()
        })
        self.modality_embeds = nn.ParameterDict({
            m: nn.Parameter(torch.randn(d_model) * 0.02) for m in self.modalities
        })
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        first = next(v for v in batch.values() if torch.is_tensor(v) and v.dim() >= 2)
        B = first.shape[0]

        tokens = [self.cls.expand(B, -1, -1).to(device)]
        masks = [torch.ones(B, 1, dtype=torch.bool, device=device)]
        for m in self.modalities:
            if m not in batch:
                continue
            x = batch[m].to(device)                              # (B, T, D_in)
            x = self.projections[m](x) + self.modality_embeds[m]  # (B, T, d_model)
            tokens.append(x)
            mk = batch.get(f"{m}_mask")
            if mk is None:
                mk = torch.ones(B, x.shape[1], dtype=torch.bool, device=device)
            masks.append(mk.to(device).bool())

        seq = torch.cat(tokens, dim=1)
        key_padding_mask = ~torch.cat(masks, dim=1)  # True = ignore
        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        return self.head(self.norm(out[:, 0]))
