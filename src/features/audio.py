"""Frozen wav2vec2 encoder: mean-pools hidden states into a fixed-length sequence."""

from __future__ import annotations

from pathlib import Path

import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModel


class AudioEncoder:
    def __init__(
        self,
        name: str,
        target_tokens: int = 64,
        chunk_seconds: int = 30,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.target_tokens = target_tokens
        self.dim = self.model.config.hidden_size
        self.sample_rate = self.fe.sampling_rate
        self.chunk_samples = chunk_seconds * self.sample_rate

    @torch.inference_mode()
    def encode(self, wav_path: Path) -> torch.Tensor:
        """Returns (target_tokens, D) on CPU. Zero-padded if audio is too short."""
        wav, _ = librosa.load(str(wav_path), sr=self.sample_rate, mono=True)

        # Chunk to bound peak memory; concatenate hidden states across chunks.
        hidden_chunks: list[torch.Tensor] = []
        for i in range(0, max(len(wav), 1), self.chunk_samples):
            chunk = wav[i:i + self.chunk_samples]
            if chunk.size == 0:
                continue
            inputs = self.fe(chunk, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            hidden_chunks.append(out.last_hidden_state.squeeze(0).cpu())
        if not hidden_chunks:
            return torch.zeros((self.target_tokens, self.dim))
        hidden = torch.cat(hidden_chunks, dim=0)
        return _pool_to(hidden, self.target_tokens)


def _pool_to(seq: torch.Tensor, n: int) -> torch.Tensor:
    """Mean-pool (T, D) into n contiguous chunks -> (n, D). Pads with zeros if T < n."""
    T, D = seq.shape
    if T <= n:
        out = torch.zeros((n, D), dtype=seq.dtype)
        out[:T] = seq
        return out
    boundaries = torch.linspace(0, T, n + 1).round().long().tolist()
    return torch.stack([
        seq[boundaries[i]:max(boundaries[i + 1], boundaries[i] + 1)].mean(dim=0)
        for i in range(n)
    ])
