# Mini-Project: Multimodal Hate Video Classification on HateMM

EE-559 (Deep Learning) group mini-project, EPFL.

## Overview

We train a small transformer that fuses frozen video, audio, and text features for binary hate / non-hate classification on the HateMM dataset (Das et al., ICWSM 2023). Unimodal baselines on the same cached features serve as the main comparison. We report standard classification metrics, a modality ablation, and a per-target-community error breakdown.

## Authors

- Youssef Dib
- Kevin Abou Jaoude
- Rami Aschkar

## Pipeline

1. Preprocess: sample frames, extract 16 kHz mono audio, transcribe with faster-whisper.
2. Feature extraction: frozen CLIP (image), wav2vec2 (audio), RoBERTa (text). Features are cached to disk.
3. Train: small transformer fusion module over cached features, plus unimodal baselines (TBD).

## Layout

```
configs/         experiment configs (YAML)
data/            HateMM lives here on the cluster (gitignored contents)
scripts/         RCP / Runai launch scripts
src/preprocess/  frame, audio, ASR extraction
src/features/    frozen encoders, caching, Dataset loader
src/models/      fusion module and baselines (TBD)
src/analysis/    ablation and per-community breakdown (TBD)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires `ffmpeg` on PATH for audio extraction.

## Run

```bash
python -m src.preprocess.run   --config configs/default.yaml
python -m src.features.extract --config configs/default.yaml
```

All knobs (encoder choices, sequence lengths, train/val/test split, fusion size, optimizer) live in `configs/default.yaml`.

## Data

HateMM is released by Das et al. under CC BY 4.0 on Zenodo. We do not redistribute it. Download once from the official record and place the contents under `data/HateMM/`.

- Dataset: https://zenodo.org/records/7799469
- Paper: https://arxiv.org/abs/2305.03915
- Reference code: https://github.com/hate-alert/HateMM

## Compute

All training and preprocessing runs on EPFL's RCP / Runai cluster under the EE-559 allocation. No external cloud or commercial model APIs are used.
