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

Two equivalent setup paths. Use whichever fits your environment.

### Option A — EPFL Noto with the EE-559 course kernel (recommended)

The course kernel (`EE559_Kernel`, venv at `/home/gnoto_venvs/ee559_venv/`) already provides everything we use except two packages we added for our pipeline:

| Package | Why we added it | Used by |
|---|---|---|
| `opencv-python-headless` | Decode video files and sample frames; headless variant avoids the libGL system dependency that the standard `opencv-python` requires on servers. | `src/preprocess/frames.py` |
| `faster-whisper` | CT2-based ASR for transcribing the audio track; substantially faster than `openai-whisper` and runs on the V100. | `src/preprocess/asr.py` |

Install both into your user packages (does not modify the shared course env):

```bash
EE559_PY=/home/gnoto_venvs/ee559_venv/bin/python3
$EE559_PY -m pip install --user opencv-python-headless faster-whisper
```

Then run every command with `$EE559_PY -m src.<module>` instead of `python -m src.<module>`.

### Option B — Fresh venv (any environment outside Noto)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Re-activate the venv at the start of each new shell.

### System dependency (both options)

`ffmpeg` (and `ffprobe`) must be on `PATH` for audio extraction. On Noto, install the static build into your home:

```bash
mkdir -p ~/.local/bin && cd /tmp \
  && wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
  && tar xf ffmpeg-release-amd64-static.tar.xz \
  && cp ffmpeg-*-amd64-static/{ffmpeg,ffprobe} ~/.local/bin/ \
  && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
  && export PATH="$HOME/.local/bin:$PATH"
```

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
