# Mini-Project: Multimodal Hate Video Classification on HateMM

EE-559 Deep Learning, Group Mini-Project, **Group 7**, EPFL Spring 2026.

## Authors

- Youssef Dib
- Kevin Abou Jaoude
- Rami Aschkar

## What this submission contains

| File / folder | Contents |
|---|---|
| `paper/main.pdf` | 3-page IEEE-style write-up (ICML 2021 template). Source in `paper/main.tex` and `paper/main.bib`. |
| `poster/poster.pdf` | A0 poster presented on 27 May. Source in `poster/poster.html` and `poster/poster.tex`. |
| `src/` | All Python source for preprocessing, feature extraction, training, evaluation. |
| `scripts/` | Multi-seed launcher, results aggregator, bar-chart plotter, optional Runai cluster launcher. |
| `configs/default.yaml` | All hyperparameters (encoder choice, sequence lengths, fusion size, training schedule, splits). |
| `results/` | Per-seed evaluation JSONs (21 runs) plus the aggregated `_multiseed_summary.json` and `_multiseed_per_community.json` used to populate Tables 1 and 2 of the paper. |
| `requirements.txt` | Python dependencies. |

Large or external artefacts that we do **not** redistribute:

- The HateMM `.mp4` files (download from Zenodo, link below).
- Pretrained encoder weights (auto-downloaded from Hugging Face on first run, link below).
- Cached preprocessing outputs (`cache/`), cached features (`features/`), and model checkpoints (`checkpoints/`). These are gitignored and recomputable from the released code plus the dataset.

## Pipeline (overview)

1. **Preprocess.** For each video: sample 32 RGB frames at 1 fps (`src/preprocess/frames.py`), extract 16 kHz mono audio via `ffmpeg` (`src/preprocess/audio.py`), transcribe the audio with faster-whisper (`src/preprocess/asr.py`). Outputs are cached under `cache/`.
2. **Frozen feature extraction.** Three pretrained encoders, all kept frozen, produce per-modality embeddings: CLIP ViT-B/32 for frames, wav2vec2-base for audio, RoBERTa-base for the transcript. Features are cached under `features/`.
3. **Fusion training.** A 2-layer transformer fusion head (`src/models/fusion.py`) is trained on top of the cached features for any subset of the three modalities. The same head supports the unimodal and pairwise baselines.
4. **Evaluation.** Per-seed metrics and per-target-community breakdown saved as JSON in `results/`. Aggregated by `scripts/aggregate_seeds.py`.

## Dataset (not redistributed)

**HateMM** (Das et al., ICWSM 2023), 1,083 BitChute videos with binary hate labels and target-community annotations, released under CC BY 4.0.

- Dataset: https://zenodo.org/records/7799469
- Paper: https://arxiv.org/abs/2305.03915
- Reference code: https://github.com/hate-alert/HateMM

Download once from the Zenodo record and place the contents under `data/HateMM/` so that `data/HateMM/hate_videos/`, `data/HateMM/non_hate_videos/`, and `data/HateMM/HateMM_annotation.csv` all exist.

## Pretrained models (not redistributed)

All three encoders are loaded from Hugging Face on first run and cached locally under `$HF_HOME`. We use them frozen; no model weights are committed to this repo.

| Modality | Hugging Face model card | License | Used by |
|---|---|---|---|
| Video frames | https://huggingface.co/openai/clip-vit-base-patch32 | MIT | `src/features/video.py` |
| Audio | https://huggingface.co/facebook/wav2vec2-base-960h | Apache-2.0 | `src/features/audio.py` |
| Text (transcript) | https://huggingface.co/roberta-base | MIT | `src/features/text.py` |
| Transcription (ASR) | https://huggingface.co/openai/whisper-small (via faster-whisper) | MIT | `src/preprocess/asr.py` |

## Setup

### Option A: local virtual environment (recommended off-cluster)

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: EPFL Noto with the course kernel

The `EE559_Kernel` venv at `/home/gnoto_venvs/ee559_venv/` ships everything we use except `opencv-python-headless` (server-friendly OpenCV) and `faster-whisper` (fast ASR). Install both into your user packages without modifying the shared environment:

```bash
EE559_PY=/home/gnoto_venvs/ee559_venv/bin/python3
$EE559_PY -m pip install --user opencv-python-headless faster-whisper
```

Run every command below as `$EE559_PY -m src.<module>` instead of `python -m src.<module>`.

### System dependency (both options)

`ffmpeg` and `ffprobe` must be on `PATH`. On Noto, the static build can be installed into the home directory:

```bash
mkdir -p ~/.local/bin && cd /tmp \
  && wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
  && tar xf ffmpeg-release-amd64-static.tar.xz \
  && cp ffmpeg-*-amd64-static/{ffmpeg,ffprobe} ~/.local/bin/ \
  && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
  && export PATH="$HOME/.local/bin:$PATH"
```

## Reproduction

All commands below assume the project root as the working directory and an activated environment.

### 1. Preprocess all 1,083 videos (~6 hours on a V100)

```bash
python -m src.preprocess.run --config configs/default.yaml
```

Writes `cache/frames/`, `cache/audio/`, `cache/transcripts/`, and `cache/manifest.jsonl`. Use `--limit 5` for a quick smoke test.

### 2. Extract frozen features (~15-30 min)

```bash
python -m src.features.extract --config configs/default.yaml
```

Downloads the CLIP, wav2vec2, and RoBERTa weights from Hugging Face on first call (about 1.5 GB total, cached under `$HF_HOME`). Writes `features/video/`, `features/audio/`, `features/text/`, and `features/index.jsonl`.

### 3a. Train and evaluate a single configuration (~15 min on a V100)

```bash
python -m src.train --config configs/default.yaml --modalities video audio text
python -m src.eval  --config configs/default.yaml --checkpoint checkpoints/video+audio+text.pt > results/video+audio+text.json
```

Replace `--modalities video audio text` with any non-empty subset of `{video, audio, text}` to train one of the seven ablation rows.

### 3b. Reproduce the full multi-seed ablation (~4 hours on a V100)

```bash
bash scripts/multiseed.sh
python scripts/aggregate_seeds.py
```

The launcher iterates over the seven modality subsets and three seeds `{42, 1337, 2025}`, writing per-run results to `results/<modalities>_seed<N>.json`. The aggregator emits the two summary JSONs already committed under `results/`.

### 4. Regenerate the paper figure

```bash
python scripts/plot_ablation.py
```

Reads from `results/_multiseed_summary.json` and writes `paper/figures/ablation_bars.{pdf,png}`.

### 5. Rebuild the paper PDF

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Optional: run on EPFL RCP / Runai

`scripts/runai_launch.sh` is a template for submitting training jobs to the EPFL Runai scheduler. Edit the `IMAGE`, `PVC`, and `PROJECT_DIR` placeholders to match your group's allocation before first use. This is not required to reproduce the results; the pipeline runs on any single GPU.

## Layout

```
configs/         experiment configs (YAML)
data/            HateMM lives here on disk; gitignored (only .gitkeep tracked)
paper/           3-page IEEE-style write-up (main.tex, main.bib, figures/, main.pdf)
poster/          A0 poster sources and rendered PDF
results/         per-seed and aggregated test-set metrics (committed for reproducibility)
scripts/         launchers and post-processing utilities
src/data.py            train/val/test split, collate, DataLoader factory
src/eval.py            evaluate a checkpoint, emit JSON with overall + per-community metrics
src/metrics.py         accuracy / F1 / AUROC / bootstrap CIs / per-community breakdown
src/train.py           training loop with early-stop on validation macro-F1
src/preprocess/        frame, audio, ASR extraction; HateMM annotation loader
src/features/          frozen CLIP / wav2vec2 / RoBERTa encoders and cached-feature Dataset
src/models/fusion.py   2-layer transformer fusion head (the only trainable component)
```

## Compute

All preprocessing, training, and evaluation runs on a single NVIDIA V100 GPU. No external cloud or commercial model APIs are used.

## Citation policy

This repository points to the HateMM dataset and to the four pretrained encoders we use; neither the dataset nor the encoder weights are redistributed here. Full IEEE-style references for all sources are in `paper/main.bib` and rendered on page 3 of `paper/main.pdf`.
