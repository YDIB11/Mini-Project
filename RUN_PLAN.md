# Run Plan

Last updated: 2026-05-16. Working from Noto with a Tesla V100.

## Step 1 — Preprocess (in progress)

Already running. Expected duration: ~6 hours with `whisper-small`. When it finishes you'll see:

```
Done. Manifest: cache/manifest.jsonl  failures: 0/1083
```

Verify counts when it's done:

```bash
ls cache/frames | wc -l
ls cache/audio | wc -l
ls cache/transcripts | wc -l
```

All three should be at or close to 1083 (some videos may fail; the manifest records why).

## Step 2 — Feature extraction (~15-30 min)

Downloads CLIP / wav2vec2 / RoBERTa weights on first run (~1.5 GB total, cached to `$HF_HOME`), then encodes all videos.

```bash
python -m src.features.extract --config configs/default.yaml
```

Verify outputs:

```bash
ls features/video | wc -l
ls features/audio | wc -l
ls features/text  | wc -l
wc -l features/index.jsonl
```

Each should be ~1083.

## Step 3 — Train four models (~10-15 min each on V100)

Same fusion transformer architecture for all four — the unimodal baselines are just the same model with a single-modality input, which makes the ablation a fair comparison.

```bash
python -m src.train --config configs/default.yaml                     # full fusion (video+audio+text)
python -m src.train --config configs/default.yaml --modalities video  # video-only baseline
python -m src.train --config configs/default.yaml --modalities audio  # audio-only baseline
python -m src.train --config configs/default.yaml --modalities text   # text-only baseline
```

Each run writes `checkpoints/<run_name>.pt` and `checkpoints/<run_name>.history.json`.

## Step 4 — Evaluate on test split

Each call prints overall metrics + per-target-community breakdown. Save them as JSON for the paper tables:

```bash
mkdir -p results
for m in video+audio+text video audio text; do
  python -m src.eval --config configs/default.yaml --checkpoint checkpoints/${m}.pt > results/${m}.json
  echo "--- ${m} ---"
  cat results/${m}.json
done
```

## Step 5 — Pull results back to local

Either via the JupyterLab file browser (right-click `results/` and `checkpoints/` → Download), or via git:

```bash
git add results/ checkpoints/*.history.json
git -c user.name="Youssef Dib" -c user.email="youssefjeandib@gmail.com" commit -m "add full-dataset training results and history"
git push
```

Don't commit `cache/`, `features/`, or the big `.pt` checkpoints — they're large and reproducible from the dataset + code. `.gitignore` already excludes them.

## Step 6 — Paper draft

EE-559 ICML-style template, 3 pages. Will need:

- Intro: problem + dataset + approach in one paragraph.
- Method: three frozen encoders, small transformer fusion, masking, splits.
- Results: one overall table, one modality-ablation table, one per-target-community table.
- Discussion: what each modality contributes, fairness observations, limitations.

Tell me when results are in and I'll draft it.
