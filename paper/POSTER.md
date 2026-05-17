# A0 Poster Content
Layout: portrait, 3 columns. Accent color: gold (#D4AF37) matching the paper/progress-report. Body sans-serif (Lato or Calibri), headers bold.

Two figures are reused from the paper:
- `paper/figures/ablation_bars.pdf` (results bar chart)
- TikZ architecture diagram from `paper/main.tex` (export from the compiled PDF as a cropped image, or recompile a standalone tikz file)

---

## Title bar (full-width header)

**Multimodal Fusion for Hate Video Classification on HateMM**
Youssef Dib · Kevin Abou Jaoude · Rami Aschkar
Group 7 · EE-559 Deep Learning · EPFL

---

## Column 1 — Motivation, Dataset, Method

### Motivation
Hate-speech moderation on user-generated video platforms is hard because hateful intent often emerges from the joint configuration of three modalities:
- **Visual cues** (what is shown)
- **Acoustic delivery** (how it is spoken)
- **Linguistic content** (what is said)

Unimodal systems risk false negatives on cross-modal hate and false positives on visually charged but linguistically innocuous content.

**Our question.** How much does multimodal fusion improve over the best unimodal classifier on HateMM, and is the gain consistent across target communities?

### Dataset: HateMM (Das et al., ICWSM 2023)
| | |
|---|---|
| Videos | 1,083 (BitChute) |
| Hate / non-hate | 431 / 652 |
| Total runtime | ~43.5 hours |
| Annotations | binary label + target community |
| License | CC BY 4.0 |

### Method
[INSERT: architecture diagram from paper/main.tex, page 1 of the compiled PDF]

Three-stage pipeline:
1. **Preprocess** — 32 frames @ 1 fps · 16 kHz mono audio · faster-whisper transcript
2. **Frozen encoders** — CLIP ViT-B/32 · wav2vec2-base · RoBERTa-base
3. **Trainable fusion** — 2-layer transformer (d=256, 4 heads), [CLS] readout, linear classifier

**Why frozen encoders?** Each unimodal baseline uses the *same* architecture restricted to one modality. The ablation isolates **modality information**, not architecture or capacity.

---

## Column 2 — Experiments & Results

### Setup
- Stratified split by label: 863 / 109 / 109 videos
- AdamW · lr 1e-4 · batch 16 · early stop on val macro-F1
- **3 seeds {42, 1337, 2025}**, data split fixed across seeds
- Single NVIDIA V100 GPU

### Modality Ablation (mean ± std over 3 seeds)
[INSERT: paper/figures/ablation_bars.pdf]

| Modality | Acc | F1 | AUROC |
|---|---|---|---|
| Video | .72±.03 | .71±.02 | .81±.01 |
| Audio | .67±.02 | .67±.03 | .77±.03 |
| Text | .79±.02 | .78±.02 | .83±.02 |
| **V+A+T** | **.79±.05** | **.78±.05** | **.88±.01** |

**Headline finding:**
- **AUROC gain is robust**: fusion +4.9 pp over text, gap ~3× combined std
- **Threshold metrics tie**: fusion and text both at 0.79 accuracy on average
- **Fusion variance is higher** (acc std 0.052 vs 0.021), consistent with higher-capacity models

A single-seed report would have **overstated** the accuracy gap. Multi-seed analysis surfaces this nuance.

### Per-Target-Community Audit (groups with n ≥ 15)
| Community | n | Text | V+A+T |
|---|---|---|---|
| Blacks | 46 | .73±.03 | **.75±.06** |
| Others | 39 | **.86±.05** | .84±.02 |
| Jews | 15 | **.76±.04** | .73±.13 |

All gaps between fusion and the best unimodal baseline are **within combined seed variance**. On Jews (smallest group), fusion's seed variance (0.13) **exceeds** its mean advantage by an order of magnitude.

---

## Column 3 — Discussion, Limitations, Conclusion

### What is fusion actually doing?
**A robust ranking gain. A noisy threshold gain.**

The fusion model learns better probability rankings (AUROC) than any unimodal model, but on threshold-dependent metrics it matches the strongest unimodal model with higher per-seed variance.

This pattern is consistent with a higher-capacity model that learns finer discrimination but is more sensitive to optimization randomness in the loss-to-prediction map.

### Per-community insight
At the subgroup level, fusion does **not** consistently outperform text. On small groups (Jews, n=15) fusion is unstable across seeds. **Group-level multimodal claims need either much larger per-group test support or methods explicitly designed for low-resource subgroups.**

### Limitations
- Small test set (n=109) inflates per-community variance
- Heuristic community-label normalization (multi-target labels canonicalized to sorted comma-joined form)
- Audio encoder pools to a fixed 64 tokens, over-averaging long videos

### Conclusion
Frozen-encoder multimodal fusion attains **AUROC 0.876 ± 0.012** on HateMM, a robust **+5 pp** gain over every unimodal baseline. On threshold-dependent metrics the gain washes out into seed variance — a finding single-seed reporting would have missed.

Pipeline is fully reproducible from the released dataset and our public code repository.

### References
[1] Das et al. (2023). *HateMM: A multi-modal dataset for hate video classification.* ICWSM.
[2] Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
[3] Baevski et al. (2020). *wav2vec 2.0.* NeurIPS.
[4] Liu et al. (2019). *RoBERTa.* arXiv:1907.11692.

### Code
github.com/YDIB11/Mini-Project

---

# Visual style guide

| Element | Spec |
|---|---|
| Body font | Lato / Calibri 24–32 pt |
| Section headings | Bold, 40–48 pt, gold (#D4AF37) |
| Title bar | 80–96 pt, dark grey/black on white |
| Author line | 30–36 pt grey |
| Tables | Booktabs-style (no vertical lines), gold header row, alt-row very light gold (#F8F2DC) |
| Figures | Already designed in the paper accent (gold + blue) |
| Background | White, optional light grey side gutters |
| Accent dividers | Thin (1–2 pt) gold horizontal rules between sections |

## Suggested column proportions (A0 portrait, 841×1189 mm)
- Title bar: full width, height ~140 mm
- 3 columns: each ~250 mm wide with ~30 mm gutters
- Vertical content area: ~1000 mm
