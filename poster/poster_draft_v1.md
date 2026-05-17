# Poster draft v1 — Multimodal Hate Video Classification on HateMM

EE-559 Deep Learning, EPFL, Spring 2026.
Authors: Youssef Dib, Kevin Abou Jaoude, Rami Aschkar.

This document holds the text content for every box in the official A0 PowerPoint template (`EE559_A0_poster_template.pptx`, A0 portrait, 84.1 × 118.9 cm). Paste each section into the matching template block. Numbers in the two tables are placeholders for the current single-seed results; replace once multi-seed averages are in.

---

## Header

- Project title: **Multimodal Hate Video Classification on HateMM**
- Course tagline (already on template): EE-559: Deep Learning, 2026
- Authors: **Youssef Dib, Kevin Abou Jaoude, Rami Aschkar**
- Group: **Group [TODO: group number]**

---

## Problem definition

Hateful content online increasingly travels as video, where meaning is carried jointly by visual context, audio delivery, and the spoken message. Single-modality classifiers struggle on the cases where these three signals must be combined to distinguish hate from satire or critical discourse. We ask whether a small transformer trained on top of frozen pretrained per-modality encoders can outperform every unimodal baseline on a real corpus of hateful and non-hateful videos, and we audit the result across target communities.

---

## Dataset(s)

We use the **HateMM** dataset (Das et al., ICWSM 2023), released by its authors under CC BY 4.0 on Zenodo. Per-video labels include the binary hate flag, the target community, and timestamped hateful segments. We do not modify or redistribute the dataset.

- 1083 BitChute videos (431 hate, 652 non-hate)
- Approximately 43.5 hours of content, 6.3 GB on disk
- Stratified split by binary label: 80 % train (863), 10 % val (109), 10 % test (109), fixed random seed
- Preprocessing succeeded on 1081 / 1083 videos; 2 source files were unreadable

---

## Method

**Pipeline.** (1) Preprocess: sample frames at 1 fps capped at 32 per video and center-cropped to 224 × 224; extract 16 kHz mono audio via ffmpeg; transcribe speech with faster-whisper (small). (2) Frozen feature extraction: CLIP ViT-B/32 image embeddings per frame, wav2vec2-base hidden states mean-pooled into 64 fixed-length audio tokens, RoBERTa-base token embeddings up to length 128 on the transcript. (3) Fusion: a 2-layer transformer encoder (d = 256, 4 heads, GELU, pre-norm) ingests the concatenated per-modality token sequences plus learnable modality-type embeddings and a [CLS] token; the [CLS] output goes through a linear head for binary classification. Optimizer AdamW, lr 1e-4, batch size 16, up to 20 epochs, early stop on validation macro-F1 with patience 5.

**Why frozen encoders.** All three encoders stay frozen so that the modality ablation is a fair comparison: only the small fusion head varies between configurations. Restricting the same architecture to a single modality at the input gives us our unimodal baselines.

**Architecture diagram for the box** (rebuild in PowerPoint using native shapes; ASCII reference in `architecture_diagram.txt`).

---

## Related works

Three key references, each with a one-line role in our work.

- **HateMM (Das et al., 2023)** [1]: defines the task, releases the dataset and the community labels we audit against; we follow its preprocessing intent but rewrite the pipeline.
- **CLIP (Radford et al., 2021)** [2]: provides the frozen vision encoder; we use its image branch only, kept frozen.
- **Attention Bottlenecks for Multimodal Fusion (Nagrani et al., 2021)** [3]: motivates the cross-modal-attention fusion design; we adopt a lighter [CLS]-readout variant suited to the dataset scale.

---

## Validation

**Table 1 — Modality ablation on the 109-video test set.** Headline numbers are single-seed; we will replace with multi-seed `mean ± std` before final.

| Modality      | Accuracy | Macro F1 | Precision | Recall | AUROC |
|---------------|---------:|---------:|----------:|-------:|------:|
| Video + Audio + Text (fusion) | **0.835** | **0.824** | **0.821** | 0.744 | **0.873** |
| Text only     | 0.780    | 0.763    | 0.757     | 0.651  | 0.842 |
| Video only    | 0.743    | 0.726    | 0.692     | 0.628  | 0.815 |
| Audio only    | 0.697    | 0.695    | 0.589     | **0.767** | 0.775 |

**Table 2 — Per-target-community test accuracy** (only communities with n ≥ 10 reported).

| Community | n  | Fusion | Video | Audio | Text  |
|-----------|---:|-------:|------:|------:|------:|
| Blacks    | 46 | 0.804  | 0.717 | 0.630 | 0.717 |
| Others    | 39 | 0.846  | 0.795 | 0.718 | **0.872** |
| Jews      | 15 | **0.867** | 0.600 | 0.733 | 0.733 |

**Figure — bar chart of macro-F1 across the four models** (source in `bar_chart.py`, output `bar_chart.png`).

**Key observations.** Fusion wins on every aggregate metric except recall, where the audio-only baseline scores higher (audio behaves as a high-recall, low-precision signal). Text is the strongest individual modality, consistent with the verbal nature of much hate speech. On the *Others* community, text alone slightly beats fusion, suggesting that for that group the discriminative signal is largely already in the transcript. On *Jews* (the smallest reported group), fusion shows the largest gain over any single modality.

---

## Limitations

- **Single seed.** Headline numbers come from one run; we add multi-seed averages to quantify uncertainty.
- **Small test set.** 109 videos; per-community statistics for groups with n < 10 are excluded as unreliable.
- **ASR quality varies.** Whisper-small degrades on shouting, music, accented speech, weakening the text modality on a subset of inputs.
- **Frozen backbones cap absolute performance.** Intentional methodological choice: it keeps modality contributions interpretable, but unfrozen finetuning would likely lift every row of Table 1.
- **Class imbalance not loss-weighted.** Recall on the hate class is 0.74 at full fusion, suggesting the model is conservative on positives.

---

## Conclusion

Fusing three frozen pretrained encoders through a small transformer head outperforms every unimodal baseline on HateMM hate video classification by a clean margin in both macro-F1 and AUROC. The per-community audit reveals that the gain is not uniform: text alone suffices for some target groups while fusion is decisive for others, with direct implications for moderation pipelines that today combine modality-specific classifiers via simple voting. Next steps: multi-seed averaging of the headline table, a counterfactual modality-masking probe for per-example modality attribution, and class-rebalanced training to lift hate recall.

---

## References (IEEE)

[1] M. Das, R. Raj, P. Saha, B. Mathew, M. Gupta, and A. Mukherjee, "HateMM: A Multi-Modal Dataset for Hate Video Classification," in *Proc. AAAI Int. Conf. Web and Social Media (ICWSM)*, vol. 17, 2023, pp. 1014-1023.

[2] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. ICML*, vol. 139, 2021, pp. 8748-8763.

[3] A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, and C. Sun, "Attention Bottlenecks for Multimodal Fusion," in *Proc. NeurIPS*, vol. 34, 2021, pp. 14200-14213.

---

## Layout notes for the PowerPoint pass

The template already structures the A0 portrait into the seven content boxes above. Suggested visual hierarchy when filling them in:

- Header strip with title and authors should occupy the top ~10 % of the canvas; keep group number large on the right.
- *Problem definition* and *Dataset(s)* read left to right on the upper third; both are short and prose-only.
- *Method* is the visual centerpiece: the architecture diagram occupies most of the box; one line of caption per stage.
- *Related works* stays compact: bullet list, three items, half a sentence each.
- *Validation* is the second visual centerpiece: Table 1 left, Table 2 right, bar chart below if space allows. This is where the eye should land first; consider a colored background tint on the fusion row of Table 1.
- *Limitations* and *Conclusion* sit on the lower third side by side; *References* hugs the bottom strip.
- Use a single accent color for the fusion-related cells (the headline row of Table 1, the "Jews" row of Table 2, the fusion bar in the chart) to anchor the eye.

When multi-seed numbers land, only eight cells in Table 1 change and three cells in Table 2 change. Plan around five minutes of edits at the end.
