# Poster presentation notes: Multimodal Hate Video Classification on HateMM

Full project walkthrough for the EE-559 poster session. Read it once, then close it and try to retell it in your own words. Where you stumble is the part to study.

---

## Step 1: The question

Hateful content on video platforms is hard to catch because the hate can hide in any one of three **modalities** at once: in what you **see** (the visual frames), in what you **hear** (tone, music, shouting in the audio track), or in what is **said** (the literal words, available only via a transcript). A classifier that only reads tweets cannot catch a slur uttered on the audio; one that only watches frames misses harmful framing in the spoken commentary. So we built a system that combines all three modalities and asked one focused question: does combining the three beat each one alone, and does it do so equally across target communities?

## Step 2: The dataset

We use **HateMM**, released by Das et al. at ICWSM 2023 under CC BY 4.0. It contains 1,083 BitChute videos, 431 labeled hate and 652 labeled non-hate. Each hate video also carries a **target-community** annotation (Blacks, Jews, Others, etc.), which is what makes the fairness audit later possible. We downloaded it once from Zenodo; we did not collect, scrape, or relabel anything.

## Step 3: Preprocessing each video into three streams

A raw `.mp4` file is too tangled to feed into a model directly. So for each of the 1,083 videos we run a one-time chopping step that splits the file into three clean streams, one per modality:

- **Frames**: we sample one still image per second (this rate is called **1 fps**, frames per second), capping at 32 frames per video, and resize each to 224 by 224 pixels.
- **Audio**: we strip out the soundtrack with the `ffmpeg` tool, producing 16 kHz mono PCM ("16 kHz" means 16,000 amplitude samples per second; "mono" means one channel).
- **Transcript**: we run the audio through **faster-whisper** (small), an **ASR** model (Automatic Speech Recognition, software that listens and writes down the words), to get an English transcript.

All three streams get cached to disk so the raw video is touched exactly once. The full preprocessing pass takes about 6 hours on a single V100 GPU.

## Step 4: Turning each stream into embeddings using frozen encoders

We then pass each stream through a pretrained **encoder**. An encoder is a neural network that converts raw input (an image, an audio segment, a sentence) into a list of numbers, called an **embedding**, that captures the input's meaning. Each output is a sequence of **tokens**, where one token corresponds to one frame, one audio slice, or one word, and each token is described by its embedding.

We use three encoders, all kept **frozen**, meaning their internal weights are locked and never updated during training:

- **Visual frames** go through **CLIP ViT-B/32**, the vision tower of OpenAI's CLIP, trained on 400 million image-text pairs. Each video gives us 32 tokens, each of 512 numbers.
- **Audio** goes through **wav2vec2-base**, Meta's speech model trained on 60,000 hours of unlabeled audio. The raw output has variable length, so we **mean-pool** it ("chunk into 64 equal pieces and average each piece") down to a fixed 64 tokens of 768 numbers each. This means a 30-second video and a 5-minute video both end up with the same audio sequence length.
- **Transcripts** go through **RoBERTa-base**, Meta's text model trained on 160 GB of text. We get up to 128 tokens of 768 numbers each.

Why frozen? Two reasons. First, it is cheap: no gradients flow back through these massive models, so a single V100 is enough hardware. Second, and more importantly, it makes the ablation fair: when we later compare V+A+T against text-only, the difference is purely about which modalities are present, not about which models were trained longer or harder. All embeddings are cached to disk, so every later experiment is just a cheap tensor read.

## Step 5: The fusion transformer (the only trainable component)

Now the small part we actually built and trained. **Fusion** means combining information from multiple modalities into a single decision. Our fusion model is a **transformer**: a neural network (the same family as ChatGPT) that processes a sequence of tokens using **attention**, a mechanism that lets each token look at every other token and decide which ones matter for the task.

The model works in five steps:

1. **Projection**. The three modalities live in different sizes (512, 768, 768). We pass each token through a small linear layer that maps it down to a common size of 256 numbers, so everything lives in the same space.
2. **Modality tagging**. Each token also gets a small learnable "name tag" vector added to it, so the transformer knows whether a given token came from video, audio, or text.
3. **The [CLS] token**. We prepend a special learnable 256-number vector at the front of the sequence. CLS stands for classification. It starts out random and will end up holding a summary of the whole video.
4. **Two transformer layers**. The full sequence (1 [CLS] + 32 video + 64 audio + 128 text = up to 225 tokens) passes through two transformer encoder layers, each with 4 attention heads (four parallel ways of comparing tokens), a small feed-forward block with GELU activation, 10% dropout for regularisation, and the pre-norm variant (a stability trick where we normalise before each sub-block rather than after). After two layers, the [CLS] token has soaked up information from every other token.
5. **Classifier**. We pull just the [CLS] output (256 numbers) and pass it through one final linear layer that produces two scores, one for "hate" and one for "non-hate". Whichever is higher is the prediction.

Total trainable parameters: about 1.5 million. Deliberately small, because with only ~865 training videos, a larger model would memorise them. The same architecture is used unchanged for the unimodal baselines: we just feed in one modality at a time and the rest of the code is identical. That uniformity is what makes the ablation a clean test of modality information rather than model capacity.

## Step 6: Training with multi-seed

We use the textbook recipe:

- **Loss**: cross-entropy, the standard measure of how wrong a probabilistic prediction is.
- **Optimizer**: AdamW, the standard adaptive gradient method.
- Learning rate 1e-4, weight decay 1e-2, batch size 16, up to 20 epochs (one epoch = one full pass over the training set), with early stopping on validation macro-F1 if performance hasn't improved for 5 epochs.
- **Splits**: 80/10/10 stratified by binary label. **Stratified** means we keep the same hate-to-non-hate ratio in each split, so no split is accidentally easier than another. That gives us 863 training videos, 109 validation, 109 test.

For each modality combination we then train **three times with different random seeds** {42, 1337, 2025}. Same data, same architecture, only the random number generator changes. This is the **multi-seed** step. It tells us whether a result is real or a lucky fluke. It turned out to be the single most important methodological choice we made, and it shaped our headline finding.

Each training run takes about 10 to 15 minutes on a V100.

## Step 7: Evaluation with the right metrics

We compute five metrics on the test set, but you need to internalise three of them cold:

- **Accuracy**: out of 109 test videos, what fraction did the model classify correctly. Easy to grasp but misleading when classes are imbalanced.
- **Macro-F1**: balances precision (of the hate predictions, how many were correct) and recall (of the actual hate videos, how many we caught), and weights hate and non-hate equally. Robust to imbalance.
- **AUROC** (Area Under the Receiver Operating Characteristic curve): this is the metric that carries our headline. It measures how well the model **ranks** examples. AUROC of 1.0 means every hate video gets a higher hate-score than every non-hate video. AUROC of 0.5 is random. Crucially, AUROC is **threshold-free**: it does not care where you draw the line between "called hate" and "called non-hate"; it only cares whether the ordering is correct.

The distinction between AUROC and accuracy is the conceptual hinge of the whole paper. AUROC = ranking quality. Accuracy = how often you get the binary call right at a fixed threshold. They can disagree.

We also run a **per-community audit**: we split the test set by which group is targeted (Blacks, Jews, Others) and compute accuracy per group, to check the model is not biased.

Finally, we run all seven possible modality subsets through this same pipeline (V alone, A alone, T alone, V+A, V+T, A+T, V+A+T). That is the **ablation**: same architecture, same training recipe, only the inputs differ, so the comparison isolates exactly what each modality contributes.

## Step 8: What we found

Three observations, in decreasing order of how loudly you should say them:

**1. AUROC improves robustly with fusion.** Full fusion (V+A+T) reaches 0.88 ± 0.01; text alone reaches 0.83 ± 0.02. The gap is well outside seed variance, so the win is real and reproducible. Each added modality lifts AUROC monotonically: 0.77 (audio alone) to 0.81 (video) to 0.83 (text) to 0.84/0.85/0.86 (the three pairs) to 0.88 (all three). Each step buys ranking signal.

**2. Accuracy and macro-F1 do not improve.** Once averaged over three seeds, full fusion ties text-only at 0.79 accuracy. So the model isn't getting more videos right at a fixed threshold; it's only improving the ordering. This matters because it changes the use case: fusion is useful for triage (sorting a moderator's queue by risk), not for autonomous yes/no moderation.

**3. The Seed Trap.** Our first training run used seed 42 only and showed fusion winning by **+5.5 accuracy points**, a great-looking headline result. When we ran two more seeds and averaged, that gap collapsed to zero. The accuracy win was a fluke. Only AUROC survived multi-seed scrutiny. A single-seed report would have published a false claim. This is the most original-sounding methodological observation in the paper, so lean into it at the poster.

## Step 9: Fairness audit

Three target communities have enough test examples to measure (n ≥ 15):

- **Blacks (n = 46)**: full fusion wins at 0.75 accuracy, narrowly beating text alone at 0.73.
- **Others (n = 39)**: text alone wins at 0.85. But the "Others" subset is mostly non-hate in our test split, so high accuracy partly reflects the model defaulting to "non-hate" rather than understanding.
- **Jews (n = 15)**: text alone wins at 0.76, but with only 15 examples the result is shaky and the per-seed standard deviation for full fusion is ±0.13.

Conclusion: no single model is best across all communities. That motivates the three deployment options on the poster: V+A+T when ranking quality matters most, V+T as a cost-aware sweet spot (skipping the audio pipeline entirely), text-only as the cheapest option that still ties V+A+T on accuracy.

## Step 10: Limitations you must acknowledge

- **Small test set** (109 videos). Per-community statistics wobble.
- **BitChute is a low-moderation platform**, so results characterise hate detection on that source distribution; transferring to YouTube or TikTok would need recalibration.
- **Whisper degrades** on shouting, music, and accented speech, so the text modality is noisier than the numbers suggest.
- **Audio mean-pools to 64 tokens** regardless of video length, so a 10-minute video gets the same audio resolution as a 30-second one.
- **Frozen encoders cap absolute performance.** Unfreezing them would lift the numbers but break the controlled ablation.

---

## One-line summary to remember on your feet

**"AUROC up, accuracy flat, Seed Trap."** If you remember those four words, you can field 80% of questions.
