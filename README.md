# Hindi Line-Level OCR: Architectural Comparison & Controlled Ablation Study

A comparative and empirical study of sequence-to-sequence architectures for **Devanagari (Hindi) line-level Optical Character Recognition (OCR)** trained on the 80,000-sample synthetic Hindi line-image dataset.

This repository investigates two distinct modeling paradigms:
1. **Custom CRNN + Multiplicative Attention (`ResNet50` + 2-layer `BiLSTM` + Luong Attention)** paired with a domain-specific **116-token Devanagari character tokenizer** and aspect-ratio-native `128 × 512` inputs—achieving **0.56% CER**, **2.16% WER**, and **92.00% exact line accuracy** with beam search.
2. **Vision-Encoder-Decoder Transformer (`TrOCR_HI`: `DeiT-base` + `mBERT`)** evaluated under a strict **one-factor-at-a-time controlled ablation protocol** across 12 training configurations (augmentation, preprocessing, weight initialization, data scaling, learning rate) and 5 inference decoding strategies.

---

## Key Benchmark Results

### 1. Primary Model Performance (`ResNet50 + BiLSTM + Luong Attention`)
Evaluated on the held-out test split (`90% train / 10% test`, 20 epochs with scheduled sampling and differential learning rates):

| Decoding Strategy | Beam Width | CER (%) $\downarrow$ | WER (%) $\downarrow$ | Char F1 $\uparrow$ | Exact Line Acc (%) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Greedy Decoding** | 1 | 0.81 | 2.95 | 0.9966 | 90.50 |
| **Beam Search** | 5 | **0.56** | **2.16** | **0.9976** | **92.00** |

**Training Progression (Epochs 12 $\rightarrow$ 20):**
* **Epoch 12 (Checkpoint Gate):** `CER = 1.45%` | `WER = 4.52%` | `Char F1 = 0.9947` | `Acc = 82.50%`
* **Epoch 16:** `CER = 0.93%` | `WER = 2.95%` | `Char F1 = 0.9962` | `Acc = 88.50%`
* **Epoch 20 (Best):** `CER = 0.81%` | `WER = 2.95%` | `Char F1 = 0.9966` | `Acc = 90.50%`

---

### 2. Controlled Ablation Study (`TrOCR_HI`: DeiT-base + mBERT)
To isolate individual design choices under a fixed compute budget (`N_train = 12,000`, `N_val = 1,000`, `N_test = 2,000`, 2 epochs, effective batch size = 8, `seed = 42`), exactly one factor was varied at a time relative to the baseline (`DeiT-base` + `mBERT`, `384 × 384` squash resize, `lr = 4e-5`, `ElasticTransform` + `GaussianBlur`, `num_beams = 4`, `length_penalty = 2.0`, `no_repeat_ngram_size = 3`).

| Group | Variant | Change vs. Baseline | Trainable Params | Val CER (%) | Test CER (%) | Test WER (%) | $\Delta$ Test CER (pp) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Reference** | `baseline` | Default baseline configuration | 293.02M | 77.91 | 78.37 | 92.04 | `0.00` |
| **Augmentation** | `elastic_only` | Remove `GaussianBlur` (keep `ElasticTransform`) | 293.02M | **75.10** | **75.22** | **88.65** | **-3.15** |
| | `no_aug` | Remove both `ElasticTransform` & `GaussianBlur` | 293.02M | 76.15 | 76.10 | 90.06 | -2.27 |
| | `blur_only` | Remove `ElasticTransform` (keep `GaussianBlur`) | 293.02M | 79.86 | 79.21 | 93.60 | +0.84 |
| **Preprocessing** | `resize_pad` | Aspect-ratio preserving resize + background padding | 293.02M | 80.49 | 80.95 | 92.79 | +2.58 |
| **Initialization** | `random_encoder` | Re-initialize `DeiT` encoder weights randomly | 293.02M | 78.45 | 77.95 | 93.55 | -0.42 |
| | `freeze_encoder` | Freeze `DeiT` encoder (train decoder + cross-attn) | 206.34M | 79.83 | 79.74 | 95.81 | +1.37 |
| | `random_decoder` | Re-initialize `mBERT` decoder weights randomly | 293.02M | 81.05 | 81.28 | 95.96 | +2.91 |
| **Data Scale** | `data_25pct` | 25% training subset (`3,000` samples) | 293.02M | 80.26 | 80.40 | 96.94 | +2.03 |
| | `data_50pct` | 50% training subset (`6,000` samples) | 293.02M | 83.21 | 83.04 | 97.78 | +4.67 |
| **Optimization** | `lr_1e-5` | Lower peak learning rate (`4e-5` $\rightarrow$ `1e-5`) | 293.02M | 79.34 | 79.47 | 95.06 | +1.10 |
| | `lr_1e-4` | Higher peak learning rate (`4e-5` $\rightarrow$ `1e-4`) | 293.02M | 123.61 | 121.22 | 250.57 | +42.85 |

#### Inference-Time Decoding Ablation (Trained Baseline Checkpoint, No Retraining)
| Decoding Variant | Generation Override | Val CER (%) | Test CER (%) | Test WER (%) | $\Delta$ Test CER (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `baseline_decode` | `num_beams=4, no_repeat_ngram_size=3, lp=2.0` | 77.91 | 78.37 | 92.04 | `0.00` |
| `greedy` | `num_beams=1, lp=1.0, early_stopping=False` | **76.02** | **76.09** | 92.48 | **-2.28** |
| `beam4_lp1.0` | `length_penalty=1.0` | 77.36 | 77.92 | **91.72** | -0.45 |
| `beam4_no_ngram_block` | `no_repeat_ngram_size=0` | 78.30 | 78.72 | 91.90 | +0.35 |
| `beam8` | `num_beams=8` | 78.94 | 78.98 | 91.82 | +0.61 |

---

## Core Scientific Insights

1. **Tokenization Bottleneck in Multilingual Subword Models:**
   A round-trip sanity check on 2,000 Devanagari lines reveals that `bert-base-multilingual-cased` imposes an **irreducible 1.62% CER floor** even with an oracle predictor—caused by WordPiece whitespace insertion around hyphenated words (`ई-सिगरेट` $\rightarrow$ `ई - सिगरेट`) and `[UNK]` emissions on rare conjuncts (12/2,000 lines). Replacing WordPiece (`119,547` tokens) with a **116-character Devanagari `CharTokenizer`** eliminates this floor completely (`0.00%`) while shrinking the output projection head by $>1000\times$.
2. **Spatial Resolution & Aspect Ratio Preservation:**
   Line images in the dataset have wide aspect ratios (`~900 × 64`). Padding into a square `384 × 384` canvas (`resize_pad`) shrinks vertical glyph height to ~27 pixels (<2 `DeiT` `16×16` patches tall), degrading CER by `+2.58` pp. Using a rectangular `128 × 512` input in the CRNN preserves horizontal character separation and diacritic (*matra* / *shirorekha*) clarity.
3. **Destructive Blur vs. Geometric Warping:**
   Aggressive `GaussianBlur` ($\sigma \in [0.1, 5.0]$, kernel $5\text{–}9$) blurs thin Devanagari top-bars and vowel diacritics, whereas `ElasticTransform` alone (`alpha=8.0, sigma=5.0`) yields the best TrOCR performance (`-3.15` pp CER vs. baseline).
4. **Initialization & Learning Rate Stability:**
   Re-initializing the `mBERT` decoder causes the steepest initialization degradation (`+2.91` pp CER), confirming that pretrained linguistic priors accelerate sequence alignment more than ImageNet visual priors (`-0.42` pp). Raising the learning rate to `1e-4` causes catastrophic divergence (`121.22%` CER) due to uninitialized cross-attention gradients.

---

