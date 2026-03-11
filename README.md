# 🧠 Age Classification — Deep Learning Spring 2026

> Binary face classification (**Young = 0**, **Old = 1**) using **ResNet-18 trained from scratch**.  
> Approach: CLIP knowledge distillation + EMA weight averaging + Test-Time Augmentation.

---

## 📁 File Structure

```
├── b23cm1033.ipynb       # Full training notebook (with outputs)
├── b23cm1033.py          # Model definition — _Net class + build_model()
├── b23cm1033.pth         # Saved model (best EMA weights)
├── b23cm1033.pdf         # 1-page report
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install torch torchvision open_clip_torch Pillow numpy
```

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| PyTorch | 2.5.1 (CUDA 12.1) |
| open_clip_torch | any recent |
| Pillow | any recent |

> `open_clip_torch` is used **only during training** to extract CLIP embeddings.  
> It is **not required at inference**.

---

## 🚀 Running Evaluation

```bash
python evaluate_submission_student.py \
  --model_path b23cm1033.pth \
  --model_file b23cm1033.py \
  --data_dir dataset
```

Expected output:
```
Device: cuda
Importing model definitions from: b23cm1033.py
Loading model from: b23cm1033.pth
Running inference on valid images...
  134 images processed.

Accuracy: 99.25%  (133/134)
```

---

## 🔍 How It Works

### The Core Problem: Spurious Correlation

The training set contains a hidden bias — all **Young (class 0)** faces are **female** and all **Old (class 1)** faces are **male**. A naively trained model learns gender as a shortcut for age, which fails badly on any balanced test set. Breaking this shortcut is the central challenge.

---

### Stage 1 — CLIP Feature Distillation (150 epochs)

CLIP ViT-B/32 is pretrained on 400M+ image-text pairs and encodes genuine visual properties — wrinkles, skin texture, facial structure — **without** the gender-age bias in our dataset.

We use CLIP **purely as a frozen teacher**:

1. Extract 512-d CLIP embeddings for all 18,466 training + validation images
2. Attach a temporary projection head (`512 → 512 → 512`) to the ResNet-18 backbone
3. Train the backbone to match CLIP features via cosine similarity loss:

```
L = 1 - cos(f_proj, f_CLIP)
```

4. Discard the projection head — only the backbone weights are kept

This forces the backbone to learn **bias-free, semantically meaningful features** before any age-specific classification begins.

```
Stage 1 Loss:  0.1959 → 0.0700  (150 epochs)
```

> CLIP is never submitted and never called at inference.

---

### Stage 2 — Classification Fine-tuning (80 epochs)

The distilled backbone is paired with a new MLP classifier head and fine-tuned end-to-end.

**Architecture:**
```
ResNet-18 backbone (weights=None)
    └── 512-d feature vector
            └── Linear(512→256) → BatchNorm → ReLU → Dropout(0.3)
                    └── Linear(256→64) → BatchNorm → ReLU → Dropout(0.3)
                            └── Linear(64→2)
```

**Key strategies for generalisation:**

| Strategy | Detail | Why it helps |
|----------|--------|-------------|
| Differential LRs | Backbone `1e-4`, Head `1e-3` | Preserves distilled features while head adapts fast |
| EMA (decay=0.9997) | Slow-moving weight copy evaluated each epoch | Averages out training noise, generalises better |
| Label Smoothing | `ε = 0.1` | Prevents overconfident predictions |
| GaussianBlur | `p=0.3` | Reduces reliance on sharp texture shortcuts |
| RandomGrayscale | `p=0.15` | Removes colour as a discriminative cue |
| RandomErasing | `p=0.25` | Forces learning of distributed features |
| ColorJitter | `(0.4, 0.4, 0.3, 0.1)` | Breaks brightness/saturation shortcuts |
| TTA at inference | 3 views averaged (normal + flip + center-crop) | Reduces prediction variance for free |

---

## 📊 Results

All numbers are on the **provided 134-image validation set**.

| Metric | Value |
|--------|-------|
| Stage 1 final loss | 0.0700 |
| Best EMA Val Accuracy | 99.25% (epoch 66 / 80) |
| TTA Val Accuracy | 99.25% |
| Eval script | 99.25% — 133 / 134 images |

> Hidden test set accuracy is pending evaluation.

---

## 📋 Compliance

- ✅ ResNet-18 trained **from scratch** (`weights=None`)
- ✅ CLIP used as **reference only** during training — not submitted
- ✅ **Single model** — no ensembles
- ✅ **Provided data only** — no external datasets
