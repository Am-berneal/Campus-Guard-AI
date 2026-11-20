# SmokingCNN — custom CNN trained from scratch

**Status:** architecture verified (Stage 1), data pipeline verified on the real dataset (Stage 2), training script ready (Stage 3). **Not trained yet.**

| Item | Value |
|---|---|
| Task | Binary image classification: `not_smoking` (0) vs `smoking` (1) |
| Origin | **Trained from scratch** — random initial weights, no pretrained backbone |
| Input | RGB tensor `(N, 3, 224, 224)`, float, normalised (normalisation defined in Stage 2) |
| Output | `(N, 1)` raw logits. Probability = `sigmoid(logit)` |
| Parameters | 454,945 (all trainable), ≈1.74 MB as float32 |
| Loss | `BCEWithLogitsLoss` |
| Files | `model.py`, `inspect_model.py`, `prepare_data.py`, `dataset.py`, `check_batches.py`, `train.py`, `metrics.py`, `checkpoint.py`, `verify_checkpoint.py` |

## Architecture

```
input 3×224×224
 ├─ Block 1: Conv3×3(3→32)    → BN → ReLU → MaxPool2   → 32×112×112
 ├─ Block 2: Conv3×3(32→64)   → BN → ReLU → MaxPool2   → 64×56×56
 ├─ Block 3: Conv3×3(64→128)  → BN → ReLU → MaxPool2   → 128×28×28
 ├─ Block 4: Conv3×3(128→256) → BN → ReLU → MaxPool2   → 256×14×14
 └─ Head: AdaptiveAvgPool → Flatten(256) → Linear(256) → ReLU → Dropout(0.5) → Linear(1)
```

### Parameter count (hand-computed, verified by `inspect_model.py`)

| Part | Formula | Params |
|---|---|---:|
| Conv weights | 3·3·(3·32 + 32·64 + 64·128 + 128·256) | 387,936 |
| BatchNorm (γ, β) | 2·(32 + 64 + 128 + 256) | 960 |
| Linear 256→256 | 256·256 + 256 | 65,792 |
| Linear 256→1 | 256 + 1 | 257 |
| **Total** | | **454,945** |

85% of the parameters are in the convolutions, mostly Block 4 (294,912): parameters grow with *input channels × output channels*.

## Design decisions

- **3×3 kernels, padding 1.** Padding keeps width/height unchanged inside the conv; only MaxPool shrinks the image.
- **`bias=False` in convolutions.** BatchNorm subtracts the batch mean right after the conv, which cancels any bias; BN's own β replaces it. Mathematically equivalent to the spec, with fewer useless parameters.
- **BatchNorm** normalises each channel's activations per batch. It makes training from scratch much more stable and less sensitive to the learning rate. It behaves differently in `train()` (batch statistics) and `eval()` (running averages) — always call `model.eval()` for validation/inference.
- **AdaptiveAvgPool** averages each of the 256 feature maps to one number. The head then sees *"how strongly does feature k appear anywhere in the image"*, not *where*. It also replaces a huge `Flatten(256·14·14 = 50,176) → Linear` layer, which would add ~12.8 M parameters and overfit a ~5,700-image dataset.
- **Dropout 0.5** randomly zeroes half the hidden units during training, so the head cannot rely on any single feature. Off in `eval()`.
- **No sigmoid in `forward()`.** `BCEWithLogitsLoss` computes sigmoid + log in one numerically stable step (avoids `log(0)` when a probability saturates to exactly 0 or 1). Use `predict_proba()` when a probability is needed.
- **Initialisation.** Kaiming (He) init for conv/linear layers, suited to ReLU. The output layer starts with tiny weights so every initial prediction is ≈ 0.5 and the starting loss ≈ ln 2 ≈ 0.693. A starting loss far from 0.693 would suggest a bug.

## What the CNN can and cannot "see"

**Receptive field** = the patch of the input image that influences one cell of the final 14×14 feature map. For this architecture it is **46×46 pixels** of the 224×224 input.

So each final feature describes a local patch, roughly the size of a hand-plus-cigarette or a face region if the person fills the frame. Global average pooling then asks *"does any patch look like X?"*. Practical consequences to watch for in Stage 2 results:

- It may learn **local cues** (a thin bright stick, a hand near the face, smoke texture) well.
- It cannot directly reason about **relationships across the whole image** (e.g. hand-at-mouth posture of a small, distant person).
- **Distant, small people in CCTV frames** are a serious limitation: a person occupying 40×80 pixels of a 1080p frame becomes a few pixels after resizing to 224. In the full pipeline the CNN will likely need **person crops** (from YOLO person detection) rather than whole frames. This is planned for evaluation, not assumed.

We keep the specified architecture. Changing depth (e.g. two convs per block) is only justified if Stage 2 results show underfitting, and would be documented as a separate experiment.

## GPU memory and batch size

Weights are small (1.7 MB). Memory during training is dominated by **activations** stored for backpropagation. Block 1 alone produces 32×224×224 ≈ 1.6 M values per layer output per image, and conv, BN and ReLU outputs at that resolution are each kept.

`inspect_model.py --memory-probe` measures the real peak memory of one training step (forward + backward + Adam) on your GPU for batch sizes 16–128, with and without mixed precision (`--amp`). Use the measured numbers, not estimates, to choose:

- **Colab (typically a T4 with ~15 GB):** batch 32 (the spec) will fit with a large margin.
- **Local RTX 4050 (6 GB, ~5.5 GB free):** probe results decide; leave ≥ 1 GB headroom for the desktop and data loading.
- Changing batch size changes training dynamics (noisier gradients at smaller batches). We start with 32 and only change it with a recorded reason.

## Stage 1 checks (`inspect_model.py`)

| Check | Why it matters |
|---|---|
| Parameter count = 454,945 | The code matches the specification |
| Output shape (4, 1) | One logit per image |
| No Sigmoid module | Required by `BCEWithLogitsLoss` |
| Backward pass, initial loss ≈ 0.69 | Gradients reach every layer; initial predictions are neutral |
| Eval mode deterministic | Dropout disabled, BatchNorm frozen at inference |
| Seeded init reproducible | Same seed → same starting weights |
| Rejects 1-channel input | Catches preprocessing bugs early |
| `predict_proba()` in [0, 1] | Sigmoid only where intended |

These checks use **random tensors**. Passing them says nothing about accuracy.

---

## Commands (Command Prompt, from `ai-service`)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat
```

| Step | Command |
|---|---|
| Architecture check | `python -m training.smoking.cnn.inspect_model` |
| Data check only | `python -m training.smoking.cnn.prepare_data --check-only` |
| Create split (seed 42) | `python -m training.smoking.cnn.prepare_data` |
| Batch checks | `python -m training.smoking.cnn.check_batches` |
| **Train** | `python -m training.smoking.cnn.train` |
| Resume | `python -m training.smoking.cnn.train --resume runs\smoking_cnn\last.pt` |
| Verify saved model | `python -m training.smoking.cnn.verify_checkpoint` |

Useful `train.py` options: `--batch-size 16` (if out of memory), `--num-workers 2` (if data loading errors), `--no-amp`, `--epochs`, `--patience`, `--lr-patience 0` (disable LR reduction).

## Data pipeline (Stage 2)

- `prepare_data.py` decodes every image, finds exact duplicates (SHA-256) and writes `splits/split_seed42.json` (stratified 80/20, seed 42). The dataset itself is never modified.
- `dataset.py`: EXIF-orient → RGB → resize 224×224 → [0, 1] → normalise (ImageNet mean/std constants). Training adds horizontal flip and ±10° rotation; validation/test are never augmented.

## Training (Stage 3)

| Setting | Value |
|---|---|
| Optimiser | Adam, lr 0.001 |
| Loss | BCEWithLogitsLoss |
| Batch | 32, AMP (float16) on CUDA |
| Epochs | max 20, early stopping on validation loss (patience 5, min improvement 1e-4) |
| LR schedule | ReduceLROnPlateau: halve LR after 2 epochs without val-loss improvement |
| Seed | 42 (shuffle order re-seeded per epoch → resume keeps the same order) |
| Test split | **never loaded** |

Outputs:

| File | Content |
|---|---|
| `models/smoking/cnn/smoking_cnn_best.pt` | Best model (lowest val loss) + metadata. Loads with `weights_only=True` via `checkpoint.load_smoking_cnn()` |
| `models/smoking/cnn/smoking_cnn_best.json` | Same metadata, human-readable |
| `runs/smoking_cnn/last.pt` | Full state after the latest epoch, for `--resume` |
| `runs/smoking_cnn/history.csv` | Per-epoch losses and validation metrics |
| `runs/smoking_cnn/loss.png`, `accuracy.png` | Curves |
| `runs/smoking_cnn/train.log` | Console log (appended across runs) |

Reading the metrics:
- **Train accuracy** is measured on augmented images with dropout active, so it can be slightly *lower* than validation accuracy early on. That is normal.
- Validation precision/recall/F1 use threshold 0.5. **ROC-AUC** ignores the threshold.
- Selecting the best epoch on validation loss means validation numbers are slightly optimistic; the untouched test set gives the honest final number.
