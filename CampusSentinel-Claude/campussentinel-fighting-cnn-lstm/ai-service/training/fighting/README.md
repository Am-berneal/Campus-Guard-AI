# Fighting detection: preprocessing + CNN-LSTM classifier (RWF-2000)

**Status:**
- **Preprocessing:** verified on the real dataset by the developer (2,000 ok clips, 0 failures).
- **CNN-LSTM model and its training, evaluation, prediction and verification scripts:** implemented and unit-tested on **synthetic** clips only. **Not trained on RWF-2000 yet**, so there are no real metrics.

| File | Purpose |
|---|---|
| `prepare_data.py`, `dataset.py`, `inspect_processed.py` | Preprocessing and `FightingVideoDataset` (unchanged) |
| `config.py` | `ModelConfig`, `TrainConfig`, paths, class mapping, normalisation |
| `model.py` | `CNNLSTM`: ResNet18 frame features → LSTM → head (1 logit) |
| `metrics.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report (no extra dependency) |
| `checkpoint.py` | Save/load checkpoints (`weights_only=True`), rebuilds the architecture from the stored config |
| `train.py` | Training with validation every epoch, early stopping, ReduceLROnPlateau, optional AMP, resume |
| `evaluate.py` | Independent **validation** evaluation of a checkpoint |
| `predict.py` | `FightPredictor` for a `.npy` clip or raw video, plus CLI |
| `verify_checkpoint.py` | Load, rebuild, forward pass, shape and finiteness checks |
| `tests/` | 31 unit tests: 17 preprocessing + 14 model/pipeline (1 GPU-only test is skipped without CUDA) |

## What preprocessing does

For every `.avi` file in `train/` and `val/` (the original split is kept exactly; there is no test split here):

1. **Open** the video with OpenCV and read its FPS, frame count, width and height.
2. **Pick 16 frame indices**, evenly spaced over the whole clip, including the first and last frame. For 150 frames that gives `0 10 20 30 40 50 60 70 79 89 99 109 119 129 139 149`.
3. **Read sequentially.** Every frame is grabbed, but only the 16 selected ones are decoded. This avoids unreliable seeking in AVI files and reveals the true length. If a damaged file has fewer frames than its header claims, the indices are recomputed from the real length and the manifest gets a `WARNING:` note.
4. **Letterbox each frame:** scale so the longer side is 224 (aspect ratio kept), then centre it on a black 224×224 square.
5. **Convert BGR → RGB** and keep **uint8**.
6. **Save** `(16, 224, 224, 3)` uint8 as `.npy`. The file is written to a temporary name and then renamed, so an interruption never leaves a half-written file that looks valid.

The original dataset is only read. Nothing in it is changed, moved or copied.

### Why uniform sampling?

A fight can happen anywhere in the 5-second clip. Spreading 16 samples over the whole video (about one every 0.33 s at 30 fps) covers the full action, where a random or initial window could miss it. It's also deterministic: the same video always gives the same frames, which makes experiments reproducible.

The trade-off: fast motions between samples are not seen. A temporal model (CNN + LSTM) learns from how the 16 frames change, which a single frame can't show. That is the difference between a "fight-looking frame" and a fighting **action**.

### Why preserve aspect ratio?

RWF-2000 resolutions range from 320×240 to 1280×720. Stretching a 16:9 frame into a square makes people look thin and distorts their limbs, and by different amounts in different videos. That is an artefact the model could learn instead of real motion. Letterboxing keeps body shapes consistent. The cost is some black padding: a 1280×720 frame becomes 224×126 of content.

### Why keep uint8 on disk?

- uint8 clips are 4× smaller than float32.
- Normalisation belongs to the model and can change without re-running preprocessing.
- `FightingVideoDataset` converts to float and normalises at load time.

## Output

```
D:\CampusSentinel-Datasets\fighting_processed\
├── train\Fight\<original-name>.npy      label 1
├── train\NonFight\<original-name>.npy   label 0
├── val\Fight\…                          label 1
├── val\NonFight\…                       label 0
├── manifest.csv                         one row per source video
├── preprocessing_report.json            counts, failures, warnings, config, disk usage
└── _previews\                           frame grids written by inspect_processed.py
```

`manifest.csv` columns: `source_path, npy_path` (relative to the output folder), `split, class_name, label, frame_count, fps, width, height, frame_indices, output_shape, status` (`ok`/`failed`), `error` (failure reason, or a `WARNING:` note).

**Disk space:** 16 × 224 × 224 × 3 bytes = 2.30 MB per clip. The full dataset of 2,000 clips needs about **4.5 GB** on D:.

## Commands (CMD, from `ai-service`)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat

rem 1) quick trial: 2 videos per class (8 videos)
python -m training.fighting.prepare_data --limit 2

rem 2) check the output
python -m training.fighting.inspect_processed

rem 3) full dataset (already-processed files are skipped)
python -m training.fighting.prepare_data

rem unit tests (synthetic videos, temporary folders)
python -m unittest discover -s training\fighting\tests -t . -v
```

The defaults are `--source D:\CampusSentinel-Datasets\fighting` and `--output D:\CampusSentinel-Datasets\fighting_processed`. Other options: `--frames`, `--size`, `--verbose`.

Exit code: 0 = all good, 2 = some videos failed or folders were missing (details printed and saved in the report).

### Resume and `--force`

- **Resume (default):** a video is skipped when its `.npy` exists **and** loads with the right shape and dtype. A damaged or partial `.npy` is regenerated automatically.
  - `manifest.csv` is merged with the previous one and saved every 25 videos. A run interrupted with Ctrl+C can simply be started again.
  - If the manifest is deleted, it is rebuilt from the valid files without reprocessing them.
- **`--force`:** regenerates every selected video. Combine it with `--limit` to regenerate only a few.
- **`--limit N`:** processes the first N videos (alphabetical) per class. A later full run skips them.

### Inspecting output (`inspect_processed.py`)

It checks:
- that every `ok` row has its file;
- for `.npy` files not listed in the manifest;
- that labels match folders.

It then loads 2 random clips per split and class, prints dtype, shape, value range and source metadata, and saves a 4×4 grid of the 16 frames to `_previews\`. Open a few previews and confirm that `Fight` clips really show fights.

## Loading in PyTorch

```python
from torch.utils.data import DataLoader
from training.fighting.dataset import FightingVideoDataset

manifest = r"D:\CampusSentinel-Datasets\fighting_processed\manifest.csv"
train_ds = FightingVideoDataset(manifest, split="train")
val_ds = FightingVideoDataset(manifest, split="val")
print(len(train_ds), train_ds.class_counts())

video, label = train_ds[0]      # float32 (16, 3, 224, 224), ImageNet-normalised; label: torch.long (0/1)
loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)   # num_workers=0 on Windows
```

- Only rows with `status == ok` are used.
- A missing file raises `FileNotFoundError`, and a wrong shape or dtype raises `ValueError`. Both name the file.
- `normalize=False` returns values in [0, 1].
- `transform=` takes a function applied to the [0, 1] tensor before normalisation (for later augmentation).
- Memory: one clip is about 9.6 MB as float32, so batch 4 is about 39 MB before the model.

## CNN-LSTM model

```
clip (B, 16, 3, 224, 224)  ImageNet-normalised
  │ reshape → (B·16, 3, 224, 224): every frame of every clip in ONE ResNet call
ResNet18, ImageNet-pretrained, final fc removed, FROZEN → (B·16, 512)
  │ reshape → (B, 16, 512): one 512-number description per frame, in time order
LSTM (batch_first, hidden 256, 1 layer by default)
  │ final hidden state of the last layer → (B, 256)
Head: Dropout(0.5) → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→1)
  ▼
(B, 1) raw logit. P(Fight) = sigmoid(logit). No sigmoid inside the model (BCEWithLogitsLoss).
```

**Why ResNet18?**
- It is small, about 11.2 M parameters, so it fits the RTX 4050's 6 GB with 64 frames per batch.
- Its ImageNet features (people, limbs, poses, textures) transfer well to a 1,600-clip dataset, which is too small to train a CNN from scratch.
- Frozen, it acts as a fixed feature extractor.

**Why an LSTM?** A single frame cannot reliably separate a fight from a hug, a dance or sport. The LSTM reads the 16 frame descriptions **in order** and learns how they change over time: fast repeated movement, people closing distance. That is the difference between a fight-looking frame and a fighting action.

**Frozen backbone details:**
- Frozen ResNet layers run under `torch.no_grad()`, so no activations are kept for backpropagation. That keeps memory low.
- They also stay in **eval mode** even during training, so their BatchNorm statistics are not overwritten by our small batches. A unit test checks this.
- Trainable parameters: **821,505** (LSTM 788,480 + head 33,025) out of 11,998,017 total.
- `--train-layer4` optionally fine-tunes ResNet18's last block with its own learning rate (`--backbone-lr`, default 1e-4). It is off by default.

**Shapes:** a single dataset sample is `(16, 3, 224, 224)` float32 with a `torch.long` label. A batch is `(B, 16, 3, 224, 224)` with `(B,)` labels. The model outputs `(B, 1)`, and the training loop converts labels to `(B, 1)` float for the loss.

## Training (`train.py`)

| Setting | Default | Option |
|---|---|---|
| Loss | BCEWithLogitsLoss | |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-2 | `--lr`, `--weight-decay` |
| Batch | 4 | `--batch-size` |
| Epochs | max 20 | `--epochs` |
| Early stopping | validation loss, patience 5 | `--patience` |
| LR schedule | ReduceLROnPlateau ×0.5, patience 2 | `--lr-patience` |
| Threshold | 0.5 | `--threshold` |
| Augmentation | whole-clip horizontal flip, p = 0.5, training only | `--no-hflip` |
| DataLoader | `num_workers=0`, `pin_memory` on CUDA, non-blocking copies | `--workers` |
| Mixed precision | off | `--amp` (CUDA only) |
| LSTM | hidden 256, 1 layer, dropout 0.3 between layers | `--hidden-size`, `--num-layers`, `--lstm-dropout`, `--head-dropout` |
| Seed | 42 (shuffle re-seeded per epoch, so resume keeps the order) | `--seed` |

Each epoch logs train and validation loss, accuracy, precision, recall, F1 and ROC-AUC. Probabilities are `sigmoid(logits)`, predictions use the threshold, and ROC-AUC uses the probabilities.

**Outputs:**
- `ai-service\runs\fighting\train_<time>\`: `history.csv`, `loss.png`, `accuracy.png`, `val_confusion_matrix_best.png`, `val_classification_report_best.txt`, `train.log`.
- `ai-service\models\fighting\cnn_lstm\`:
  - `best.pt`: lowest validation loss;
  - `last.pt`: latest epoch, including optimizer, scheduler, scaler and history, used by `--resume`;
  - `best.json`: readable summary.

**The dataset location is never hard-coded.** Pass `--data-root` or set `CS_FIGHTING_DATA`.

### Checkpoint format

`best.pt` and `last.pt` are dicts with:
- `model_state_dict` (including ResNet18) and `optimizer_state_dict`;
- `epoch`, `best_val_loss`, `best_epoch` and `val_metrics`;
- `model_config` and `train_config`;
- `class_mapping` (`{"0": "NonFight", "1": "Fight"}`), `normalization` (ImageNet mean/std) and `input_clip_shape` (`[16, 224, 224, 3]`);
- `torch_version`.

`last.pt` also holds `scheduler_state_dict`, `scaler_state_dict`, `history` and `epochs_no_improve`.

Loading never downloads anything: `checkpoint.load_checkpoint()` rebuilds `CNNLSTM` from `model_config` and loads all weights with `torch.load(weights_only=True)`.

## Evaluation (`evaluate.py`): validation only

RWF-2000 has **no test split**, and none is invented. `evaluate.py` scores a checkpoint on the **validation** split and writes these files to `runs\fighting\eval_<time>\`:
- `val_metrics.json`
- `val_classification_report.txt`
- `val_confusion_matrix.png`
- `val_predictions.csv`

Because validation loss also chose the best epoch, these numbers are **optimistic** and must be reported as validation results, never as test results.

## Prediction (`predict.py`)

```python
from training.fighting.predict import FightPredictor
p = FightPredictor(r"models\fighting\cnn_lstm\best.pt")      # GPU if available
p.predict_npy(r"D:\CampusSentinel-Datasets\fighting_processed\val\Fight\xxx.npy")
p.predict_video(r"path\to\video.avi")    # same preprocessing as prepare_data.py
# {"predicted_class": "Fight", "fight_probability": 0.87, "nonfight_probability": 0.13, "threshold": 0.5}
```

- Input clips must be uint8 `(16, 224, 224, 3)`. Anything else raises `ValueError`.
- A prediction is a **potential incident for administrator review**. Working inference does not make the model production-ready.

## Commands (CMD, from `ai-service`)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat
set CS_FIGHTING_DATA=D:\CampusSentinel-Datasets\fighting_processed

python -m unittest discover -s training\fighting\tests -t . -v
python -m training.fighting.train --epochs 20 --batch-size 4 --workers 0 --amp
python -m training.fighting.train --epochs 20 --batch-size 4 --workers 0 --amp --resume
python -m training.fighting.verify_checkpoint --checkpoint models\fighting\cnn_lstm\best.pt
python -m training.fighting.evaluate --checkpoint models\fighting\cnn_lstm\best.pt
python -m training.fighting.predict --checkpoint models\fighting\cnn_lstm\best.pt --clip "D:\CampusSentinel-Datasets\fighting_processed\val\Fight\<name>.npy"
```

`set` lasts only for the current CMD window. Alternatively, add `--data-root "D:\CampusSentinel-Datasets\fighting_processed"` to train and evaluate.

The first training run downloads ResNet18's ImageNet weights (about 45 MB) into the PyTorch cache (`%USERPROFILE%\.cache\torch`). This needs internet once.

## GPU memory

- With the frozen backbone, each step pushes `B × 16` frames (64 at batch 4) through ResNet18 **without gradients**. Only the LSTM and head keep activations for backpropagation, so memory per step is small. `--amp` roughly halves activation memory and speeds up the CNN.
- With `--train-layer4`, layer4 activations are kept for backprop, so memory rises. Start at batch 2–4 there.
- Out-of-memory errors are caught with a message suggesting a smaller batch or `--amp`, and `--resume` continues from `last.pt`.
- Data loading: each clip is a 2.3 MB `.npy` read from D:. With `num_workers=0`, disk reads and the GPU work alternate. If the GPU sits idle, try `--workers 2` once the tests pass. That's optional; workers caused problems on Windows before.

Actual memory use and epoch times on the RTX 4050 are **not measured yet**. They will be recorded from the first run.

## Known limitations

- **No test set.** Validation is used for model selection and reporting, so the reported performance is optimistic.
- **Domain gap.** RWF-2000 is YouTube surveillance-style footage. Campus cameras (angle, distance, lighting, crowd density) differ, and accuracy there is unknown.
- **16 frames over 5 s** (about 3 fps) can miss very short actions. Clip-level labels mean the whole 5 s is labelled Fight even if fighting fills only part of it.
- **The frozen ImageNet features** were never trained on motion. If validation results plateau low, `--train-layer4` is the first thing to try.
- **Look-alike activities:** sports, play-fighting, crowded scenes and dancing can be confused with fights. Every prediction needs human review.
- **No person localisation.** The model says *whether* a clip looks like a fight, not *who* is fighting or *where* in the frame.

## Dataset notes

- RWF-2000: 2,000 surveillance clips, 5 s each, `Fight`/`NonFight`, with an official train (1,600) and val (400) split. The developer inspected 20 random videos: all readable, 30 fps, 150 frames, 320×240 to 1280×720.
- **Licence and terms of use:** to be recorded from the official RWF-2000 source before any results are reported.
