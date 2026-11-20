# Fighting detection: video preprocessing (RWF-2000)

**Status:** preprocessing pipeline and PyTorch dataset are implemented and tested on **synthetic** videos only. Nothing has run on the real RWF-2000 data yet, and no model exists.

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

## Dataset notes

- RWF-2000: 2,000 surveillance clips, 5 s each, `Fight`/`NonFight`, with an official train (1,600) and val (400) split. The developer inspected 20 random videos: all readable, 30 fps, 150 frames, 320×240 to 1280×720.
- **Licence and terms of use:** to be recorded from the official RWF-2000 source before any results are reported.
