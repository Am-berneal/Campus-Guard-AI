# Cigarette detector (YOLO)

**Status:** processed dataset verified by the developer (2026-11-14). Training and inference scripts are ready. **Not trained yet**, so there are no validation results.

This model finds visible cigarette **objects** (boxes + confidence). It does **not** decide whether a person is smoking: a cigarette may be held, unlit, lying on a table, or a look-alike object. Combining it with SmokingCNN comes later.

| | SmokingCNN | Cigarette YOLO |
|---|---|---|
| Task | Image classification | Object detection |
| Labels | One per image (folder name) | One box per cigarette |
| Origin | Trained from scratch | Pretrained Ultralytics weights, fine-tuned by us (planned) |

## Dataset

### Processed dataset (verified by the developer's `prepare_dataset` run, 2026-11-14)

| Split | Images | With cigarette | Background | Used now? |
|---|---:|---:|---:|---|
| train | 1,080 | 853 | 227 | yes |
| val | 270 | 236 | 34 | yes (model selection) |
| test | 318 | 318 | 0 | **no**. Evaluated later with all models |

Validation: 0 invalid images, 0 invalid labels, 0 orphan labels, 0 train/test leakage, 0 warnings. One class: `0: cigarette`. The original export `Cigarettes Detection.v13i.yolov8` (Roboflow) is unchanged. **Licence and source details still need to be copied here** from the `data.yaml`/README output.

### Preparation: `prepare_dataset.py`

It reads the original dataset and writes a **separate copy**:

```
ai-service/datasets/smoking_yolo/
├── train/images, train/labels   ≈80% of valid source-train images
├── val/images,   val/labels     ≈20% of valid source-train images (seed 42)
├── test/images,  test/labels    the source test set (valid files only)
├── data.yaml                    absolute path + train/val/test + {0: cigarette}
├── prepare_report.json          every exclusion with its reason, counts, file lists
└── .campussentinel_prepared     marker so --overwrite only ever deletes this folder
```

| Check | Rule | If it fails |
|---|---|---|
| Image decodes | Full Pillow decode | Excluded and reported |
| Label exists | Same file stem as the image | Excluded ("missing label") |
| Unique stems | No `a.jpg` + `a.png` (case-insensitive, as on Windows) | Both excluded, so files can't overwrite each other |
| Line format | Exactly 5 values: `class x y w h` | Excluded. Polygon (segmentation) lines are named as such |
| Class ID | Integer `0` only | Excluded |
| Coordinates | 0..1, w > 0, h > 0, box inside image (±0.5%) | Excluded |
| Empty label file | Valid **background** image (no cigarette) | Kept and counted |
| Orphan labels, non-image files | — | Reported only |
| Duplicate boxes in one file | — | Warning only |

**Why the split groups images.** Roboflow exports often contain several augmented copies of one photo, named `<photo>_jpg.rf.<hash>.jpg`. If copies of one photo land in both train and val, validation scores look better than they really are. So copies of one photo stay in the **same** split. The script also checks for **leakage against the test set**: if any copy of a photo is identical to a test image, or comes from the same source photo, all its copies are left out of train and val.

**The test set:**
- The original test folder is never modified.
- The processed `test/` folder is a copy of the valid test files.
- Any test file that fails validation is reported. Test evaluation is postponed until all models are built.

## Commands (CMD, from `ai-service`)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat
python -m training.smoking.yolo.prepare_dataset --dry-run
python -m training.smoking.yolo.prepare_dataset
```

| Option | Use |
|---|---|
| `--dry-run` | Validate and report only |
| `--overwrite` | Rebuild `datasets\smoking_yolo` (only if this script created it) |
| `--source`, `--output` | Use other folders |
| `--val-fraction`, `--seed` | Change the split (defaults 0.2 and 42) |

Needs only Pillow, which is already installed with torchvision. Nothing gets installed and no training runs.

## Model

| Item | Value |
|---|---|
| Architecture | **YOLO26n** (Ultralytics nano detector, ~2.4 M parameters, 5.4 GFLOPs at 640) |
| Base weights | `models/smoking/yolo/base/yolo26n.pt`. **Pretrained by Ultralytics on COCO**, which has no cigarette class. Downloaded on the first training run |
| Our model | `models/smoking/yolo/finetuned/best.pt`. **Fine-tuned by us** for one class, `0: cigarette` |
| Metadata | `models/smoking/yolo/finetuned/best.json`: provenance, dataset counts, training arguments, **validation** metrics |
| Ultralytics | 8.4.212 (pinned in `ai-service/requirements.txt`) |

Transfer learning: the backbone already recognises generic shapes, edges and textures from COCO. Fine-tuning replaces the 80-class head with a 1-class head and retrains on cigarette boxes.

## Training: `train_yolo.py`

| Setting | Value | Note |
|---|---|---|
| Data | `datasets/smoking_yolo/data.yaml` | Checked before training: 1 class, train and val image/label counts match |
| Splits | train → training, val → validation and model selection | **test is never loaded** |
| Epochs | max 50 | Early stopping `patience=15` (epochs without val improvement) |
| Image size | 640 | |
| Batch | 8 | Increase with `--batch 16` only if the `GPU_mem` column stays below ~3.5 GB at batch 8 |
| AMP | on | Ultralytics runs its own AMP check on CUDA and falls back to FP32 if it fails |
| Seed | 42, `deterministic=True` | |
| Optimiser, learning rate, augmentation | Ultralytics defaults (`optimizer=auto`) | Recorded in the run's `args.yaml` and in `best.json` |
| Best checkpoint | Epoch with the highest validation fitness, which is **val mAP@50-95** in Ultralytics 8.4.212 | |

After training, the script:
1. re-validates `best.pt` on **val** (confusion matrix and PR curves in `runs/smoking_yolo/<run>_val/`);
2. copies it to `models/smoking/yolo/finetuned/best.pt`;
3. writes `best.json`.

Run folder `runs/smoking_yolo/<run>/` contains `results.csv` (per-epoch losses and val P/R/mAP), `results.png`, confusion matrices, PR/F1 curves, `args.yaml`, `weights/best.pt` and `weights/last.pt`.

**Resume:** `--resume` continues the most recent **unfinished** run. A finished run is refused: Ultralytics would otherwise silently start a new 100-epoch run with default settings, which I saw happen in testing.

**Out of GPU memory:** the script prints a clear message suggesting `--batch 4` or `--imgsz 512`.

## Inference: `cigarette_detector.py` + `infer_yolo.py`

```python
from training.smoking.yolo.cigarette_detector import CigaretteDetector
det = CigaretteDetector()               # loads finetuned/best.pt, GPU if available, float16 on GPU
dets = det.detect(frame_bgr)            # also accepts a file path
# [{"class_id": 0, "class_name": "cigarette", "confidence": 0.81,
#   "bbox_xyxy": [x1, y1, x2, y2], "bbox_xyxy_norm": [...]}]
dets, annotated = det.detect_and_draw(frame_bgr)
det.close()
```

- The detector refuses any weights file that isn't a single-class `cigarette` detector, such as the COCO base model.
- The default `conf=0.25` is the **untuned** Ultralytics default. Thresholds may be tuned on **val only**.

## Commands (CMD, from `ai-service`)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat
python -m training.smoking.yolo.train_yolo
python -m training.smoking.yolo.train_yolo --resume
python -m training.smoking.yolo.infer_yolo --image path\to\photo.jpg
python -m training.smoking.yolo.infer_yolo --video path\to\clip.mp4 --show
```

Inference output goes to `runs\smoking_yolo\inference\<name>_<time>\`: an annotated image or `annotated.mp4`, plus `detections.json`.

## Validation results

_Not available yet. They will be filled in from the developer's training output (val split only)._

## Next stages

1. Test-set evaluation together with the other models.
2. Fusion with SmokingCNN plus person association (later).
