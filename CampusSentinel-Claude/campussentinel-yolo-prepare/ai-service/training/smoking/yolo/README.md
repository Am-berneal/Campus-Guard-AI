# Cigarette detector (YOLO)

**Status:** dataset preparation script ready. **No processed dataset has been verified yet, and no model is trained.**

This model finds visible cigarette **objects** (boxes + confidence). It does **not** decide whether a person is smoking: a cigarette may be held, unlit, lying on a table, or a look-alike object. Combining it with SmokingCNN comes later.

| | SmokingCNN | Cigarette YOLO |
|---|---|---|
| Task | Image classification | Object detection |
| Labels | One per image (folder name) | One box per cigarette |
| Origin | Trained from scratch | Pretrained Ultralytics weights, fine-tuned by us (planned) |

## Dataset

### Facts reported by the developer (2026-11-14, not yet confirmed by the script)

| Item | Value |
|---|---|
| Location | `C:\Users\hp\Projects\CampusSentinel-Claude\Cigarettes Detection.v13i.yolov8` (kept unchanged) |
| Format | YOLO detection export (folder name suggests a Roboflow "v13" YOLOv8 export) |
| `train/` | 1,350 images, 1,350 label files |
| `test/` | 318 images, 318 label files |
| `valid/` | none |
| Classes (`data.yaml`) | 1: `cigarette` |

The **source, licence and annotation validity** get recorded here from the script's output, which prints `data.yaml` and any Roboflow README. Until then they are **unverified**.

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

## Next stages (not implemented yet)

1. Fine-tune a pretrained Ultralytics nano detector (Ultralytics 8.4.212 is pinned in `ai-service/requirements.txt`) on `datasets/smoking_yolo/data.yaml`. Select the model on **val**.
2. Inference script plus a reusable function for the later FastAPI service.
3. Test-set evaluation together with the other models.

Weights will go in `ai-service/models/smoking/yolo/` (`base/` = Ultralytics pretrained, `finetuned/` = ours). Runs will go in `ai-service/runs/smoking_yolo/`.
