# CampusSentinel — Datasets

_Status: **plan only**. No dataset has been downloaded or verified yet. Nothing in this file is a verified fact until its "Verified" field is filled in with a date._

---

## Verification checklist (applied to every dataset)

1. Reliable source identified (original authors / official repository).
2. Official download page or repository recorded.
3. Download instructions tested (actually downloaded).
4. License and usage restrictions checked and recorded.
5. Structure documented (folders, file formats, label format).
6. Class counts measured (not copied from a paper).
7. Preprocessing script written (`training/<task>/prepare_data.py`).
8. Reproducible train/val/test split with a fixed seed; split file saved.
9. Leakage check (see below).
10. Imbalance and augmentation plan documented.

### Data-leakage rules
- **Video datasets:** split by **video**, never by frame. Frames of the same clip must all be in one split.
- **Image datasets scraped from video:** near-duplicate frames must stay in the same split. Use perceptual hashing to detect duplicates across splits.
- **Same person / same scene:** where source metadata allows, keep the same scene/person together.
- The **test split is touched once**, for final reported metrics. Tuning (thresholds, fusion weights) uses validation only.

---

## 1. Smoking — binary image classification (custom CNN)

| Field | Value |
|---|---|
| Purpose | Train CNN from scratch: `smoking` vs `not_smoking` |
| Candidate sources | _To be researched in Phase 3_ |
| Verified | No |
| License | _Unknown_ |
| Class distribution | _To be measured_ |

## 2. Cigarette — object detection (YOLO)

| Field | Value |
|---|---|
| Purpose | Fine-tune YOLO to detect `cigarette` boxes |
| Candidate sources | _To be researched in Phase 3_ (need bounding-box labels in YOLO format or convertible) |
| Base model | Ultralytics YOLO pretrained on COCO (COCO has **no** cigarette class, so fine-tuning is required) |
| Verified | No |
| License | _Unknown_ |

Note: detecting a cigarette does **not** prove a person is smoking (it could be held, unlit, on a table, or a false positive such as a pen).

## 3. Fighting — video action recognition

| Field | Value |
|---|---|
| Candidate | **RWF-2000** (Real-World Fighting, surveillance-style clips, two classes: Fight / NonFight) |
| Source | _To be verified in Phase 4 (original authors' repository)_ |
| Access | _To be verified._ It is believed that access requires a signed academic-use agreement; if so, the request should be sent early because approval takes time |
| License | _To be verified_ |
| Verified | No |
| Alternatives if unavailable | _To be researched_ (other publicly licensed violence/fight video datasets) |

Third-party re-uploads (e.g. on file-sharing or dataset-hosting sites) will **not** be used unless the original license allows redistribution.

## 4. Restricted zone — person detection

| Field | Value |
|---|---|
| Model | Pretrained Ultralytics YOLO (COCO `person` class) — **pretrained, not trained by us** |
| Training data | None required initially |
| Evaluation | Our own short recorded test clips with manually noted ground truth (zone entry times) |
