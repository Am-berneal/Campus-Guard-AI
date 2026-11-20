# Model card — Cigarette detector (YOLO)

- **Origin:** pretrained Ultralytics base weights (COCO), **to be fine-tuned by us**. COCO contains no cigarette class.
- **Status:** planned; no dataset verified, **not trained**. No metrics exist yet.
- **Base model:** nano/small Ultralytics detection model — exact model chosen and recorded in Stage 5.
- **Dataset:** to be selected and verified in Stage 3 (source, license, bounding-box annotations).
- **Weights:** `ai-service/models/smoking/yolo/finetuned/best.pt` (after Stage 5).

## Metrics to report (test split)
Precision, recall, mAP@50, mAP@50-95 — measured values only.

## Limitations (expected)
- Cigarettes are small, thin objects; recall drops at low resolution and distance.
- Detection ≠ smoking: held, unlit or discarded cigarettes and look-alike objects.

## Intended use
Supporting evidence for possible smoking; administrator review required.
