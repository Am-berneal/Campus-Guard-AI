# Model cards

One file per model, created when the model is actually trained or integrated:

| File | Model | Origin |
|---|---|---|
| `smoking-cnn.md` | Smoking / not-smoking CNN | Trained **from scratch** (Phase 3) |
| `cigarette-yolo.md` | Cigarette detector | Pretrained YOLO **fine-tuned** by us (Phase 3) |
| `smoking-hybrid.md` | CNN + YOLO fusion | Fusion rule tuned on validation set (Phase 3) |
| `fight-detector.md` | Video fight classifier | Trained by us — architecture decided in Phase 4 |
| `person-detector.md` | Person detection + tracking | **Pretrained** YOLO, not trained by us (Phase 5) |

## Template

```markdown
# <Model name>

- Origin: from scratch | fine-tuned from <base> | pretrained (source: …)
- Version / date trained:
- Weights file: ai-service/models/<file> (size, SHA-256)
- Training notebook/script:

## Architecture
Why this architecture was chosen.

## Data
Dataset(s), split sizes, class counts, preprocessing, augmentation.

## Input / output
Input shape, preprocessing at inference, output format.

## Results (test split, measured — not estimated)
| Metric | Value |
Confusion matrix. Training curves. Hardware and training time.

## Limitations and failure cases
Known false positives / negatives, lighting, camera angles, bias.

## Intended use
Flags possible events for human review only. Never used for automatic action.
```
