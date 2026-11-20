# YOLO cigarette detector weights

Nothing here yet.

```
yolo/
├── base/        PRETRAINED base weights downloaded from Ultralytics (not trained by us)
└── finetuned/   OUR fine-tuned best.pt (Stage 5) + training metadata
```

Rules:
- `base/` files keep their original Ultralytics filename; note source URL and download date.
- `finetuned/best.pt` is described as "fine-tuned by us from <base model> on <dataset>".
- Never place a third-party, already-trained cigarette detector here as if it were ours.
- Weights are copied here manually from Colab; not included in code ZIPs.
