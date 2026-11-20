# SmokingCNN checkpoints

Produced by `python -m training.smoking.cnn.train` (trained from scratch, no pretrained weights).

| File | Content |
|---|---|
| `smoking_cnn_best.pt` | Best epoch by validation loss: weights + architecture config, class mapping, image size, normalisation, best epoch, validation metrics |
| `smoking_cnn_best.json` | The same metadata, readable |

Load for inference:

```python
from training.smoking.cnn.checkpoint import load_smoking_cnn
model, meta = load_smoking_cnn("models/smoking/cnn/smoking_cnn_best.pt", device="cuda")
```

Weight files are created on the developer's machine and are not included in code ZIPs.
