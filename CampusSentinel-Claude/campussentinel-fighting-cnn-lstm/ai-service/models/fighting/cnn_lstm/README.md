# Fighting CNN-LSTM checkpoints

Written by `python -m training.fighting.train`:

- `best.pt` — lowest VALIDATION loss (model + optimizer state, config, class mapping, normalisation)
- `last.pt` — most recent epoch, full state for `--resume`
- `best.json` — readable summary with VALIDATION metrics (RWF-2000 has no test split)

Backbone: ResNet18 with ImageNet weights from TorchVision (pretrained, frozen); LSTM + head trained by us.
Checkpoints are created on the developer's machine and are not included in code ZIPs.
