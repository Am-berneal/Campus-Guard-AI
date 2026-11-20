"""Save and load CNN-LSTM checkpoints.

Checkpoints contain only tensors and plain Python values, so they load with
torch.load(weights_only=True). Loading never downloads ImageNet weights: the full
state dict (including ResNet18) is stored in the checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from training.fighting.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, CLIP_SHAPE, ModelConfig
from training.fighting.model import CNNLSTM

FORMAT_VERSION = 1


def save_checkpoint(path: Path, model: CNNLSTM, optimizer, epoch: int, best_val_loss: float,
                    train_config: dict, val_metrics: dict | None = None, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_name": "CNNLSTM (ResNet18 + LSTM)",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "val_metrics": val_metrics or {},
        "model_config": model.cfg.to_dict(),
        "class_mapping": {str(k): v for k, v in CLASS_NAMES.items()},
        "positive_class": "Fight",
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "input_clip_shape": list(CLIP_SHAPE),
        "train_config": train_config,
        "torch_version": str(torch.__version__),
        **(extra or {}),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)   # never leaves a half-written checkpoint


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[CNNLSTM, dict]:
    """Rebuild the architecture from the stored config and load weights (eval mode)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {ckpt.get('format_version')!r} in {path}")
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    cfg.pretrained = False                      # weights come from the checkpoint, not the internet
    model = CNNLSTM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])  # strict: architecture must match exactly
    model.to(device).eval()
    return model, ckpt
