"""Save/load the SmokingCNN checkpoint used by evaluation and local inference.

The checkpoint contains ONLY tensors and plain Python types (dict/list/str/int/float),
so it loads with torch.load(..., weights_only=True) — the safe mode that cannot run
arbitrary code from a file.
"""

from __future__ import annotations

from pathlib import Path

import torch

from training.smoking.cnn.dataset import MEAN, STD
from training.smoking.cnn.model import CLASS_NAMES, INPUT_SIZE, SmokingCNN

FORMAT_VERSION = 1
DEFAULT_MODEL_CONFIG = {
    "in_channels": 3,
    "channels": [32, 64, 128, 256],
    "hidden_units": 256,
    "dropout": 0.5,
}


def build_metadata(best_epoch: int, best_val_metrics: dict, train_config: dict) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "model_name": "SmokingCNN",
        "origin": "trained from scratch (no pretrained weights)",
        "model_config": dict(DEFAULT_MODEL_CONFIG),
        "class_names": list(CLASS_NAMES),          # index → name
        "positive_class": "smoking",
        "positive_index": 1,
        "output": "single logit; probability(smoking) = sigmoid(logit)",
        "decision_threshold": 0.5,
        "image_size": INPUT_SIZE,
        "normalization": {"mean": list(MEAN), "std": list(STD)},
        "preprocessing": "EXIF-orient, RGB, resize to 224x224 (no aspect preservation), scale to [0,1], normalise",
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "train_config": train_config,
        "torch_version": str(torch.__version__),
    }


def save_best(path: Path, model: SmokingCNN, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: never leaves a half-written best checkpoint


def load_smoking_cnn(path: Path, device: torch.device | str = "cpu") -> tuple[SmokingCNN, dict]:
    """Load the model in eval mode plus its metadata (everything except the weights)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {ckpt.get('format_version')!r}")

    cfg = ckpt["model_config"]
    model = SmokingCNN(
        in_channels=cfg["in_channels"],
        channels=tuple(cfg["channels"]),
        hidden_units=cfg["hidden_units"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])  # strict: architecture must match exactly
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta
