"""Configuration for the fighting CNN-LSTM: model hyper-parameters, training defaults, paths.

Dataset location is NOT hard-coded: pass --data-root or set the environment variable
CS_FIGHTING_DATA to the folder that contains manifest.csv (the output of prepare_data.py).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = AI_SERVICE_ROOT / "models" / "fighting" / "cnn_lstm"
RUNS_DIR = AI_SERVICE_ROOT / "runs" / "fighting"
DATA_ENV_VAR = "CS_FIGHTING_DATA"

CLASS_NAMES = {0: "NonFight", 1: "Fight"}   # label → name (Fight is the positive class)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_SHAPE = (16, 224, 224, 3)              # stored .npy: (T, H, W, C) uint8 RGB


@dataclass
class ModelConfig:
    feature_dim: int = 512          # ResNet18 output after global average pooling
    hidden_size: int = 256          # LSTM hidden units
    num_layers: int = 1             # LSTM layers
    lstm_dropout: float = 0.3       # between LSTM layers (only used when num_layers > 1)
    head_dropout: float = 0.5       # in the classification head
    pretrained: bool = True         # ImageNet weights for ResNet18 (False only for offline tests)
    freeze_backbone: bool = True    # ResNet18 frozen: only LSTM + head are trained
    train_layer4: bool = False      # optional later fine-tuning of ResNet18's last block

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-3                # LSTM + head
    backbone_lr: float = 1e-4       # only used when train_layer4 is on
    weight_decay: float = 1e-2      # AdamW decoupled weight decay
    patience: int = 5               # early stopping on validation loss
    lr_patience: int = 2            # ReduceLROnPlateau patience (factor 0.5)
    threshold: float = 0.5          # probability threshold for Fight
    workers: int = 0                # 0 = no DataLoader worker processes (safe on Windows)
    amp: bool = False               # mixed precision, CUDA only, optional
    hflip: bool = True              # clip-level random horizontal flip (training only)
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_data_root(cli_value: str | Path | None) -> Path:
    """--data-root, else $CS_FIGHTING_DATA. Must contain manifest.csv."""
    value = cli_value or os.environ.get(DATA_ENV_VAR)
    if not value:
        raise ValueError(
            f"No dataset location given. Use --data-root <folder with manifest.csv> "
            f"or set the environment variable {DATA_ENV_VAR}."
        )
    root = Path(value)
    if not (root / "manifest.csv").is_file():
        raise FileNotFoundError(f"{root / 'manifest.csv'} not found. Run: python -m training.fighting.prepare_data")
    return root
