"""PyTorch Dataset + DataLoaders for SmokingCNN, driven by the split JSON.

Preprocessing (identical for train/val/test, so the model always sees the same format):
    open → apply EXIF rotation → RGB → resize to 224×224 → float in [0, 1] → normalise
Training only adds: random horizontal flip + random rotation (±10°).

The same `eval_transform` must be used later for inference, so it lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from training.smoking.cnn.model import INPUT_SIZE

# Fixed normalisation constants (the widely used ImageNet channel statistics).
# These are just numbers for centring pixel values, not pretrained weights.
# They are saved with each checkpoint so inference uses exactly the same values.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool) -> v2.Compose:
    steps = [
        v2.ToImage(),                                              # PIL → uint8 tensor (C, H, W)
        v2.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),       # fixed 224×224 (aspect ratio not kept)
    ]
    if train:
        steps += [
            v2.RandomHorizontalFlip(p=0.5),                        # smoking with left or right hand
            v2.RandomRotation(degrees=10),                         # small camera tilt; corners filled black
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True),                     # uint8 0..255 → float 0..1
        v2.Normalize(mean=MEAN, std=STD),
    ]
    return v2.Compose(steps)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)  # respect camera orientation tag (phone photos)
        return img.convert("RGB")           # grayscale / RGBA / palette → 3-channel RGB


class SmokingDataset(Dataset):
    """Returns (image_tensor[3,224,224], label_tensor[1] float) for one split."""

    def __init__(self, data_root: Path, records: list[dict], transform: v2.Compose) -> None:
        self.data_root = Path(data_root)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = self.data_root / rec["path"]
        try:
            img = load_rgb(path)
        except Exception as exc:  # fail loudly with the file name; never silently skip
            raise RuntimeError(f"Could not load image {path}: {exc}") from exc
        # Label shape (1,) float, matching the model's (N, 1) logits for BCEWithLogitsLoss.
        return self.transform(img), torch.tensor([float(rec["label"])])


def load_split(split_file: Path) -> dict:
    split_file = Path(split_file)
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Run: python -m training.smoking.cnn.prepare_data"
        )
    return json.loads(split_file.read_text(encoding="utf-8"))


def build_dataloaders(
    data_root: Path,
    split_file: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    """Create train/val (and optionally test) DataLoaders.

    include_test stays False during training and tuning. The test loader is only
    built for the single final evaluation.
    """
    split = load_split(split_file)
    use_cuda = torch.cuda.is_available()
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_cuda,                    # faster CPU → GPU copies
        persistent_workers=num_workers > 0,     # don't restart workers every epoch (slow on Windows)
    )

    # A seeded generator makes the shuffle order AND each worker's augmentation
    # randomness reproducible (PyTorch derives worker seeds from it).
    generator = torch.Generator().manual_seed(seed)

    loaders = {
        "train": DataLoader(
            SmokingDataset(data_root, split["train"], build_transforms(train=True)),
            shuffle=True, drop_last=False, generator=generator, **common,
        ),
        "val": DataLoader(
            SmokingDataset(data_root, split["val"], build_transforms(train=False)),
            shuffle=False, **common,
        ),
    }
    if include_test:
        loaders["test"] = DataLoader(
            SmokingDataset(data_root, split["test"], build_transforms(train=False)),
            shuffle=False, **common,
        )
    return loaders
