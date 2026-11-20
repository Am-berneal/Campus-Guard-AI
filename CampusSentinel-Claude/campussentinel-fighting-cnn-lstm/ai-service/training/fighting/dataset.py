"""PyTorch Dataset for preprocessed RWF-2000 clips.

    from training.fighting.dataset import FightingVideoDataset
    ds = FightingVideoDataset(r"D:\\CampusSentinel-Datasets\\fighting_processed\\manifest.csv", split="train")
    video, label = ds[0]          # video: float32 (16, 3, 224, 224), ImageNet-normalised; label: torch.long

Files on disk stay uint8 (T, H, W, C). Conversion to float and normalisation happen
here, at load time, so preprocessing and model normalisation remain separate.
Use DataLoader(num_workers=0) on Windows unless workers have been tested.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LABELS = {"NonFight": 0, "Fight": 1}


class FightingVideoDataset(Dataset):
    def __init__(self, manifest: Path | str, split: str, root: Path | str | None = None,
                 normalize: bool = True, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        """
        manifest   path to manifest.csv written by prepare_data.py
        split      "train" or "val"
        root       folder that npy paths are relative to (default: the manifest's folder)
        transform  optional function on the (T, C, H, W) float tensor in [0, 1], applied BEFORE normalisation
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        manifest = Path(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}. Run: python -m training.fighting.prepare_data")
        self.root = Path(root) if root else manifest.parent
        self.split, self.normalize, self.transform = split, normalize, transform

        with manifest.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split and r["status"] == "ok"]
        if not rows:
            raise ValueError(f"no processed ('ok') {split} clips listed in {manifest}")
        for r in rows:
            if LABELS.get(r["class_name"]) != int(r["label"]):
                raise ValueError(f"label mismatch in manifest for {r['npy_path']}: {r['class_name']} → {r['label']}")
        self.items = [(self.root / r["npy_path"], int(r["label"])) for r in rows]
        self.expected_shape = tuple(int(v) for v in rows[0]["output_shape"].split("x"))  # (T, H, W, C)
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def class_counts(self) -> dict[str, int]:
        inv = {v: k for k, v in LABELS.items()}
        c = Counter(label for _, label in self.items)
        return {inv[k]: c[k] for k in sorted(inv)}

    def load_uint8(self, idx: int) -> np.ndarray:
        """Raw clip (T, H, W, C) uint8, with validation."""
        path, _ = self.items[idx]
        if not path.is_file():
            raise FileNotFoundError(f"processed clip missing: {path} (re-run prepare_data)")
        try:
            arr = np.load(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        if arr.dtype != np.uint8 or arr.shape != self.expected_shape:
            raise ValueError(f"{path}: expected uint8 {self.expected_shape}, got {arr.dtype} {arr.shape}")
        return arr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = self.load_uint8(idx)
        video = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)   # (T, C, H, W) in [0, 1]
        if self.transform is not None:
            video = self.transform(video)
        if self.normalize:
            video = (video - self._mean) / self._std
        return video, torch.tensor(self.items[idx][1], dtype=torch.long)
