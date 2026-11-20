"""Verify the DataLoaders before training: shapes, dtypes, labels, value ranges.

Does NOT train. Runs one batch through the untrained SmokingCNN only to confirm
the data and model fit together. Saves a preview image of one training batch so
you can visually confirm that labels match the pictures.

Run from the ai-service folder:
    python -m training.smoking.cnn.check_batches
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from torchvision.utils import save_image

from training.common.reproducibility import get_device, set_seed
from training.smoking.cnn.dataset import MEAN, STD, build_dataloaders, load_split
from training.smoking.cnn.model import CLASS_NAMES, SmokingCNN
from training.smoking.cnn.prepare_data import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR

AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]


def denormalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_DIR / "split_seed42.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    set_seed(args.seed)
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok &= bool(cond)

    split = load_split(args.split_file)
    loaders = build_dataloaders(args.data_root, args.split_file, args.batch_size, args.num_workers, args.seed)

    print("Split sizes (from split file)")
    for name in ("train", "val", "test"):
        c = Counter(r["label"] for r in split[name])
        print(f"  {name:<6} {len(split[name]):>5} images  ({CLASS_NAMES[0]}={c[0]}, {CLASS_NAMES[1]}={c[1]})")

    print("\nLeakage checks")
    sets = {k: {r["path"] for r in split[k]} for k in ("train", "val", "test")}
    check(not (sets["train"] & sets["val"]), "train ∩ val = ∅")
    check(not (sets["train"] & sets["test"]), "train ∩ test = ∅")
    check(not (sets["val"] & sets["test"]), "val ∩ test = ∅")
    check(all(p.startswith("testing/") for p in sets["test"]), "all test paths are under testing/")
    check(not any(p.startswith("testing/") for p in sets["train"] | sets["val"]), "no testing/ path in train or val")
    check(len(loaders["train"].dataset) == len(split["train"]), "train loader size matches split")

    print("\nTraining batch")
    images, labels = next(iter(loaders["train"]))
    print(f"  images {tuple(images.shape)} {images.dtype}   labels {tuple(labels.shape)} {labels.dtype}")
    print(f"  pixel range after normalisation: min {images.min():.2f}, max {images.max():.2f}, mean {images.mean():.2f}")
    print(f"  labels in batch: {dict(Counter(int(v) for v in labels.flatten()))}")
    check(images.shape[1:] == (3, 224, 224), "image tensor is (N, 3, 224, 224)")
    check(images.dtype == torch.float32, "images are float32")
    check(labels.shape == (images.shape[0], 1) and labels.dtype == torch.float32, "labels are (N, 1) float32")
    check(set(labels.flatten().tolist()) <= {0.0, 1.0}, "labels are only 0 or 1")
    check(torch.isfinite(images).all().item(), "no NaN/Inf in images")

    print("\nValidation batch (no augmentation → identical on every pass)")
    v1, _ = next(iter(loaders["val"]))
    v2_, _ = next(iter(loaders["val"]))
    check(torch.equal(v1, v2_), "two passes over val give identical tensors")

    print("\nModel compatibility (untrained model, one forward pass, no training)")
    device = get_device()
    model = SmokingCNN().to(device).eval()
    with torch.no_grad():
        logits = model(images.to(device))
    check(logits.shape == labels.shape, f"logits {tuple(logits.shape)} match labels {tuple(labels.shape)}")

    out_dir = AI_SERVICE_ROOT / "runs" / "smoking_cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = out_dir / "preview_train_batch.png"
    save_image(denormalise(images[:16]), preview, nrow=8)
    print(f"\nPreview of first 16 training images saved: {preview}")
    print("  Labels (row by row): " + " ".join("S" if v else "N" for v in labels[:16].flatten().int().tolist()))
    print("  S = smoking, N = not_smoking. Open the image and confirm the labels look right.")

    print(f"\nResult: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":  # required on Windows when num_workers > 0
    sys.exit(main())
