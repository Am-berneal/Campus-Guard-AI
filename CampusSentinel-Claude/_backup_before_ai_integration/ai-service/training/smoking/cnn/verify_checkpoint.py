"""Load the saved best checkpoint the way inference will, and re-check it on VALIDATION data.

Confirms that: the file loads safely (weights_only=True), the architecture matches,
metadata is complete, and the reloaded model reproduces the stored validation loss.
The test set is NOT used.

Run from the ai-service folder:
    python -m training.smoking.cnn.verify_checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch import nn

from training.common.reproducibility import get_device, set_seed
from training.smoking.cnn.checkpoint import load_smoking_cnn
from training.smoking.cnn.dataset import build_dataloaders
from training.smoking.cnn.prepare_data import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR
from training.smoking.cnn.train import BEST_NAME, DEFAULT_MODEL_DIR, evaluate

REQUIRED_KEYS = ["model_config", "class_names", "image_size", "normalization", "best_epoch", "best_val_metrics"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / BEST_NAME)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_DIR / "split_seed42.json")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    set_seed(42)
    device = get_device()
    model, meta = load_smoking_cnn(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (weights_only=True) on {device}")

    missing = [k for k in REQUIRED_KEYS if k not in meta]
    print(f"  metadata keys present: {'yes' if not missing else 'MISSING ' + str(missing)}")
    print(f"  classes {meta['class_names']} | image {meta['image_size']} | best epoch {meta['best_epoch']}")

    val_loader = build_dataloaders(args.data_root, args.split_file, 32, args.num_workers, 42)["val"]
    val_loss, vm = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), device, use_amp=device.type == "cuda")
    stored = meta["best_val_metrics"]
    print(f"  val loss: stored {stored['val_loss']:.4f} | recomputed {val_loss:.4f}")
    print(f"  val acc : stored {stored['accuracy']:.4f} | recomputed {vm['accuracy']:.4f}")

    ok = not missing and abs(val_loss - stored["val_loss"]) < 5e-3 and abs(vm["accuracy"] - stored["accuracy"]) < 5e-3
    print(f"\nResult: {'CHECKPOINT OK' if ok else 'MISMATCH — report this output'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
