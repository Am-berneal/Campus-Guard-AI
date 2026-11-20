"""Verify a CNN-LSTM checkpoint: load safely, rebuild the architecture, load weights,
run a forward pass, and check the output shape and that values are finite.

Uses a random input by default (checks wiring, not accuracy). With --clip, a real
processed clip is used instead.

Run from the ai-service folder:
    python -m training.fighting.verify_checkpoint --checkpoint models\\fighting\\cnn_lstm\\best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from training.fighting.checkpoint import load_checkpoint
from training.fighting.config import CHECKPOINT_DIR
from training.fighting.model import count_parameters

REQUIRED = ["model_state_dict", "epoch", "best_val_loss", "model_config", "class_mapping", "normalization",
            "train_config", "input_clip_shape"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    ap.add_argument("--clip", type=Path, default=None, help="optional processed .npy clip")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args(argv)
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    try:
        model, meta = load_checkpoint(args.checkpoint, device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    missing = [k for k in REQUIRED if k not in meta]
    total, _ = count_parameters(model)
    print(f"Loaded {args.checkpoint} (weights_only=True) on {device}")
    print(f"  epoch {meta['epoch']} | best val loss {meta['best_val_loss']:.4f} | params {total:,}")
    print(f"  model config {meta['model_config']}")
    print(f"  classes {meta['class_mapping']} | normalization {meta['normalization']}")
    print(f"  required keys: {'all present' if not missing else 'MISSING ' + str(missing)}")

    t, h, w, c = meta["input_clip_shape"]
    if args.clip:
        from training.fighting.predict import FightPredictor
        pred = FightPredictor(args.checkpoint, device.type)
        x = pred._to_tensor(np.load(args.clip))
        src = f"clip {args.clip}"
    else:
        x = torch.randn(2, t, c, h, w, device=device)
        src = "random input"
    with torch.no_grad():
        out = model(x)
    ok_shape = tuple(out.shape) == (x.shape[0], 1)
    ok_finite = bool(torch.isfinite(out).all())
    print(f"  forward pass ({src}): input {tuple(x.shape)} → output {tuple(out.shape)} "
          f"| shape {'OK' if ok_shape else 'WRONG'} | finite {'OK' if ok_finite else 'NO'}")
    ok = not missing and ok_shape and ok_finite
    print(f"Result: {'CHECKPOINT OK' if ok else 'CHECKPOINT PROBLEM'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
