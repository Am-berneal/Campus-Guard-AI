"""Predict Fight / NonFight for one processed clip (.npy) or a raw video file.

    from training.fighting.predict import FightPredictor
    predictor = FightPredictor("models/fighting/cnn_lstm/best.pt")
    predictor.predict_npy("clip.npy")
    # {"predicted_class": "Fight", "fight_probability": 0.87, "nonfight_probability": 0.13, "threshold": 0.5}

A prediction is a POTENTIAL incident for administrator review, never an automatic decision.

Run from the ai-service folder:
    python -m training.fighting.predict --checkpoint models\\fighting\\cnn_lstm\\best.pt --clip "path\\to\\clip.npy"
    python -m training.fighting.predict --checkpoint models\\fighting\\cnn_lstm\\best.pt --video "path\\to\\video.avi"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from training.fighting.checkpoint import load_checkpoint
from training.fighting.config import CHECKPOINT_DIR, CLIP_SHAPE


class FightPredictor:
    def __init__(self, checkpoint: Path | str = CHECKPOINT_DIR / "best.pt", device: str = "auto",
                 threshold: float | None = None) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.meta = load_checkpoint(Path(checkpoint), self.device)
        self.threshold = float(threshold if threshold is not None else self.meta["train_config"].get("threshold", 0.5))
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        norm = self.meta["normalization"]
        self.mean = torch.tensor(norm["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(norm["std"], device=self.device).view(1, 3, 1, 1)
        self.clip_shape = tuple(self.meta.get("input_clip_shape", CLIP_SHAPE))

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        """uint8 (T, H, W, C) RGB → normalised float (1, T, C, H, W), same as FightingVideoDataset."""
        if not isinstance(clip, np.ndarray):
            raise TypeError("clip must be a numpy array")
        if clip.dtype != np.uint8 or clip.shape != self.clip_shape:
            raise ValueError(f"expected uint8 {self.clip_shape}, got {clip.dtype} {clip.shape}")
        x = torch.from_numpy(np.ascontiguousarray(clip)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        return x.unsqueeze(0)

    @torch.no_grad()
    def predict_array(self, clip: np.ndarray) -> dict:
        logit = self.model(self._to_tensor(clip)).float().item()
        if not np.isfinite(logit):
            raise RuntimeError("model produced a non-finite output")
        p_fight = float(torch.sigmoid(torch.tensor(logit)))
        return {
            "predicted_class": "Fight" if p_fight >= self.threshold else "NonFight",
            "fight_probability": round(p_fight, 4),
            "nonfight_probability": round(1.0 - p_fight, 4),
            "threshold": self.threshold,
        }

    def predict_npy(self, path: Path | str) -> dict:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"clip not found: {path}")
        return self.predict_array(np.load(path))

    def predict_video(self, path: Path | str) -> dict:
        """Raw video → the SAME preprocessing as prepare_data.py (16 frames, letterbox 224, RGB uint8)."""
        from training.fighting.prepare_data import process_video

        t, size = self.clip_shape[0], self.clip_shape[1]
        clip, info = process_video(Path(path), t, size)
        return {**self.predict_array(clip), "frame_indices": info.indices, "source_frames": info.frame_count}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fight / NonFight prediction for one clip.")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", type=Path, help="processed .npy clip (16x224x224x3 uint8)")
    src.add_argument("--video", type=Path, help="raw video file; preprocessed like prepare_data.py")
    ap.add_argument("--threshold", type=float, default=None, help="default: value stored in the checkpoint")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args(argv)
    try:
        predictor = FightPredictor(args.checkpoint, args.device, args.threshold)
        result = predictor.predict_npy(args.clip) if args.clip else predictor.predict_video(args.video)
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # e.g. VideoError from preprocessing
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    print("Potential incident only: requires administrator review.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
