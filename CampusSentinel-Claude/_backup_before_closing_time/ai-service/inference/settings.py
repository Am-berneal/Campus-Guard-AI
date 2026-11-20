"""Runtime settings for video analysis, read from environment variables (all optional).

Paths are relative to the ai-service folder unless absolute. Nothing here is a
machine-specific absolute path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

CAMERAS = {
    "library": "Outside Library",
    "b_block": "B Block",
    "cos": "COS",
    "kravings": "Kravings",
}
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else AI_SERVICE_ROOT / p


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # ---- general
    device: str = "auto"                       # AI_DEVICE: auto | cuda | cpu
    upload_dir: Path = AI_SERVICE_ROOT / "uploads"
    results_dir: Path = AI_SERVICE_ROOT / "analysis_results"
    evidence_dir: Path = AI_SERVICE_ROOT / "evidence"
    # ---- model files
    fight_checkpoint: Path = AI_SERVICE_ROOT / "models" / "fighting" / "cnn_lstm" / "best.pt"
    smoking_cnn_checkpoint: Path = AI_SERVICE_ROOT / "models" / "smoking" / "cnn" / "smoking_cnn_best.pt"
    cigarette_checkpoint: Path = AI_SERVICE_ROOT / "models" / "smoking" / "yolo" / "finetuned" / "best.pt"
    person_weights: Path | None = None         # None → search known locations (see model_manager)
    zone_config_dir: Path = AI_SERVICE_ROOT / "training" / "restricted_zone" / "configs" / "cameras"
    # ---- fighting
    fight_threshold: float | None = None       # None → threshold stored in the checkpoint
    fight_window_seconds: float = 5.0          # RWF-2000 clips are 5 s; each window → 16 frames
    fight_max_windows: int = 24                # long videos: windows spread evenly over the video
    # ---- smoking
    smoking_sample_interval_s: float = 0.5     # analyse one frame every 0.5 s
    smoking_max_frames: int = 240
    smoking_cnn_threshold: float = 0.5
    cigarette_confidence: float = 0.35
    smoking_rule: str = "both"                 # both | either | cnn | yolo
    smoking_min_frames: int = 2                # positive sampled frames needed for an event
    # ---- restricted zone
    zone_sample_fps: float = 5.0
    # ---- events
    event_gap_s: float = 2.0                   # positives closer than this are one event
    max_evidence_per_event: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        person = os.environ.get("PERSON_WEIGHTS")
        fight_thr = os.environ.get("FIGHT_THRESHOLD")
        return cls(
            device=os.environ.get("AI_DEVICE", "auto"),
            upload_dir=_path("AI_UPLOAD_DIR", "uploads"),
            results_dir=_path("AI_RESULTS_DIR", "analysis_results"),
            evidence_dir=_path("AI_EVIDENCE_DIR", "evidence"),
            fight_checkpoint=_path("FIGHT_CHECKPOINT", "models/fighting/cnn_lstm/best.pt"),
            smoking_cnn_checkpoint=_path("SMOKING_CNN_CHECKPOINT", "models/smoking/cnn/smoking_cnn_best.pt"),
            cigarette_checkpoint=_path("CIGARETTE_CHECKPOINT", "models/smoking/yolo/finetuned/best.pt"),
            person_weights=_path("PERSON_WEIGHTS", person) if person else None,
            zone_config_dir=_path("ZONE_CONFIG_DIR", "training/restricted_zone/configs/cameras"),
            fight_threshold=float(fight_thr) if fight_thr else None,
            fight_window_seconds=_float("FIGHT_WINDOW_SECONDS", 5.0),
            fight_max_windows=_int("FIGHT_MAX_WINDOWS", 24),
            smoking_sample_interval_s=_float("SMOKING_SAMPLE_INTERVAL_S", 0.5),
            smoking_max_frames=_int("SMOKING_MAX_FRAMES", 240),
            smoking_cnn_threshold=_float("SMOKING_CNN_THRESHOLD", 0.5),
            cigarette_confidence=_float("CIGARETTE_CONFIDENCE", 0.35),
            smoking_rule=os.environ.get("SMOKING_RULE", "both"),
            smoking_min_frames=_int("SMOKING_MIN_FRAMES", 2),
            zone_sample_fps=_float("ZONE_SAMPLE_FPS", 5.0),
            event_gap_s=_float("EVENT_GAP_S", 2.0),
        ).validated()

    # Per-request overrides allowed through the API (thresholds and sampling only, never paths).
    OVERRIDABLE = ("fight_threshold", "smoking_sample_interval_s", "smoking_cnn_threshold",
                   "cigarette_confidence", "smoking_rule", "smoking_min_frames", "zone_sample_fps")

    def with_overrides(self, overrides: dict | None) -> "Settings":
        if not overrides:
            return self
        unknown = set(overrides) - set(self.OVERRIDABLE)
        if unknown:
            raise ValueError(f"unknown option(s): {sorted(unknown)}; allowed: {list(self.OVERRIDABLE)}")
        return replace(self, **overrides).validated()

    def validated(self) -> "Settings":
        def between(name, lo, hi):
            v = getattr(self, name)
            if v is not None and not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} (got {v})")
        between("fight_threshold", 0.01, 0.99)
        between("smoking_cnn_threshold", 0.01, 0.99)
        between("cigarette_confidence", 0.01, 0.99)
        between("smoking_sample_interval_s", 0.04, 10)
        between("zone_sample_fps", 0.2, 30)
        between("fight_window_seconds", 1, 30)
        between("smoking_min_frames", 1, 100)
        between("fight_max_windows", 1, 200)
        between("smoking_max_frames", 1, 5000)
        if self.smoking_rule not in ("both", "either", "cnn", "yolo"):
            raise ValueError("smoking_rule must be both, either, cnn or yolo")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("AI_DEVICE must be auto, cuda or cpu")
        return self

    def public(self) -> dict:
        return {f.name: (str(getattr(self, f.name)) if isinstance(getattr(self, f.name), Path)
                         else getattr(self, f.name)) for f in fields(self)
                if not f.name.endswith(("_dir", "_checkpoint", "_weights"))}
