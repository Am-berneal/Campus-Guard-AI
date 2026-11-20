"""Pretrained YOLO person detector + ByteTrack tracking (Ultralytics).

The model is PRETRAINED by Ultralytics on COCO (class 0 = person). We do not train it.
Ultralytics is imported lazily, so the rule-engine tests run without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.common.ultralytics_compat import precision_kwargs

PERSON_CLASS_ID = 0  # COCO "person"


@dataclass(frozen=True)
class PersonDetection:
    bbox: tuple[float, float, float, float]   # pixels x1, y1, x2, y2
    confidence: float
    track_id: int | None                      # None until the tracker confirms the person


def parse_boxes(xyxy, conf, cls, ids=None, min_conf: float = 0.0) -> list[PersonDetection]:
    """Convert raw arrays (as produced by Ultralytics) into PersonDetection objects.

    Keeps only class 0 (person) with confidence >= min_conf. `ids` may be None
    (plain detection, or tracker has not assigned IDs yet).
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float64).ravel()
    cls = np.asarray(cls).ravel().astype(int)
    if not (len(xyxy) == len(conf) == len(cls)):
        raise ValueError(f"length mismatch: boxes {len(xyxy)}, conf {len(conf)}, cls {len(cls)}")
    id_list = [None] * len(conf) if ids is None else [int(i) for i in np.asarray(ids).ravel()]
    if len(id_list) != len(conf):
        raise ValueError("ids length does not match boxes")

    out = []
    for box, c, k, tid in zip(xyxy, conf, cls, id_list):
        if k == PERSON_CLASS_ID and c >= min_conf:
            out.append(PersonDetection(tuple(float(v) for v in box), float(c), tid))
    return out


class PersonDetector:
    """Wraps one Ultralytics model. Create once, reuse for every frame, then close()."""

    def __init__(self, weights: Path, confidence=0.4, iou=0.5, image_size=640,
                 device="auto", half=True, tracker_config: Path | None = None) -> None:
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.half = bool(half and device == "cuda")  # float16 halves activation memory on GPU
        self.confidence, self.iou, self.image_size = confidence, iou, image_size
        self.tracker_config = str(tracker_config) if tracker_config else "bytetrack.yaml"

        weights = Path(weights)
        weights.parent.mkdir(parents=True, exist_ok=True)
        # If the file is missing and the name is an official Ultralytics asset
        # (e.g. yolo26n.pt), Ultralytics downloads it to exactly this path.
        self.model = YOLO(str(weights))
        if self.model.names.get(PERSON_CLASS_ID) != "person":
            raise ValueError(f"{weights} is not a COCO detector: class 0 is {self.model.names.get(0)!r}")
        self.weights = weights

    def _kwargs(self) -> dict:
        # float16 on GPU: `quantize=16` on Ultralytics 8.4.x, `half=True` on older versions.
        return dict(classes=[PERSON_CLASS_ID], conf=self.confidence, iou=self.iou, imgsz=self.image_size,
                    device=self.device, verbose=False, **precision_kwargs(self.half))

    @staticmethod
    def _to_detections(result, min_conf: float) -> list[PersonDetection]:
        b = result.boxes
        if b is None or len(b) == 0:
            return []
        ids = b.id.cpu().numpy() if b.id is not None else None
        return parse_boxes(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), ids, min_conf)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Single-frame detection without tracking (track_id is None)."""
        result = self.model.predict(frame, **self._kwargs())[0]
        return self._to_detections(result, self.confidence)

    def track(self, frame: np.ndarray, persist: bool = True) -> list[PersonDetection]:
        """Detection + tracking. Call with consecutive frames of ONE video; persist=True
        keeps the tracker state between calls so the same person keeps the same ID.
        Pass persist=False on the FIRST frame of a new video to start with fresh track IDs."""
        result = self.model.track(frame, persist=persist, tracker=self.tracker_config, **self._kwargs())[0]
        return self._to_detections(result, self.confidence)

    def close(self) -> None:
        """Free the model and cached GPU memory."""
        self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
