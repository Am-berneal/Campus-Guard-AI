"""Reusable cigarette detector (for the CLI now, and the FastAPI service later).

    from training.smoking.yolo.cigarette_detector import CigaretteDetector
    det = CigaretteDetector()                        # loads models/smoking/yolo/finetuned/best.pt
    detections = det.detect(frame_bgr_or_path)       # list of dicts (see Detection below)

A detection means "a cigarette-like OBJECT is visible here". It is not evidence that
a person is smoking. The default confidence threshold (0.25, the Ultralytics default)
has NOT been tuned; tune it on the validation split only, never on test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.common.ultralytics_compat import precision_kwargs

AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = AI_SERVICE_ROOT / "models" / "smoking" / "yolo" / "finetuned" / "best.pt"


class CigaretteDetector:
    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, conf: float = 0.25, iou: float = 0.5,
                 imgsz: int = 640, device: str = "auto", half: bool = True) -> None:
        import torch
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"fine-tuned weights not found: {weights} (train first)")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = bool(half and device != "cpu")   # float16 on GPU only
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.model = YOLO(str(weights))
        if self.model.task != "detect" or self.model.names.get(0) != "cigarette" or len(self.model.names) != 1:
            raise ValueError(f"{weights} is not the cigarette detector (task={self.model.task}, "
                             f"{len(self.model.names)} classes, class 0 = {self.model.names.get(0)!r})")
        self.names = dict(self.model.names)
        self.weights = weights

    def _run(self, source):
        return self.model.predict(source, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                                  verbose=False, **precision_kwargs(self.half))[0]

    @staticmethod
    def _to_dicts(result) -> list[dict]:
        h, w = result.orig_shape
        out = []
        b = result.boxes
        if b is None:
            return out
        for xyxy, conf, cls in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            out.append({
                "class_id": int(cls),
                "class_name": result.names[int(cls)],
                "confidence": round(float(conf), 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_xyxy_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            })
        return out

    def detect(self, image: np.ndarray | str | Path) -> list[dict]:
        """Detect cigarettes in ONE image (BGR numpy array as from OpenCV, or a file path)."""
        return self._to_dicts(self._run(str(image) if isinstance(image, Path) else image))

    def detect_and_draw(self, image: np.ndarray | str | Path) -> tuple[list[dict], np.ndarray]:
        """Detections plus an annotated BGR image (boxes, class name, confidence)."""
        result = self._run(str(image) if isinstance(image, Path) else image)
        return self._to_dicts(result), result.plot()

    def close(self) -> None:
        self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
