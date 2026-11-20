"""Loads each model ONCE and shares it across requests.

- Models are loaded lazily on first use (or eagerly via preload()) and kept in memory.
- A model whose file is missing is reported as "unavailable"; one that fails to load is
  "failed". Neither stops the other models from working.
- Nothing is ever downloaded: Ultralytics would fetch official weights automatically when a
  known file name is missing, so we only pass files that already exist.
- `inference_lock` serialises GPU work: one video is analysed at a time, so the four small
  models never compete for the RTX 4050's ~6 GB at once.

All four models together need well under 1 GB of VRAM (ResNet18-LSTM ≈ 48 MB of weights,
SmokingCNN ≈ 2 MB, two YOLO26n ≈ 5 MB each, plus activations), so they stay resident.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from inference.settings import AI_SERVICE_ROOT, Settings

LOG = logging.getLogger("inference.models")

# Places where a COCO-pretrained YOLO (class 0 = person) may already exist in this repository.
PERSON_WEIGHT_CANDIDATES = (
    "models/person/yolo26n.pt",
    "models/smoking/yolo/base/yolo26n.pt",   # base weights downloaded for cigarette fine-tuning
)


@dataclass
class ModelSlot:
    name: str
    description: str
    path: Path | None = None
    status: str = "not_loaded"      # not_loaded | loaded | unavailable | failed
    reason: str | None = None
    obj: Any = None
    load_seconds: float | None = None
    details: dict = field(default_factory=dict)

    def public(self) -> dict:
        rel = None
        if self.path is not None:
            try:
                rel = self.path.relative_to(AI_SERVICE_ROOT).as_posix()
            except ValueError:
                rel = self.path.name
        return {"name": self.name, "description": self.description, "status": self.status,
                "available": self.status == "loaded", "reason": self.reason, "path": rel,
                "load_seconds": self.load_seconds, **self.details}


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._select_device(settings.device)
        self.inference_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.slots = {
            "fighting": ModelSlot("fighting", "CNN-LSTM (ResNet18 + LSTM) fight classifier"),
            "smoking_cnn": ModelSlot("smoking_cnn", "SmokingCNN frame classifier (trained from scratch)"),
            "cigarette_yolo": ModelSlot("cigarette_yolo", "YOLO26n cigarette detector (fine-tuned)"),
            "person_yolo": ModelSlot("person_yolo", "YOLO26n person detector (COCO-pretrained) + ByteTrack"),
        }

    @staticmethod
    def _select_device(pref: str) -> str:
        if pref == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if pref == "cuda":
            LOG.warning("AI_DEVICE=cuda but CUDA is not available; using CPU")
        return "cpu"

    # ------------------------------------------------------------------ loading
    def preload(self) -> None:
        for name in self.slots:
            self.get(name)

    def get(self, name: str):
        """Return the loaded model object, or None if unavailable/failed (see status())."""
        slot = self.slots[name]
        if slot.status == "not_loaded":
            with self._load_lock:
                if slot.status == "not_loaded":
                    self._load(slot)
        return slot.obj if slot.status == "loaded" else None

    def _load(self, slot: ModelSlot) -> None:
        t0 = time.perf_counter()
        try:
            loader = getattr(self, f"_load_{slot.name}")
            loader(slot)
            if slot.status == "not_loaded":
                slot.status = "loaded"
                slot.load_seconds = round(time.perf_counter() - t0, 2)
                LOG.info("Loaded %s from %s on %s in %.2fs", slot.name, slot.path, self.device, slot.load_seconds)
        except Exception as exc:  # noqa: BLE001 — record and keep the other models working
            slot.status, slot.obj = "failed", None
            slot.reason = f"{type(exc).__name__}: {str(exc)[:300]}"
            LOG.exception("Failed to load %s", slot.name)

    def _require(self, slot: ModelSlot, path: Path, hint: str) -> bool:
        slot.path = path
        if not path.is_file():
            slot.status, slot.reason = "unavailable", f"checkpoint not found ({hint})"
            LOG.warning("%s unavailable: %s missing", slot.name, path)
            return False
        return True

    def _load_fighting(self, slot: ModelSlot) -> None:
        if not self._require(slot, self.settings.fight_checkpoint, "models/fighting/cnn_lstm/best.pt"):
            return
        from training.fighting.predict import FightPredictor   # existing predictor (no training code runs)
        pred = FightPredictor(slot.path, device=self.device)
        slot.obj = pred
        slot.details = {"threshold": pred.threshold, "clip_shape": list(pred.clip_shape),
                        "best_epoch": pred.meta.get("best_epoch", pred.meta.get("epoch"))}

    def _load_smoking_cnn(self, slot: ModelSlot) -> None:
        if not self._require(slot, self.settings.smoking_cnn_checkpoint, "models/smoking/cnn/smoking_cnn_best.pt"):
            return
        from training.smoking.cnn.checkpoint import load_smoking_cnn
        from training.smoking.cnn.dataset import build_transforms
        model, meta = load_smoking_cnn(slot.path, self.device)
        slot.obj = {"model": model, "meta": meta, "transform": build_transforms(train=False)}
        slot.details = {"classes": meta.get("class_names"), "image_size": meta.get("image_size"),
                        "best_epoch": meta.get("best_epoch")}

    def _load_cigarette_yolo(self, slot: ModelSlot) -> None:
        if not self._require(slot, self.settings.cigarette_checkpoint, "models/smoking/yolo/finetuned/best.pt"):
            return
        from training.smoking.yolo.cigarette_detector import CigaretteDetector
        det = CigaretteDetector(slot.path, conf=self.settings.cigarette_confidence, device=self.device)
        slot.obj = det
        slot.details = {"classes": det.names}

    def _find_person_weights(self) -> tuple[Path | None, list[str]]:
        tried = []
        candidates = [self.settings.person_weights] if self.settings.person_weights else \
            [AI_SERVICE_ROOT / c for c in PERSON_WEIGHT_CANDIDATES]
        for p in candidates:
            tried.append(str(p))
            if p.is_file():
                return p, tried
        return None, tried

    def _load_person_yolo(self, slot: ModelSlot) -> None:
        path, tried = self._find_person_weights()
        if path is None:
            slot.status = "unavailable"
            slot.reason = ("no COCO person-detector weights found; looked in: "
                           + ", ".join(Path(t).relative_to(AI_SERVICE_ROOT).as_posix()
                                       if Path(t).is_relative_to(AI_SERVICE_ROOT) else t for t in tried)
                           + ". Set PERSON_WEIGHTS or place yolo26n.pt in models/person/.")
            LOG.warning("person_yolo unavailable: %s", slot.reason)
            return
        slot.path = path
        from training.restricted_zone.detector import PersonDetector
        tracker = AI_SERVICE_ROOT / "training" / "restricted_zone" / "trackers" / "bytetrack_campus.yaml"
        det = PersonDetector(path, device=self.device, tracker_config=tracker)  # rejects non-COCO weights
        slot.obj = det
        slot.details = {"tracker": "ByteTrack (track_buffer 60)"}

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        gpu = None
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            gpu = {"name": props.name, "total_gb": round(props.total_memory / 1024**3, 2),
                   "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1)}
        return {"device": self.device, "gpu": gpu, "torch": str(torch.__version__),
                "models": {name: slot.public() for name, slot in self.slots.items()}}


_MANAGER: ModelManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> ModelManager:
    """Process-wide singleton."""
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = ModelManager(Settings.from_env())
    return _MANAGER
