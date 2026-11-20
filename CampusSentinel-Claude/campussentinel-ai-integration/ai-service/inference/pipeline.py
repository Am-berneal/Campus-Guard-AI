"""Central video analysis: runs every available detector on one uploaded video.

Each module is isolated: if a model is unavailable it is reported as such; if a module
raises, it is reported as "failed" (and logged) and the other modules still run.
Nothing is ever invented: every number comes from a model run on this video.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from inference.model_manager import ModelManager
from inference.settings import Settings
from inference.video_io import VideoReadError, probe

LOG = logging.getLogger("inference.pipeline")

DISCLAIMER = ("All results are POTENTIAL incidents produced by AI models from a simulated camera upload. "
              "They may be wrong and must be reviewed by an administrator before any action.")


def _unavailable(reason: str) -> dict:
    return {"available": False, "status": "unavailable", "reason": reason, "potential_incident": False, "events": []}


def _failed(module: str, exc: Exception) -> dict:
    LOG.exception("%s analysis failed", module)
    return {"available": True, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "potential_incident": False, "events": []}


def analyze_video(manager: ModelManager, video_path: Path, upload_id: str, camera_id: str, camera_name: str,
                  settings: Settings | None = None, recorded_at: datetime | None = None,
                  modules: tuple[str, ...] = ("fighting", "smoking", "restricted_zone")) -> dict:
    """Synchronous (CPU/GPU-bound). FastAPI calls it in a worker thread."""
    settings = settings or manager.settings
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    info = probe(video_path)                      # raises VideoReadError for unreadable files
    if info.frame_count <= 0:
        raise VideoReadError("video reports zero frames")
    evidence_dir = settings.evidence_dir / upload_id
    analysis: dict[str, dict] = {}
    timings: dict[str, float] = {}

    with manager.inference_lock:                  # one video at a time on the GPU
        if "fighting" in modules:
            t = time.perf_counter()
            predictor = manager.get("fighting")
            if predictor is None:
                analysis["fighting"] = _unavailable(manager.slots["fighting"].reason or "model not loaded")
            else:
                try:
                    from inference.fighting.analyzer import analyze as fight
                    analysis["fighting"] = {"available": True, "status": "completed", "model": "cnn_lstm",
                                            **fight(video_path, info, predictor, settings, evidence_dir)}
                except Exception as exc:  # noqa: BLE001
                    analysis["fighting"] = _failed("fighting", exc)
            timings["fighting"] = round(time.perf_counter() - t, 2)

        if "smoking" in modules:
            t = time.perf_counter()
            cnn, yolo = manager.get("smoking_cnn"), manager.get("cigarette_yolo")
            if cnn is None and yolo is None:
                analysis["smoking"] = _unavailable("neither smoking CNN nor cigarette YOLO could be loaded")
            else:
                try:
                    from inference.smoking.analyzer import analyze as smoke
                    analysis["smoking"] = {"available": True, "status": "completed",
                                           "models": {"smoking_cnn": cnn is not None, "cigarette_yolo": yolo is not None},
                                           **smoke(video_path, info, cnn, yolo, manager.device, settings, evidence_dir)}
                except Exception as exc:  # noqa: BLE001
                    analysis["smoking"] = _failed("smoking", exc)
            timings["smoking"] = round(time.perf_counter() - t, 2)

        if "restricted_zone" in modules:
            t = time.perf_counter()
            person = manager.get("person_yolo")
            if person is None:
                analysis["restricted_zone"] = _unavailable(manager.slots["person_yolo"].reason or "person detector not loaded")
            else:
                try:
                    from inference.restricted_zone.analyzer import analyze as zones
                    analysis["restricted_zone"] = zones(video_path, info, person, camera_id, settings, evidence_dir,
                                                        recorded_at)
                except Exception as exc:  # noqa: BLE001
                    analysis["restricted_zone"] = _failed("restricted_zone", exc)
            timings["restricted_zone"] = round(time.perf_counter() - t, 2)

        if manager.device == "cuda":
            torch.cuda.empty_cache()

    statuses = [m["status"] for m in analysis.values()]
    overall = "completed" if all(s == "completed" for s in statuses) else \
        ("failed" if all(s in ("failed", "unavailable") for s in statuses) else "partial")
    return {
        "upload_id": upload_id,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "source": "simulated camera (uploaded recording)",
        "status": overall,
        "video": info.to_dict(),
        "analysis": analysis,
        "potential_incident_count": sum(len(m.get("events", [])) for m in analysis.values()),
        "processing": {"device": manager.device, "duration_seconds": round(time.perf_counter() - t0, 2),
                       "module_seconds": timings, "started_at": started.isoformat(timespec="seconds"),
                       "settings": settings.public()},
        "disclaimer": DISCLAIMER,
    }
