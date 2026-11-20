"""AI analysis endpoints (added alongside the existing upload router; /videos/upload is untouched).

    POST /videos/{upload_id}/analyze   run all available detectors on a saved upload
    GET  /models/status                which models are loaded / unavailable / failed
    GET  /evidence/{upload_id}/{name}  evidence image written during analysis

Inference is CPU/GPU-bound, so it runs in a worker thread (run_in_threadpool) and never
blocks the event loop. One analysis per upload at a time (409 for duplicates); the GPU work
itself is serialised by the model manager's lock. Finished results are cached on disk and
returned again unless "force": true.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from inference.model_manager import get_manager
from inference.pipeline import analyze_video
from inference.settings import CAMERAS, VIDEO_EXTENSIONS
from inference.video_io import VideoReadError

LOG = logging.getLogger("api.analysis")
router = APIRouter(tags=["analysis"])

UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
EVIDENCE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}\.jpg$")
_in_progress: set[str] = set()
_in_progress_lock = threading.Lock()


class AnalyzeRequest(BaseModel):
    camera_id: str = Field(..., description="library | b_block | cos | kravings")
    recorded_at: datetime | None = Field(None, description="Campus-local recording start; default = now")
    force: bool = Field(False, description="Re-run even if a cached result exists")
    options: dict | None = Field(None, description="Threshold/sampling overrides, see Settings.OVERRIDABLE")


def _resolve_upload(upload_id: str, upload_dir: Path) -> Path:
    """Find <upload_id>.<ext> inside the upload folder without allowing path traversal."""
    if not UPLOAD_ID_RE.fullmatch(upload_id):
        raise HTTPException(400, "Invalid upload_id")
    base = upload_dir.resolve()
    for ext in VIDEO_EXTENSIONS:
        candidate = (base / f"{upload_id}{ext}").resolve()
        if candidate.parent != base:            # defence in depth; the regex already blocks / \ ..
            raise HTTPException(400, "Invalid upload_id")
        if candidate.is_file():
            return candidate
    raise HTTPException(404, f"No uploaded video with id {upload_id}")


def _result_path(results_dir: Path, upload_id: str) -> Path:
    return results_dir / f"{upload_id}.json"


@router.post("/videos/{upload_id}/analyze")
async def analyze(upload_id: str, body: AnalyzeRequest):
    manager = get_manager()
    if body.camera_id not in CAMERAS:
        raise HTTPException(422, f"camera_id must be one of {list(CAMERAS)}")
    video = _resolve_upload(upload_id, manager.settings.upload_dir)
    try:
        settings = manager.settings.with_overrides(body.options)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc

    cached = _result_path(manager.settings.results_dir, upload_id)
    if cached.is_file() and not body.force and not body.options:
        result = json.loads(cached.read_text(encoding="utf-8"))
        if result.get("camera_id") == body.camera_id:
            return {**result, "cached": True}

    with _in_progress_lock:
        if upload_id in _in_progress:
            raise HTTPException(409, "Analysis of this upload is already running")
        _in_progress.add(upload_id)
    try:
        LOG.info("Analysing upload %s (camera %s, %s)", upload_id, body.camera_id, video.name)
        result = await run_in_threadpool(analyze_video, manager, video, upload_id, body.camera_id,
                                         CAMERAS[body.camera_id], settings, body.recorded_at)
    except VideoReadError as exc:
        raise HTTPException(422, f"Video cannot be read: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — logged with traceback, generic message to client
        LOG.exception("Analysis of %s failed", upload_id)
        raise HTTPException(500, "Analysis failed unexpectedly; see AI service log") from exc
    finally:
        with _in_progress_lock:
            _in_progress.discard(upload_id)

    if result["status"] == "failed" and all(not m.get("available") for m in result["analysis"].values()):
        raise HTTPException(503, "No AI model is available; check GET /models/status")

    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    os.replace(tmp, cached)
    LOG.info("Upload %s analysed: %s, %d potential incident(s), %.1fs", upload_id, result["status"],
             result["potential_incident_count"], result["processing"]["duration_seconds"])
    return {**result, "cached": False}


@router.get("/models/status")
async def models_status(load: bool = True):
    """load=true (default) loads any model not loaded yet, so the answer is definitive."""
    manager = get_manager()
    if load:
        await run_in_threadpool(manager.preload)
    return manager.status()


@router.get("/evidence/{upload_id}/{name}")
async def evidence(upload_id: str, name: str):
    if not UPLOAD_ID_RE.fullmatch(upload_id) or not EVIDENCE_NAME_RE.fullmatch(name) or ".." in name:
        raise HTTPException(400, "Invalid evidence path")
    base = get_manager().settings.evidence_dir.resolve()
    path = (base / upload_id / name).resolve()
    if path.parent != base / upload_id or not path.is_file():
        raise HTTPException(404, "Evidence not found")
    return FileResponse(path, media_type="image/jpeg")


def preload_in_background() -> None:
    """Optional: start loading models when the service starts (AI_PRELOAD_MODELS=1, default)."""
    if os.environ.get("AI_PRELOAD_MODELS", "1") != "0":
        threading.Thread(target=get_manager().preload, name="model-preload", daemon=True).start()
