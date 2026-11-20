# CampusSentinel — AI integration

Upload (simulated camera) → FastAPI stores the video → **AI analysis by every available model** → Node.js stores **potential** incidents in MongoDB → React dashboard for administrator review.

There are no physical CCTV cameras and no live streaming. Every result is a potential incident that an administrator must review. The administrator contacts security by walkie-talkie; guards never use the system.

## Architecture

```
React (5173) ──/api──► Express (5000) ──HTTP──► FastAPI (8000) ──► inference/ (PyTorch, Ultralytics)
                         │  POST /api/videos/upload ─────► POST /videos/upload            (unchanged)
                         │  POST /api/videos/:id/analyze ► POST /videos/{id}/analyze ─► pipeline.analyze_video
                         │  GET  /api/ai/models/status ──► GET  /models/status
                         │  GET  /api/incidents/:id/evidence/:n ► GET /evidence/{id}/{file}
                         └─► MongoDB: incidents (potential, pending_review → reviewed / dismissed)
```

```
ai-service/
├── api/routers/analysis.py        NEW: /videos/{id}/analyze, /models/status, /evidence/...
├── inference/                     NEW: inference layer (no training code runs here)
│   ├── settings.py                env-based settings, camera list
│   ├── model_manager.py           loads each model ONCE, status, device, GPU lock
│   ├── pipeline.py                runs all modules, isolates failures, builds the JSON
│   ├── video_io.py                probe / sequential frame reading / evidence images
│   ├── fighting/analyzer.py       5 s windows × 16 frames → existing FightPredictor
│   ├── smoking/analyzer.py        SmokingCNN + cigarette YOLO on sampled frames → events
│   ├── restricted_zone/analyzer.py existing PersonDetector + ByteTrack + RuleEngine
│   └── tests/test_inference.py
├── training/common/ultralytics_compat.py   NEW: quantize/half argument compatibility
├── training/restricted_zone/configs/cameras/<camera_id>.yaml   NEW: fictional zones per camera
├── evidence/<upload_id>/*.jpg     created at runtime
└── analysis_results/<upload_id>.json   created at runtime (cache)
```

## Models

| Module | Model | File | Origin |
|---|---|---|---|
| Fighting | CNN-LSTM (ResNet18 + LSTM) | `models/fighting/cnn_lstm/best.pt` | Trained by us |
| Smoking (frame) | SmokingCNN | `models/smoking/cnn/smoking_cnn_best.pt` | Trained from scratch by us |
| Smoking (object) | YOLO26n cigarette | `models/smoking/yolo/finetuned/best.pt` | Fine-tuned by us |
| Restricted zone | YOLO26n person (COCO) + ByteTrack | first found of: `PERSON_WEIGHTS`, `models/person/yolo26n.pt`, `models/smoking/yolo/base/yolo26n.pt` | **Pretrained by Ultralytics**, not trained by us |

**Person detector.** `models/person/` only holds a README. The repository *does* contain a compatible COCO detector: `models/smoking/yolo/base/yolo26n.pt`, the pretrained base downloaded before cigarette fine-tuning.
- It is used automatically. On load, `PersonDetector` verifies that class 0 is `person`.
- If no candidate file exists, the module is reported **unavailable**. Nothing is downloaded: the manager only passes existing files to Ultralytics, which would otherwise auto-download known names.
- To use a dedicated copy, run `copy models\smoking\yolo\base\yolo26n.pt models\person\yolo26n.pt`, or set `PERSON_WEIGHTS`.

**Loading:**
- Each model is loaded once, then reused for every request.
- Models load lazily on first use, or in the background at startup (`AI_PRELOAD_MODELS=1`, the default).
- CUDA is used automatically if available, with CPU as fallback.
- A missing file → `unavailable`. A load error → `failed`, logged with a traceback. Neither affects the other models.
- All four models need well under 1 GB of VRAM. A lock analyses **one video at a time**, so models never compete for the RTX 4050's memory.

**Ultralytics compatibility fix.** 8.4.x replaced `half=True` with `quantize=16`, and older versions only accept `half`. `training/common/ultralytics_compat.py` picks the argument the installed version supports. It is used by `cigarette_detector.py` and `restricted_zone/detector.py`.

## How each module analyses a video

| Module | Sampling | Logic | Output |
|---|---|---|---|
| Fighting | The video is split into 5 s windows (max 24, spread evenly). Each window gets **the existing 16-frame uniform sampling + 224 letterbox** (`prepare_data.uniform_indices` / `preprocess_frame`) | `FightPredictor.predict_array` per window. Window ≥ threshold → positive; neighbouring positives merge into one event | `predicted_class`, `fight_probability` (max window), `nonfight_probability`, `threshold`, per-window probabilities, events |
| Smoking | One frame every 0.5 s (max 240) | CNN (exact eval preprocessing from `smoking/cnn/dataset.py`) batched on sampled frames; YOLO on sampled frames. Default rule `both`: CNN ≥ 0.5 **and** cigarette ≥ 0.35 in the same frame, on ≥ 2 frames within 2 s | Events, CNN and cigarette statistics **separately**, cigarette detections with timestamps |
| Restricted zone | 5 frames per second | Existing `PersonDetector.track` (tracker reset per video) → `RuleEngine` (bottom-centre point in polygon, campus-time schedule, 1 s dwell, 300 s cooldown) | One event per person per zone, with zone, campus timestamp, box, confidence |

**Recording time.** An uploaded file has no trustworthy clock. Restricted-time rules use `recorded_at` from the request; a time without an offset is campus time. Otherwise the analysis time is used. The source is reported in the result.

**The smoking fusion rule was never validated on labelled data.** It is a transparent heuristic, configurable through `SMOKING_RULE` and the thresholds. A detected cigarette alone never creates an event under the default rule.

## API

### FastAPI (8000)

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Unchanged |
| POST | `/videos/upload` | Unchanged |
| POST | `/videos/{upload_id}/analyze` | Body `{"camera_id": "cos", "recorded_at": "2026-11-16T22:00", "force": false, "options": {...}}` |
| GET | `/models/status` | Loads models if needed, then reports each one: `loaded` / `unavailable` / `failed` |
| GET | `/evidence/{upload_id}/{file}.jpg` | Evidence image |

**`/videos/{upload_id}/analyze` status codes:**

| Code | Meaning |
|---|---|
| 200 | Done. A cached result is returned with `"cached": true` unless `force` |
| 400 | Bad ID |
| 404 | No such upload |
| 409 | This upload is already being analysed |
| 422 | Bad camera, bad options, or unreadable video |
| 503 | No model available at all |
| 500 | Unexpected error (logged) |

**How the endpoint works:**
- The upload ID must match `^[A-Za-z0-9_-]{1,64}$`. The video is resolved only as `uploads/<id>.<mp4|avi|mov|mkv>` inside the upload folder, so path traversal is impossible.
- Inference runs in a worker thread, so `/health` stays responsive during an analysis.

`options` (per request) may override only these: `fight_threshold`, `smoking_sample_interval_s`, `smoking_cnn_threshold`, `cigarette_confidence`, `smoking_rule`, `smoking_min_frames`, `zone_sample_fps`.

**Response (shape; values come from the models):**

```json
{
  "upload_id": "…", "camera_id": "cos", "camera_name": "COS", "status": "completed | partial | failed",
  "video": {"fps": 25, "frame_count": 500, "duration_seconds": 20, "width": 640, "height": 720},
  "analysis": {
    "fighting": {"available": true, "status": "completed", "predicted_class": "NonFight",
                 "fight_probability": 0.12, "nonfight_probability": 0.88, "threshold": 0.5,
                 "potential_incident": false, "events": [], "windows": [...]},
    "smoking": {"available": true, "status": "completed", "rule": "both", "summary": {...},
                "detections": [...], "potential_incident": false, "events": []},
    "restricted_zone": {"available": false, "status": "unavailable", "reason": "…", "events": []}
  },
  "potential_incident_count": 0,
  "processing": {"device": "cuda", "duration_seconds": 12.3, "module_seconds": {...}},
  "disclaimer": "All results are POTENTIAL incidents …"
}
```

Each event has `event_id`, `type`, `start_s`, `end_s`, `confidence`, `description` and `evidence[]`. Each evidence item has `time_s`, `frame_index`, `confidence`, `image` and `details`.

### Express (5000)

| Method | Path | Notes |
|---|---|---|
| POST | `/api/videos/upload` | Unchanged |
| POST | `/api/videos/:uploadId/analyze` | Body `{camera_id, recorded_at?, force?, original_name?}` → `{analysis, incidents}` |
| GET | `/api/ai/models/status` | Proxy |
| GET | `/api/incidents` | `?status&type&cameraId&videoId&page&limit` |
| GET | `/api/incidents/stats` | |
| GET | `/api/incidents/:id` | |
| PATCH | `/api/incidents/:id/review` | `{status: reviewed \| dismissed \| pending_review, note?}` |
| GET | `/api/incidents/:id/evidence/:n` | Evidence image proxied from FastAPI |

**How incidents are stored:**
- Incidents are **upserted** by `(videoId, incidentType, eventKey)`. Re-analysing a video therefore never duplicates incidents and never resets a review.
- If MongoDB is unreachable, the backend answers `503 DATABASE_UNAVAILABLE`. FastAPI has cached the analysis, so a retry is instant.

## Environment variables (all optional)

**AI service:**

| Variable | Default | Purpose |
|---|---|---|
| `AI_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `AI_PRELOAD_MODELS` | `1` | Load models in the background at startup |
| `AI_UPLOAD_DIR` | `uploads` | Where uploaded videos are |
| `AI_RESULTS_DIR` | `analysis_results` | Cached analysis results |
| `AI_EVIDENCE_DIR` | `evidence` | Evidence images |
| `FIGHT_CHECKPOINT` | | Fighting model path |
| `SMOKING_CNN_CHECKPOINT` | | Smoking CNN path |
| `CIGARETTE_CHECKPOINT` | | Cigarette YOLO path |
| `PERSON_WEIGHTS` | | Person detector path |
| `ZONE_CONFIG_DIR` | | Per-camera zone YAML folder |
| `FIGHT_THRESHOLD` | from checkpoint | |
| `FIGHT_WINDOW_SECONDS` | `5` | |
| `FIGHT_MAX_WINDOWS` | `24` | |
| `SMOKING_SAMPLE_INTERVAL_S` | `0.5` | |
| `SMOKING_MAX_FRAMES` | `240` | |
| `SMOKING_CNN_THRESHOLD` | `0.5` | |
| `CIGARETTE_CONFIDENCE` | `0.35` | |
| `SMOKING_RULE` | `both` | `both`, `either`, `cnn` or `yolo` |
| `SMOKING_MIN_FRAMES` | `2` | |
| `ZONE_SAMPLE_FPS` | `5` | |
| `EVENT_GAP_S` | `2` | |

All paths are relative to `ai-service`.

**Backend:** `AI_ANALYSIS_TIMEOUT_MS` (default 1800000 = 30 min). The existing settings are unchanged.

**No new Python or npm dependencies.** Everything uses packages already installed: FastAPI, PyTorch, TorchVision, Ultralytics, OpenCV, Pillow, Mongoose, Axios.

## Running (CMD, three windows plus MongoDB service)

```cmd
:: 1) AI service, started as you normally start your FastAPI app, e.g.:
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

:: 2) Backend
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\backend
npm run dev

:: 3) Frontend
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\frontend
npm run dev
```

Open http://localhost:5173. Pick a camera, choose a video and click Upload. Analysis starts automatically. Then review the results on the Incidents page.

## Testing

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat
python -m unittest discover -s inference\tests -t . -v
python -m unittest discover -s training\restricted_zone\tests -t .
python -m unittest discover -s training\fighting\tests -t .
curl.exe http://127.0.0.1:8000/models/status

cd ..\backend
node --test tests\incident.service.test.js
```

End-to-end through the backend:

```cmd
curl.exe -F camera_id=cos -F "file=@C:\path\to\clip.mp4" http://localhost:5000/api/videos/upload
:: copy aiService.upload_id from the answer:
curl.exe -H "Content-Type: application/json" -d "{\"camera_id\":\"cos\",\"recorded_at\":\"2026-11-16T22:00\"}" http://localhost:5000/api/videos/<upload_id>/analyze
curl.exe http://localhost:5000/api/incidents/stats
```

## Known limitations

- **Accuracy is unknown.** No model has been evaluated on a held-out test set. Fighting validation metrics are optimistic, and RWF-2000 has no test split.
- **Domain gap.** All training data differs from campus cameras.
- **Fighting:** 5 s windows mirror training. A fight that straddles two windows may score lower in each.
- **Smoking:**
  - The CNN classifies whole frames. It is weak for small or distant people.
  - The fusion rule is unvalidated.
  - Cigarette detection is not proof of smoking.
- **Restricted zone:**
  - The zones are **fictional** and drawn without seeing your clips. Redraw them per camera with `python -m training.restricted_zone.pick_zone --source <video>`.
  - Track ID switches create extra incidents, one per new ID.
  - Recording time must be supplied for meaningful time rules.
- **Speed:** analysis is synchronous within the request. Long videos keep the browser waiting, up to 30 minutes on CPU. A background job queue would be the next step for long recordings.
- **No authentication** yet.
