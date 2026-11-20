# Merging the AI integration into CampusSentinel

This ZIP contains **only new or modified files**, with paths relative to the project root. It does **not** contain:
- checkpoints, datasets, uploads or `.env` files;
- `ai-service/api/main.py` or `ai-service/api/routers/videos.py` (your files, not visible to Claude: see the manual change below);
- `package.json` (no new npm dependencies).

## 1. Back up first (copies only, nothing is deleted)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude
robocopy ai-service\api        _backup_before_ai_integration\ai-service\api        /E /XD __pycache__
robocopy ai-service\training   _backup_before_ai_integration\ai-service\training   /E /XD __pycache__
robocopy ai-service\inference  _backup_before_ai_integration\ai-service\inference  /E /XD __pycache__
robocopy backend\src           _backup_before_ai_integration\backend\src           /E
copy backend\.env.example      _backup_before_ai_integration\backend\
robocopy frontend\src          _backup_before_ai_integration\frontend\src          /E
copy docs\PROJECT_STATUS.md    _backup_before_ai_integration\docs\
```

(`robocopy` exit codes 0–7 mean success.)

## 2. Extract

```cmd
tar -xf "%USERPROFILE%\Downloads\campussentinel-ai-integration.zip"
```

## 3. Manual change: `ai-service\api\main.py` (two lines, plus one optional)

Add next to your existing router import and `include_router` call:

```python
from api.routers import analysis          # NEW (keep your existing `videos` import)

app.include_router(analysis.router)       # NEW (after app.include_router(videos.router))
analysis.preload_in_background()          # OPTIONAL: start loading models when the service starts
```

The AI service must be started **from the `ai-service` folder** (`python -m uvicorn api.main:app ...`). The new code imports the `inference` and `training` packages from there.

## 4. Optional

- **Person detector.** The code uses `models\smoking\yolo\base\yolo26n.pt` if `models\person\` has no weights. To give it its own copy:

  ```cmd
  copy ai-service\models\smoking\yolo\base\yolo26n.pt ai-service\models\person\yolo26n.pt
  ```

- **Backend `.env`:** add `AI_ANALYSIS_TIMEOUT_MS=1800000`. This is optional; it is the default.

## Files

**New:**
- `ai-service/AI_INTEGRATION.md`
- `ai-service/api/routers/analysis.py`
- `ai-service/inference/settings.py`, `model_manager.py`, `pipeline.py`, `video_io.py`
- `ai-service/inference/fighting/__init__.py`, `fighting/analyzer.py`
- `ai-service/inference/smoking/analyzer.py`
- `ai-service/inference/restricted_zone/__init__.py`, `restricted_zone/analyzer.py`
- `ai-service/inference/tests/__init__.py`, `tests/test_inference.py`
- `ai-service/training/common/ultralytics_compat.py`
- `ai-service/training/restricted_zone/configs/cameras/library.yaml`, `b_block.yaml`, `cos.yaml`, `kravings.yaml`
- `backend/src/models/Incident.js`
- `backend/src/services/incident.service.js`
- `backend/src/controllers/incident.controller.js`
- `backend/src/routes/incident.routes.js`, `ai.routes.js`
- `backend/tests/incident.service.test.js`
- `frontend/src/components/AnalysisResults.jsx`, `ModelStatusPanel.jsx`
- `frontend/src/services/incidentService.js`
- `frontend/src/styles/incidents.css`
- `MERGE_INTEGRATION.md`

**Modified (replace your copy):**
- `ai-service/training/smoking/yolo/cigarette_detector.py`: Ultralytics `quantize`/`half` compatibility.
- `ai-service/training/restricted_zone/detector.py`: same compatibility fix, plus `track(persist=...)` so each video starts with fresh track IDs.
- `backend/.env.example`: `AI_ANALYSIS_TIMEOUT_MS`.
- `backend/src/config/env.js`: `aiService.analysisTimeoutMs`.
- `backend/src/services/aiService.client.js`: analyze, model-status and evidence calls, plus more specific error mapping.
- `backend/src/controllers/video.controller.js`: `analyzeVideo`. Upload is unchanged.
- `backend/src/routes/index.js`: mounts `/incidents` and `/ai`.
- `backend/src/routes/video.routes.js`: `POST /:uploadId/analyze`.
- `frontend/src/components/VideoUploadPanel.jsx`: analysis step and results.
- `frontend/src/services/videoService.js`: `analyzeVideo()`.
- `frontend/src/pages/DashboardPage.jsx`: real stats and the model-status panel.
- `frontend/src/pages/IncidentsPage.jsx`: list, evidence and review.
- `frontend/src/styles/upload.css`, `components.css`: added rules only.
- `docs/PROJECT_STATUS.md`

If you changed any of the **modified** files since Claude last delivered them, compare them before replacing. The backup from step 1 keeps your version.
