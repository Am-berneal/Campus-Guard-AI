# Merging the closing-time update

**Apply this AFTER `campussentinel-ai-integration.zip`.** It changes only the 19 files below, plus this note.
It contains no checkpoints, datasets, uploads or `.env` files, and does not touch `ai-service/api/main.py`. No new dependencies.

## Back up first (copies only)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude
robocopy ai-service\training\restricted_zone _backup_before_closing_time\ai-service\training\restricted_zone /E /XD __pycache__
robocopy ai-service\inference _backup_before_closing_time\ai-service\inference /E /XD __pycache__
robocopy ai-service\api _backup_before_closing_time\ai-service\api /E /XD __pycache__
copy ai-service\AI_INTEGRATION.md _backup_before_closing_time\ai-service\
robocopy backend\src _backup_before_closing_time\backend\src /E
robocopy backend\tests _backup_before_closing_time\backend\tests /E
robocopy frontend\src _backup_before_closing_time\frontend\src /E
copy docs\PROJECT_STATUS.md _backup_before_closing_time\docs\
```

## Extract

```cmd
tar -xf "%USERPROFILE%\Downloads\campussentinel-closing-time.zip"
```

No manual edits are needed.

## Files

**New:**
- `ai-service/inference/tests/test_closing_time.py`
- `MERGE_CLOSING_TIME.md`

**Modified:**
- `ai-service/training/restricted_zone/config.py`: optional `camera_name`, `closing_time`, `restricted_until`. Older configs still load.
- `ai-service/training/restricted_zone/zones.py`: optional zone flag `calibrated` (default `false`).
- `ai-service/training/restricted_zone/configs/cameras/library.yaml`, `b_block.yaml`, `cos.yaml`, `kravings.yaml`: official closing times and placeholder polygons.
- `ai-service/inference/restricted_zone/analyzer.py`: recording-time rule, `time_unknown` fallback, closing and recorded time in each event.
- `ai-service/api/routers/analysis.py`: the cache is keyed by camera **and** recording time; the request is echoed back.
- `ai-service/inference/tests/test_inference.py`: adapted to the safe fallback, plus a cache test.
- `ai-service/AI_INTEGRATION.md`
- `backend/src/models/Incident.js`: optional `closingTime`, `recordedAt`, `recordedLocalTime`, `zone.calibrated`. Existing records stay valid.
- `backend/src/services/incident.service.js`: maps the new fields.
- `backend/tests/incident.service.test.js`
- `frontend/src/components/AnalysisResults.jsx`, `frontend/src/components/VideoUploadPanel.jsx`, `frontend/src/pages/IncidentsPage.jsx`, `frontend/src/services/incidentService.js`
- `docs/PROJECT_STATUS.md`

## Behaviour change to know

Restricted-zone analysis **without a recording time** now returns `time_unknown` and creates no incidents. Previously it used the analysis time. Enter the recording start time on the dashboard to get restricted-zone results.
