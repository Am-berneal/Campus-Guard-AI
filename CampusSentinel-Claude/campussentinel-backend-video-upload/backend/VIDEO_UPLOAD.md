# Video upload forwarding: `POST /api/videos/upload`

Receives a video from the dashboard and forwards it to the Python FastAPI service
(`AI_SERVICE_URL` + `/videos/upload`). No incident is created and no AI result is interpreted here.

| Multipart field | Rule |
|---|---|
| `camera_id` | `library`, `b_block`, `cos` or `kravings` (send it **before** `file` to be rejected early) |
| `file` | `.mp4`, `.avi`, `.mov`, `.mkv` (case-insensitive), 1 byte – 500 MB |

Flow: Multer writes the upload to `backend/storage/tmp-uploads/` under a random name →
controller validates → `services/aiService.client.js` streams it to FastAPI → the temp file is
deleted in every case (success or error) before the reply is sent. Leftovers older than 24 h are
removed at startup.

| Result | HTTP | `error.code` |
|---|---|---|
| Forwarded | FastAPI's 2xx | — (body: `cameraId`, `originalName`, `sizeBytes`, `aiService` = FastAPI JSON) |
| Bad camera / type / empty / missing file / wrong field | 400 | `INVALID_CAMERA_ID`, `UNSUPPORTED_FILE_TYPE`, `EMPTY_FILE`, `FILE_REQUIRED`, `UNEXPECTED_FIELD` |
| Not multipart | 415 | `UNSUPPORTED_MEDIA_TYPE` |
| Over 500 MB | 413 | `FILE_TOO_LARGE` |
| FastAPI rejected it (4xx) | same 4xx | `AI_SERVICE_REJECTED` (+ short FastAPI `detail`) |
| FastAPI failed (5xx) | 502 | `AI_SERVICE_ERROR` (FastAPI body not passed on) |
| FastAPI not running | 503 | `AI_SERVICE_UNAVAILABLE` |
| FastAPI too slow | 504 | `AI_SERVICE_TIMEOUT` |

Settings (`.env`): `AI_SERVICE_URL`, `AI_SERVICE_TIMEOUT_MS` (default 600000), `UPLOAD_TMP_DIR`
(default `storage/tmp-uploads`), `VIDEO_UPLOAD_MAX_MB` (default 500).

Test (CMD, backend and FastAPI running):

```cmd
curl.exe -i -F camera_id=library -F "file=@C:\path\to\clip.mp4" http://localhost:5000/api/videos/upload
```
