# CampusSentinel — API Contract (Draft v0.1)

_Status: **draft** from Phase 0. Nothing here is implemented yet. Each section is finalized when it is built and tested, and this file is updated to match the real code._

Base URLs (local development):
- Backend: `http://localhost:5000/api`
- AI service: `http://localhost:8000`

Conventions:
- JSON request/response bodies, `camelCase` field names.
- All timestamps: ISO 8601 in **UTC** (e.g. `2026-11-12T13:35:00.000Z`).
- Confidence values: number between `0` and `1`.
- Errors: `{ "error": { "code": "VALIDATION_ERROR", "message": "…", "details": [...] } }` with an appropriate HTTP status.

---

## 1. Authentication (admin)

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/login` | none | `{ email, password }` → `{ token, admin: { id, name, email } }` |
| GET | `/auth/me` | JWT | Current admin profile |

- Passwords are hashed (bcrypt); never stored or returned in plain text.
- There is **no public registration**. The first admin is created with a seed script (`backend/scripts/seed-admin.js`).
- Login failures return a generic message (do not reveal whether the email exists).

---

## 2. Cameras

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/cameras` | JWT | List cameras |
| POST | `/cameras` | JWT | Create camera |
| GET | `/cameras/:id` | JWT | Camera details |
| PATCH | `/cameras/:id` | JWT | Update camera |
| DELETE | `/cameras/:id` | JWT | Delete camera (blocked if it has zones, or zones deleted with it — decide in Phase 2) |

Camera shape:
```json
{
  "id": "…",
  "name": "Library entrance (test webcam)",
  "location": "Main library, ground floor",
  "sourceType": "file | webcam | rtsp",
  "sourceUri": "sample-videos/library_test.mp4 | 0 | rtsp://…",
  "enabledDetections": ["smoking", "fighting", "restricted_zone"],
  "isActive": true,
  "createdAt": "…",
  "updatedAt": "…"
}
```
`sourceUri` for RTSP may contain credentials later; it must never be sent to the browser in full (masked in responses).

---

## 3. Restricted zones

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/zones?cameraId=…` | JWT | List zones (optionally per camera) |
| POST | `/zones` | JWT | Create zone |
| GET | `/zones/:id` | JWT | Zone details |
| PATCH | `/zones/:id` | JWT | Update polygon / schedule / active flag |
| DELETE | `/zones/:id` | JWT | Delete zone |

Zone shape:
```json
{
  "id": "…",
  "cameraId": "…",
  "name": "Chemistry lab corridor",
  "polygon": [[0.12, 0.40], [0.55, 0.38], [0.60, 0.95], [0.10, 0.97]],
  "schedule": [
    { "days": [0, 1, 2, 3, 4, 5, 6], "start": "22:00", "end": "06:00" }
  ],
  "timezone": "Asia/Kolkata",
  "isActive": true
}
```
- `polygon`: at least 3 points; each `[x, y]` normalized to `0–1` (x = fraction of frame width from the left, y = fraction of frame height from the top).
- `days`: `0` = Sunday … `6` = Saturday.
- `start` > `end` means the window crosses midnight.

---

## 4. Incidents

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/incidents` | JWT | List with filters: `status`, `type`, `cameraId`, `from`, `to`, `includeTest`, `page`, `limit` |
| GET | `/incidents/stats` | JWT | Counts by status and type (dashboard cards) |
| GET | `/incidents/:id` | JWT | Full incident, including evidence list |
| PATCH | `/incidents/:id/review` | JWT | `{ status: "reviewed" \| "dismissed", reviewNote?, actionTaken? }` |

Incident shape:
```json
{
  "id": "…",
  "type": "smoking | fighting | restricted_zone",
  "isTest": false,
  "source": "ai_service | test_script",
  "cameraId": "…",
  "zoneId": "… (restricted_zone only)",
  "detectedAt": "2026-11-12T16:40:12.000Z",
  "confidence": 0.81,
  "detection": {
    "modelName": "smoking-cnn",
    "modelVersion": "v1",
    "modelOrigin": "from_scratch | fine_tuned | pretrained",
    "boundingBoxes": [{ "label": "cigarette", "confidence": 0.74, "box": [0.31, 0.22, 0.36, 0.27] }],
    "trackId": 17,
    "details": {}
  },
  "evidence": [
    { "id": "…", "kind": "image | video", "mimeType": "image/jpeg", "sizeBytes": 84211, "capturedAt": "…" }
  ],
  "status": "new | reviewed | dismissed",
  "review": {
    "reviewedBy": "admin id",
    "reviewedAt": "…",
    "reviewNote": "Looked like a pen, not a cigarette",
    "actionTaken": "Guard contacted by radio"
  },
  "createdAt": "…"
}
```
- Bounding boxes are normalized `[x1, y1, x2, y2]`.
- `modelOrigin` makes it explicit whether a result came from a model trained from scratch, fine-tuned, or used as pretrained.
- `actionTaken` is free text entered **by the admin**. The system never records an action it did not see a human take.
- Evidence file paths on disk are **not** exposed; the client uses the evidence endpoint below.

---

## 5. Evidence

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/incidents/:id/evidence/:evidenceId` | JWT | Streams the file. 404 if not found or not allowed |

Not served from a public static folder. (Detail for Phase 2: `<img>`/`<video>` tags cannot send an `Authorization` header, so the frontend will fetch the file with Axios as a blob, or the backend will issue a short-lived signed URL.)

---

## 6. AI service → backend (machine-to-machine)

All require header `X-Service-Key: <AI_SERVICE_KEY>`. JWTs are **not** accepted here, and the service key is **not** accepted on admin routes.

| Method | Path | Description |
|---|---|---|
| POST | `/ai/detections` | Submit a detection. `multipart/form-data`: field `metadata` (JSON, incident fields above minus status/review/ids) + zero or more `evidence` files |
| GET | `/ai/config` | Active cameras + zones + schedules the AI service should apply |

Backend responsibilities on `POST /ai/detections`:
1. Validate metadata (type, confidence range, camera exists, zone belongs to camera, file types/sizes).
2. Store evidence files, then the incident with `status: "new"`.
3. Emit Socket.IO `incident:new`.
4. Respond `201 { incidentId }`.

**Test incidents (Phase 2):** `backend/scripts/send-test-incident.js` calls this same endpoint with `isTest: true` and `source: "test_script"`, so the real path is exercised. The backend refuses `isTest: true` combined with `source: "ai_service"`, and the dashboard always labels test incidents **TEST**.

---

## 7. Socket.IO events (backend → dashboard)

Connection: client passes the JWT in the handshake (`auth: { token }`); invalid tokens are rejected.

| Event | Payload | When |
|---|---|---|
| `incident:new` | Incident summary: `{ id, type, isTest, cameraId, cameraName, confidence, detectedAt, status }` | New incident stored |
| `incident:updated` | `{ id, status, review }` | Admin reviews/dismisses (keeps other open dashboards in sync) |

Payloads are summaries; the dashboard fetches full details via REST.

---

## 8. AI service API (FastAPI)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + which models are loaded + GPU availability |
| GET | `/status` | Per-camera worker state, frames processed, last error |
| POST | `/cameras/{cameraId}/start` | Start processing a camera (later) |
| POST | `/cameras/{cameraId}/stop` | Stop processing (later) |

FastAPI auto-generates interactive docs at `http://localhost:8000/docs`.

---

## 9. Environment variables (planned)

**backend/.env**
```
PORT=5000
MONGODB_URI=mongodb://127.0.0.1:27017/campussentinel
JWT_SECRET=<long random string>
JWT_EXPIRES_IN=8h
AI_SERVICE_KEY=<long random string, same value as ai-service>
CORS_ORIGIN=http://localhost:5173
EVIDENCE_DIR=./storage/evidence
EVIDENCE_MAX_FILE_MB=20
EVIDENCE_RETENTION_DAYS=30
```

**ai-service/.env**
```
BACKEND_URL=http://localhost:5000/api
AI_SERVICE_KEY=<same as backend>
CAMPUS_TIMEZONE=Asia/Kolkata
DEVICE=cuda
```

**frontend/.env**
```
VITE_API_URL=http://localhost:5000/api
VITE_SOCKET_URL=http://localhost:5000
```

Only `.env.example` templates are ever shared. Real `.env` files stay on the developer's machine.
