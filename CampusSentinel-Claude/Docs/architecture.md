# CampusSentinel — Architecture

_Status: agreed in Phase 0 (2026-11-12). Update this file whenever a design decision changes._

---

## 1. System overview

```
 ┌──────────────────────┐
 │ Video sources        │  Phase 1–6: recorded clips + webcam
 │ (files / webcam /    │  Later (with written permission): RTSP CCTV
 │  RTSP later)         │
 └─────────┬────────────┘
           │ frames (OpenCV)
           ▼
 ┌──────────────────────────────────────────┐
 │ Python AI service  (FastAPI, port 8000)  │
 │  pipeline → detectors → rules → evidence │
 └─────────┬───────────────────▲────────────┘
           │ POST detection    │ GET cameras + zones
           │ + evidence files  │ (service API key)
           ▼                   │
 ┌──────────────────────────────────────────┐      ┌───────────────┐
 │ Node.js backend (Express, port 5000)     │◄────►│ MongoDB       │
 │  validation · auth · incidents · storage │      │ localhost:    │
 │  Socket.IO server                        │      │ 27017 only    │
 └─────────┬────────────────────────────────┘      └───────────────┘
           │ REST (JWT)  +  Socket.IO "incident:new"
           ▼
 ┌──────────────────────────────────────────┐
 │ React admin dashboard (Vite, port 5173)  │
 └─────────┬────────────────────────────────┘
           │ human decision
           ▼
   Administrator ──walkie-talkie──► Security guard
   (guard has no account and never interacts with the system)
```

### Core principle: the backend is the single source of truth

- The AI service **never writes to MongoDB directly**. It sends detections to the backend.
- The backend validates every input, stores it, controls access to evidence, and notifies the dashboard.
- Validation, authorization, and data ownership therefore live in **one** place.

---

## 2. Components: what runs where

| Component | Runs on | Port | Responsibility |
|---|---|---|---|
| AI service | Laptop (RTX 4050, ~5.5 GB usable VRAM) | 8000 | Read frames, run models, apply campus rules, extract evidence, send detections |
| Backend | Laptop | 5000 | REST API, JWT auth, Mongoose models, evidence storage, Socket.IO notifications |
| Frontend | Laptop (Vite dev server) | 5173 | Admin login, dashboard, incident review, camera + zone management |
| MongoDB | Laptop (Windows service) | 27017 | Persistent data; bound to localhost only |
| Training | Google Colab (GPU) | — | Dataset preparation, training, evaluation; outputs weight files |

Training artifacts flow **one way**: Colab → download weights → `ai-service/models/`. The AI service never trains.

---

## 3. Communication

| From → To | Mechanism | Authentication | Purpose |
|---|---|---|---|
| AI service → Backend | HTTP `POST` multipart (JSON metadata + evidence files) | `X-Service-Key` header (shared secret from `.env`) | Submit a detection / incident |
| AI service → Backend | HTTP `GET` | `X-Service-Key` | Fetch active cameras, zones and time rules |
| Backend → AI service | HTTP (optional, later) | Service key | Start/stop processing a camera, health check |
| Frontend → Backend | REST over Axios | `Authorization: Bearer <JWT>` | Everything the admin does |
| Backend → Frontend | Socket.IO | JWT checked during the socket handshake | Push `incident:new` / `incident:updated` events |
| Backend → MongoDB | Mongoose | Local connection string from `.env` | Persistence |

Why two different kinds of authentication?
- **JWT** identifies a *human administrator* who logged in with a password.
- The **service key** identifies a *machine* (the AI service). It is not a user and must not be able to log in to the dashboard. It can only submit detections and read configuration.

Full endpoint and payload details: [`api-contract.md`](api-contract.md).

---

## 4. Incident lifecycle

```
 AI detection (or labelled TEST incident)
        │
        ▼
 Backend validates → stores incident (status: "new") + evidence files
        │
        ▼
 Socket.IO "incident:new" → dashboard shows notification
        │
        ▼
 Admin opens incident, reviews evidence, type, confidence, camera, timestamp
        │
        ├──► "reviewed"  (admin may note action taken, e.g. "guard contacted by radio")
        └──► "dismissed" (false positive / not actionable)
```

Rules:
- Every incident starts as `new` and **requires a human decision**.
- The system never assigns blame, identifies individuals, or triggers dispatch.
- Test incidents carry `isTest: true` and are displayed with a visible **TEST** label. They are never presented as AI detections.

---

## 5. AI service internal design (planned)

```
pipeline/   one worker per camera: capture frames, sample at a configured rate,
            keep a short rolling buffer (needed for clip evidence and the fight model)
   │
   ▼
detectors/  smoking.py      CNN classification + YOLO cigarette boxes → fused score
            fighting.py     temporal model over a clip of N frames → fight probability
            person_zone.py  YOLO person detection + tracker → tracked people with IDs
   │
   ▼
rules/      polygon test (is the person's foot point inside the zone?),
            time-window test (is now within restricted hours, in campus time?),
            thresholds, cooldown/deduplication (avoid 300 incidents for one event)
   │
   ▼
evidence/   snapshot (JPEG) and/or short clip from the rolling buffer
   │
   ▼
clients/    send to backend; retry on failure
```

### Resource constraints (RTX 4050, 6 GB VRAM)
- ~616 MiB is already used by the desktop; plan for **≤ 5 GB** model memory in total.
- Load each model **once** and share it across camera workers.
- Prefer small model variants (e.g. YOLO nano/small); process frames at a reduced rate rather than every frame.
- Use one deep-learning framework only (PyTorch) to avoid two CUDA runtimes competing for VRAM.

---

## 6. Restricted-zone design notes

- **Polygon coordinates are normalized** (0.0–1.0 relative to frame width/height), not pixels. A zone drawn on a 1920×1080 preview still works if the stream later arrives at 1280×720.
- A zone belongs to **one camera**: a polygon only makes sense in that camera's image.
- **Which point of the person counts?** The bottom-centre of the bounding box (roughly the feet) is used, because the top of a box can overlap a zone while the person is standing outside it.
- **Camera calibration:** polygons are drawn in image space. If a camera is moved or zoomed, its zones must be redrawn. Real-world distances are not measured.
- **Time rules** are evaluated in the campus time zone (**Asia/Kolkata**). Windows may cross midnight (e.g. 22:00 → 06:00). All timestamps are **stored in UTC** in MongoDB.
- **Tracking IDs** let us say "person #17 entered the zone" once, instead of creating an incident for every frame.

---

## 7. Agreed decisions

| ID | Decision | Reason |
|---|---|---|
| D1 | **FastAPI** for the AI service API | Automatic request/response validation (Pydantic), auto-generated `/docs` for testing, good fit for background camera workers |
| D2 | **PyTorch only** (no TensorFlow) | TensorFlow has no native Windows GPU support after 2.10, and 2.10 does not support Python 3.12. Ultralytics YOLO already uses PyTorch. One CUDA runtime saves VRAM. The CNN is still written and trained **from scratch** |
| D3 | **AI service uploads evidence** to the backend with the incident (multipart) | Backend owns storage, access control, and retention in one place. Evidence is served only to authenticated admins, never as public static files |
| D4 | **No campus CCTV initially**: recorded clips + webcam | Avoids privacy issues during development. Real CCTV integration only with written permission |
| D5 | **Files delivered as ZIPs**, extracted by the developer | Developer runs every command and reports real output |
| D6 | Ports 5173 / 5000 / 8000 / 27017 | Defaults for Vite, common Express choice, FastAPI/Uvicorn default, MongoDB default |
| D7 | Timestamps stored in **UTC**; time rules evaluated in **Asia/Kolkata** | Avoids time-zone bugs; rules match campus clocks |
| D8 | Zone polygons stored as **normalized coordinates** | Resolution-independent |
| D9 | Native Windows 11, PowerShell, VS Code; Colab for training; **no Git** for now | Project constraint |

### Proposed, to confirm in Phase 1–2
- **JWT storage in the browser:** keep the token in `sessionStorage` with a short expiry (e.g. 8 h). Simple to understand; trade-off is exposure to XSS, mitigated by not rendering untrusted HTML. Alternative: httpOnly cookies (safer against XSS, but more setup: CSRF protection, cookie config for Socket.IO).
- **Evidence retention:** delete evidence of `dismissed` incidents after a configurable number of days (`EVIDENCE_RETENTION_DAYS`).

---

## 8. Folder structure

```
CampusSentinel-Claude/
├── README.md
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── architecture.md          (this file)
│   ├── api-contract.md
│   ├── datasets.md
│   └── model-cards/
│
├── backend/
│   ├── package.json
│   ├── .env.example
│   ├── src/
│   │   ├── server.js            starts HTTP server + Socket.IO
│   │   ├── app.js               Express app, middleware, routes
│   │   ├── config/              env loading, MongoDB connection
│   │   ├── models/              Admin, Camera, Zone, Incident
│   │   ├── routes/
│   │   ├── controllers/
│   │   ├── middleware/          JWT auth, service-key auth, validation, error handler
│   │   ├── validators/
│   │   ├── sockets/             Socket.IO auth + emit helpers
│   │   └── utils/
│   ├── scripts/                 seed-admin.js, send-test-incident.js
│   └── storage/evidence/        evidence files (served only via authenticated route)
│
├── frontend/
│   ├── package.json, vite.config.js, index.html, .env.example
│   └── src/
│       ├── api/                 Axios instance + endpoint functions
│       ├── socket/              Socket.IO client
│       ├── context/             auth state
│       ├── pages/               Login, Dashboard, Incidents, IncidentDetail, Cameras, Zones
│       ├── components/
│       └── styles/
│
├── ai-service/
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py              FastAPI app
│   │   ├── config.py
│   │   ├── pipeline/
│   │   ├── detectors/
│   │   ├── rules/
│   │   ├── evidence/
│   │   └── clients/
│   ├── models/                  real trained/pretrained weights only
│   └── scripts/                 run detectors on a video file
│
├── training/
│   ├── common/                  seeding, split utilities, metrics
│   ├── smoking_cnn/
│   ├── cigarette_yolo/
│   ├── hybrid_eval/
│   └── fighting/
│
└── sample-videos/               local test clips
```

Large datasets live in Google Drive (for Colab), not in this folder.

---

## 9. Dependencies (identified; exact versions pinned at install time)

Versions are **not** pinned here on purpose. They will be checked against the official install instructions when each component is installed, then recorded in `PROJECT_STATUS.md`.

**Backend (npm):** `express`, `mongoose`, `socket.io`, `jsonwebtoken`, `bcryptjs` (password hashing), `dotenv`, `cors`, `helmet` (security headers), `multer` (multipart uploads), a validation library (`zod` or `express-validator`, chosen in Phase 2), `morgan` (request logging). Dev: `nodemon`.

**Frontend (npm):** `react`, `react-dom`, `react-router-dom`, `axios`, `socket.io-client` (Vite scaffold provides the rest).

**AI service (pip, in a virtual environment):** `fastapi`, `uvicorn`, `pydantic-settings`, `httpx` (HTTP client), `python-multipart`, `opencv-python`, `numpy`, `torch` + `torchvision` (CUDA build matching driver; driver 616.56 supports CUDA ≤ 13.4), `ultralytics`. Tracking uses Ultralytics' built-in trackers (ByteTrack/BoT-SORT) — no extra package expected.

**Training (Colab):** `torch`, `torchvision`, `ultralytics`, `scikit-learn` (metrics, stratified splits), `matplotlib`, `opencv-python`, `tqdm`.
