# CampusSentinel

AI-assisted campus safety and anomaly detection system.

CampusSentinel processes CCTV/video feeds, flags **possible** unsafe activity, and sends each one to an administrator dashboard as an incident for **human review**:

1. **Smoking** — custom CNN (trained from scratch) + YOLO cigarette detection, with an experimentally validated fusion strategy.
2. **Fighting** — video-based action recognition (temporal model over clips, not single frames).
3. **Restricted zone / restricted time** — pretrained YOLO person detection + tracking + polygon zones + time rules.

The administrator reviews each incident and, if needed, contacts campus security by walkie-talkie.
**The AI never dispatches guards and never makes disciplinary decisions. Every detection may be wrong.**

---

## Current status

**Phase 1 — Project foundation: in progress.** Backend foundation tested; frontend foundation delivered. See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for detailed progress.

## Documentation

| Document | Purpose |
|---|---|
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | Progress log: phases, files, tests actually performed, decisions, next step |
| [`docs/architecture.md`](docs/architecture.md) | Components, where they run, how they communicate, folder structure, design decisions |
| [`docs/api-contract.md`](docs/api-contract.md) | Draft REST endpoints, Socket.IO events, and data shapes between components |
| [`docs/datasets.md`](docs/datasets.md) | Dataset plan, verification checklist, and (later) verified sources/licenses/splits |
| [`docs/model-cards/`](docs/model-cards/README.md) | One card per model: architecture, data, real metrics, limitations |

## Technology

- **AI service:** Python 3.12, FastAPI, OpenCV, PyTorch, Ultralytics YOLO, NumPy
- **Backend:** Node.js, Express, MongoDB + Mongoose, Socket.IO, JWT
- **Frontend:** React + Vite (JavaScript), Axios, Socket.IO client, plain CSS
- **Training:** Google Colab (GPU); trained weights are downloaded into `ai-service/models/`
- **Development:** Windows 11 (native), PowerShell, VS Code. No WSL. No Git for now.

## Setup and execution

Detailed instructions: [`backend/README.md`](backend/README.md), [`frontend/README.md`](frontend/README.md).

**Backend** (verified on Windows, 2026-11-13):

```powershell
cd "C:\Users\hp\Projects\CampusSentinel-Claude\backend"
Copy-Item .env.example .env   # first time only
npm install                   # first time only
npm run dev
```
Check: `curl.exe -i http://localhost:5000/api/health` → HTTP 200, `"status":"ok"`.

Frontend and AI service instructions will be added here once they have been run successfully.

Planned local ports:

| Component | URL |
|---|---|
| Frontend (Vite dev server) | http://localhost:5173 |
| Backend (Express + Socket.IO) | http://localhost:5000 |
| AI service (FastAPI) | http://localhost:8000 |
| MongoDB | mongodb://localhost:27017 (local only — never exposed) |
