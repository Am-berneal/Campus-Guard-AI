# CampusSentinel — Project Status

_Last updated: 2026-11-13_

## Phases

| Phase | Name | Status |
|---|---|---|
| 0 | Environment and architecture | ✅ Complete (documentation only) |
| 1 | Project foundation | ✅ Backend + frontend tested · AI-service FastAPI foundation deferred (developer decision) |
| 2 | Minimal end-to-end workflow (test incident → MongoDB → live dashboard) | ⏸ Deferred — developer chose to start Phase 3 first (2026-11-13) |
| 3 | Smoking detection (CNN + YOLO + hybrid evaluation) | 🔄 Stage 1 ✅ verified · Stage 2 preprocessing delivered, awaiting run |
| 4 | Fighting detection (video action recognition) | Not started |
| 5 | Restricted zone / time detection | Not started |
| 6 | Complete integration | Not started |
| 7 | Testing, documentation, demo | Not started |

**Current phase:** Phase 3 — Smoking detection, Stage 2 (CNN data preprocessing). Deadline approaching: priority is a working demo over extra experiments.

---

## Environment (reported by the developer — not run by Claude)

| Item | Value | Source |
|---|---|---|
| OS | Windows 11 (native; no WSL) | Project brief |
| Python | 3.12.10 | Project brief |
| pip | 25.0.1 | Project brief |
| Node.js | v26.2.0 | Project brief |
| npm | 11.13.0 | Project brief |
| VS Code | 1.145.0 | Project brief |
| MongoDB Server | 8.3.8, Windows service running, connected on localhost:27017 | Project brief |
| MongoDB Shell | 2.10.0 | Project brief |
| Git | Installed, **not used** for now | Project brief |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB VRAM (~616 MiB used by desktop at idle), WDDM | `nvidia-smi`, 2026-11-12 |
| NVIDIA driver | 616.56, supports CUDA ≤ 13.4 | `nvidia-smi`, 2026-11-12 |
| Training | Google Colab | Project brief |

## Files created

| File | Phase | Purpose |
|---|---|---|
| `README.md` | 0 | Project overview, status, planned ports |
| `docs/PROJECT_STATUS.md` | 0 | This progress log |
| `docs/architecture.md` | 0 | Architecture, decisions, folder structure, dependency list |
| `docs/api-contract.md` | 0 | Draft API, Socket.IO events, data shapes, env variables |
| `docs/datasets.md` | 0 | Dataset plan + verification checklist (nothing verified yet) |
| `docs/model-cards/README.md` | 0 | Model card index + template |
| `backend/package.json`, `backend/.env.example`, `backend/README.md` | 1 | Backend package, config template, instructions |
| `backend/src/server.js`, `backend/src/app.js` | 1 | Entry point (startup/shutdown) and Express app |
| `backend/src/config/env.js`, `backend/src/config/database.js` | 1 | Env validation; Mongoose connection, status, ping |
| `backend/src/routes/*`, `controllers/health.controller.js`, `services/health.service.js` | 1 | `GET /api/health` |
| `backend/src/middleware/notFound.js`, `errorHandler.js`, `utils/AppError.js`, `utils/logger.js` | 1 | JSON 404, consistent error format, logging |
| `frontend/` (36 files) | 1 | React + Vite dashboard foundation (developer reported working, 2026-11-13) |
| `ai-service/README.md` | 3.1 | AI-service layout, venv + PyTorch CUDA install |
| `ai-service/training/common/reproducibility.py` | 3.1 | `set_seed()`, `get_device()` |
| `ai-service/training/smoking/cnn/model.py` | 3.1 | SmokingCNN architecture (from scratch) |
| `ai-service/training/smoking/cnn/inspect_model.py` | 3.1 | Summary, parameter count, dummy-input checks, GPU memory probe |
| `ai-service/training/smoking/cnn/README.md` | 3.1 | Architecture explanation, design decisions, memory guidance |
| `ai-service/training/smoking/yolo/README.md` | 3.1 | Planned YOLO workflow (no dataset yet) |
| `ai-service/training/smoking/fusion/README.md` | 3.1 | Fusion evaluation plan |
| `ai-service/models/smoking/{cnn,yolo}/README.md`, `inference/smoking/README.md` | 3.1 | Weight placement rules, Stage 7 placeholder |
| `docs/model-cards/smoking-cnn.md`, `cigarette-yolo.md` | 3.1 | Model cards (status: not trained) |
| `ai-service/training/smoking/cnn/prepare_data.py` | 3.2 | Readability check, exact-duplicate check (SHA-256), stratified 80/20 split → `splits/split_seed42.json` |
| `ai-service/training/smoking/cnn/dataset.py` | 3.2 | `SmokingDataset`, transforms (224×224, normalise, train-only flip ±rotation), `build_dataloaders()` |
| `ai-service/training/smoking/cnn/check_batches.py` | 3.2 | Leakage, shape, dtype, label checks + preview image |

## Dependencies installed

| Component | Package | Version range | Status |
|---|---|---|---|
| backend | express | ^5.3.0 | Installed (developer, 2026-11-13) |
| backend | mongoose | ^9.12.1 | Installed (developer, 2026-11-13) |
| backend | dotenv | ^18.0.10 | Installed (developer, 2026-11-13) |
| frontend | react, react-dom | ^19.3.1 | Installed |
| frontend | react-router | ^8.5.1 | Installed |
| frontend | axios | ^1.21.1 | Installed |
| frontend (dev) | vite ^8.3.8, @vitejs/plugin-react ^6.3.0 | — | Installed (developer reported frontend working) |
| ai-service | torch | 2.15.0 (CUDA wheel, `cu134` index) | Installed, CUDA working (developer, 2026-11-13) |
| ai-service | torchvision | matching torch 2.15.0, `cu134` index | Instructions delivered, not yet confirmed |

Backend `npm install`: 0 vulnerabilities reported. No nodemon: `npm run dev` uses Node's built-in `--watch`.

## Features implemented

- **Backend:** env validation; MongoDB connection via Mongoose (fail-fast at startup, auto-reconnect later); `GET /api/health` (200 `ok` / 503 `degraded`, includes DB ping); JSON 404; central JSON error handler; graceful shutdown on Ctrl+C.
- **Frontend:** sidebar layout (responsive), dashboard with labelled placeholder statistics, incidents empty state, live backend status indicator from `GET /api/health`, Vite proxy `/api` → backend.

## Tests actually performed

| Date | Test | Result | Performed by |
|---|---|---|---|
| 2026-11-12 | `nvidia-smi` | GPU detected, driver 616.56, CUDA ≤ 13.4 | Developer |
| 2026-11-12 | `Test-Path "C:\Users\hp\Projects\CampusSentinel-Claude"` | `True` (folder exists) | Developer |
| 2026-11-13 | Backend `npm install` | Successful, 0 vulnerabilities | Developer (Windows) |
| 2026-11-13 | Backend start (`npm run dev`) + `GET /api/health` | HTTP 200, `status: ok`, MongoDB connected to `campussentinel`, ping successful | Developer (Windows) |
| 2026-11-13 | `GET /api/does-not-exist` | HTTP 404 with expected JSON error body | Developer (Windows) |
| 2026-11-13 | Backend graceful shutdown (Ctrl+C) | Not reported yet | — |
| 2026-11-13 | Backend failure paths (missing `.env`, MongoDB unreachable → exit 1; 503 without DB; invalid JSON → 400) | Behaved as designed | Claude, Linux sandbox (Node 22, no MongoDB) — not on developer machine |
| 2026-11-13 | Frontend `npm install` + `npm run dev` + dashboard | Developer reported "frontend working" (individual checks 1–6 and backend shutdown log not reported in detail) | Developer (Windows) |
| 2026-11-13 | SmokingCNN `inspect_model.py` (CPU) | 8/8 checks passed, 454,945 params, initial loss ≈ 0.67 on random data | Claude, Linux sandbox, PyTorch 2.15.0 CPU — not on developer machine; GPU memory probe not run |
| 2026-11-13 | SmokingCNN `inspect_model.py` on RTX 4050, incl. AMP memory probe | 8/8 checks passed; AMP memory tests passed. Chosen config: 224 px, batch 32, AMP on | Developer (Windows, RTX 4050) |
| 2026-11-13 | Stage 2 scripts on a synthetic dataset (corrupt file, test-leak copy, cross-class and within-class duplicates) | Detected/excluded as designed; split reproducible across runs; all batch checks passed | Claude, Linux sandbox — **not on the real dataset** |
| 2026-11-13 | Frontend build + Vite proxy (503 passthrough with backend up/no DB; 502 with backend down) + screenshots desktop/mobile | Built and rendered as designed | Claude, Linux sandbox — not on developer machine |

## Decisions made

| ID | Decision |
|---|---|
| D1 | FastAPI for the AI service |
| D2 | PyTorch only (TensorFlow has no native Windows GPU support for Python 3.12); CNN still trained from scratch |
| D3 | AI service uploads evidence with the incident; backend stores it and serves it only to authenticated admins |
| D4 | No campus CCTV initially — recorded clips + webcam. Real CCTV later only with written permission |
| D5 | Files delivered as downloadable ZIPs; developer extracts and runs commands |
| D6 | Ports: frontend 5173, backend 5000, AI service 8000, MongoDB 27017 (localhost only) |
| D7 | Store timestamps in UTC; evaluate time rules in Asia/Kolkata |
| D8 | Zone polygons use normalized (0–1) coordinates |
| D9 | Windows native + PowerShell + VS Code; Colab for training; no Git |

| D10 | Express 5 + Mongoose 9 + ES modules; backend exits at startup if MongoDB is unreachable, reconnects automatically afterwards |
| D11 | MongoDB URI uses `127.0.0.1` (not `localhost`) to avoid IPv6 `::1` resolution issues |
| D12 | Frontend calls `/api` on its own origin; Vite dev proxy forwards to the backend → no CORS needed in development (CORS revisited for production) |
| D13 | `react-router` v8 (declarative mode) for page routing |

| D14 | Model training code lives in `ai-service/training/`, weights in `ai-service/models/`, local inference in `ai-service/inference/` (replaces Phase 0's root-level `training/`) |
| D15 | Phase order changed by developer: Phase 3 (smoking) before Phase 2 (end-to-end incident workflow) and before the FastAPI foundation |
| D16 | SmokingCNN: conv `bias=False` (redundant before BatchNorm); Kaiming init; output layer initialised near zero so initial loss ≈ ln 2 |
| D18 | Preprocessing: EXIF-orient → RGB → resize to 224×224 (no aspect preservation) → ImageNet mean/std normalisation; train-only augmentation: horizontal flip p=0.5, rotation ±10° |
| D19 | Exact duplicates excluded from train/val (not deleted): copies of test images, images in both classes, extra copies within a class. Test set never modified |
| D17 | Fusion thresholds tuned only on the CNN validation split; `dataset_smoking/testing` used once for final comparison |

Proposed, to confirm later: JWT in `sessionStorage` with 8 h expiry; evidence retention for dismissed incidents (`EVIDENCE_RETENTION_DAYS`).

## Known issues / risks

- **RWF-2000 access** may require a signed academic agreement and waiting time — verify early (Phase 4 prep).
- **VRAM budget** (~5.5 GB usable) limits how many models can run at once; models must be small and shared.
- **Datasets for smoking / cigarette** not yet identified; quality and licensing unknown.
- Run project commands from a **normal (non-admin)** PowerShell to avoid file-permission problems.
- **`dataset_smoking` source and license unknown** — needed for documentation and for near-duplicate/leakage checks (Stage 2).
- `dataset_smoking` has **no validation split**; one must be carved from the training folders (Stage 2).
- SmokingCNN receptive field is 46×46 px; likely weak on small/distant people in CCTV frames without person crops.
- Phase 2 deferred: no incident path exists yet between the AI service and the backend.

## Next step

Developer: install torchvision, run `prepare_data --check-only`, `prepare_data`, `check_batches`, and report output + preview image. Then Stage 2b: CNN training notebook (Colab).
