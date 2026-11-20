# CampusSentinel — Project Status

_Last updated: 2026-11-13_

## Phases

| Phase | Name | Status |
|---|---|---|
| 0 | Environment and architecture | ✅ Complete (documentation only) |
| 1 | Project foundation | ✅ Backend + frontend tested · AI-service FastAPI foundation deferred (developer decision) |
| 2 | Minimal end-to-end workflow (test incident → MongoDB → live dashboard) | ⏸ Incident workflow still deferred · **Video upload path built:** dashboard → Express `/api/videos/upload` → FastAPI `/videos/upload` (backend ✅ verified by developer; dashboard panel delivered, awaiting test) |
| 3 | Smoking detection (CNN + YOLO + hybrid evaluation) | 🔄 CNN ✅ trained + checkpoint verified (developer report; metrics not yet recorded) · YOLO cigarette: dataset prepared ✅, training + inference scripts delivered, not trained yet |
| 4 | Fighting detection (video action recognition) | 🔄 Preprocessing ✅ (2,000 clips verified by developer) · CNN-LSTM model + train/evaluate/predict delivered, not trained yet |
| 5 | Restricted zone / time detection | 🔄 Module delivered (pretrained YOLO26n + ByteTrack + zones + rules); awaiting developer test |
| 6 | Complete integration | Not started |
| 7 | Testing, documentation, demo | Not started |

**Current phase:** Phase 5 — restricted-zone/time module (built ahead of YOLO cigarette and fighting, which wait on dataset downloads). Final test-set evaluation of all models deferred until all are built.

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
| `ai-service/training/smoking/cnn/train.py` | 3.3 | Training loop: AMP, early stopping, ReduceLROnPlateau, best checkpoint, history CSV, plots, resume |
| `ai-service/training/smoking/cnn/metrics.py` | 3.3 | Accuracy, precision, recall, F1, ROC-AUC, confusion counts |
| `ai-service/training/smoking/cnn/checkpoint.py` | 3.3 | Save best checkpoint + metadata; `load_smoking_cnn()` for inference |
| `ai-service/training/smoking/cnn/verify_checkpoint.py` | 3.3 | Reload best checkpoint safely and recompute validation metrics |
| `ai-service/requirements.txt` | 5 | ultralytics==8.4.212, lap, tzdata (torch installed separately with CUDA) |
| `ai-service/training/restricted_zone/` (config.py, schedule.py, zones.py, detector.py, rules.py, demo_video.py, pick_zone.py) | 5 | Restricted-zone module |
| `ai-service/training/restricted_zone/configs/zones.sample.yaml` | 5 | Sample config, two fictional zones |
| `ai-service/training/restricted_zone/trackers/bytetrack_campus.yaml` | 5 | ByteTrack defaults with track_buffer 60 |
| `ai-service/training/restricted_zone/tests/test_restricted_zone.py` | 5 | 25 synthetic tests |
| `ai-service/training/restricted_zone/README.md`, `ai-service/models/person/README.md` | 5 | Module docs; pretrained weight provenance |
| `backend/src/routes/video.routes.js`, `controllers/video.controller.js`, `services/aiService.client.js`, `middleware/videoUpload.js` (+ `config/env.js`, `routes/index.js`, `.env.example` edits, `VIDEO_UPLOAD.md`) | Upload | `POST /api/videos/upload`: Multer temp storage, camera/extension/size checks, streams to FastAPI, temp file always deleted, AI-service errors mapped to 503/504/502/4xx |
| `frontend/src/components/VideoUploadPanel.jsx`, `services/videoService.js`, `constants/cameras.js`, `styles/upload.css` (+ `pages/DashboardPage.jsx` edit) | Upload | Dashboard upload panel for 4 simulated cameras, progress/cancel, result from FastAPI |
| `ai-service/training/fighting/config.py`, `model.py`, `metrics.py`, `checkpoint.py` | 4.2 | CNN-LSTM (frozen ImageNet ResNet18 → LSTM → 1 logit), metrics, checkpoint save/load |
| `ai-service/training/fighting/train.py`, `evaluate.py`, `predict.py`, `verify_checkpoint.py` | 4.2 | Training (AdamW, early stop, ReduceLROnPlateau, optional AMP, resume), VALIDATION evaluation, prediction, checkpoint verification |
| `ai-service/training/fighting/tests/test_cnn_lstm.py`, README (updated), `models/fighting/cnn_lstm/README.md` | 4.2 | 14 model/pipeline tests; docs |
| `ai-service/training/fighting/prepare_data.py` | 4.1 | RWF-2000 → 16×224×224×3 uint8 RGB .npy (uniform sampling, letterbox), manifest.csv, report, resume/force/limit |
| `ai-service/training/fighting/dataset.py` | 4.1 | `FightingVideoDataset` (manifest-driven; float32 (16,3,224,224), ImageNet-normalised; long labels) |
| `ai-service/training/fighting/inspect_processed.py` | 4.1 | Output checks + frame-grid previews |
| `ai-service/training/fighting/tests/test_fighting_preprocessing.py`, `README.md` | 4.1 | 17 unit tests; module docs |
| `ai-service/training/smoking/yolo/train_yolo.py` | 3.6 | YOLO26n fine-tuning (train/val only), resume, OOM handling, copies best.pt + best.json to `models/smoking/yolo/finetuned/` |
| `ai-service/training/smoking/yolo/cigarette_detector.py` | 3.6 | Reusable `CigaretteDetector` (detect, detect_and_draw) for future FastAPI |
| `ai-service/training/smoking/yolo/infer_yolo.py` | 3.6 | CLI inference on image/video → annotated output + detections.json |
| `ai-service/training/smoking/yolo/prepare_dataset.py` | 3.5 | Validates cigarette YOLO dataset; group-aware 80/20 train/val split (seed 42); copies to `ai-service/datasets/smoking_yolo/` with data.yaml + report |
| `ai-service/training/smoking/yolo/README.md` | 3.5 | Updated: dataset facts (developer-reported), preparation rules |
| `ai-service/training/smoking/yolo/inspect_assets.py` | 3.4 | Read-only search for YOLO datasets, YAMLs, label folders, COCO/VOC annotations, weights; environment report |

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
| ai-service | torchvision | matching torch 2.15.0, `cu134` index | Installed (Stage 2 ran successfully) |
| ai-service | matplotlib | latest | Installed (CNN training ran) |
| ai-service | ultralytics | 8.4.212 (pinned) | Not installed before Phase 5 (inspection); via `requirements.txt`, awaiting confirmation |
| ai-service | lap | ≥0.5.12 | Required by ByteTrack; via `requirements.txt` |
| ai-service | tzdata | latest | IANA time zones on Windows; via `requirements.txt` |

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
| 2026-11-13 | Stage 2 on real `dataset_smoking` | No unreadable images, no exact duplicates, no train/test or cross-class duplicates; splits disjoint; batch shapes OK; val deterministic; CNN forward OK. **Split: train 4,536 (2,198 N / 2,338 S), val 1,134 (550 N / 584 S), test 396 (200 N / 196 S)** | Developer (Windows) |
| 2026-11-18 | Backend `/api/videos/upload` → real FastAPI | Working (developer report): Express returns FastAPI JSON (`status: uploaded`, `upload_id`, …) | Developer (Windows) |
| 2026-11-18 | Dashboard upload panel: `npm run build`; headless-Chrome run against Express + a FastAPI stand-in returning the reported response shape | Build OK. Validation (no camera/no file/.txt), multipart POST to `/api/videos/upload` with `camera_id`+`file`, result display, upload-another, progress + cancel (throttled), FastAPI down → AI_SERVICE_UNAVAILABLE + hint, backend down → hint, mobile layout. No console errors | Claude, Linux sandbox — **not against the real FastAPI, not on Windows** |
| 2026-11-15 | Fighting preprocessing on full RWF-2000 | 2,000 valid manifest rows, 0 failed, 0 missing .npy, 0 label/folder mismatches, correct shapes/dtypes | Developer (Windows) |
| 2026-11-15 | CNN-LSTM: syntax checks, 31 unit tests (14 new: init/frozen backbone, layer4 option, forward shape/finite, batched-reshape order, BatchNorm stats unchanged when frozen, multilayer LSTM, metrics, dataset/labels, checkpoint round-trip with weights_only, predictor incl. video, train→resume→evaluate→verify) — 1 CUDA test skipped (no GPU); all CLI --help | 30 passed, 1 skipped. ResNet18 used with random weights (ImageNet download blocked in sandbox) | Claude, Linux sandbox CPU — **no real training, no GPU, pretrained-weight loading untested** |
| 2026-11-15 | RWF-2000 spot check | 20 random videos readable in OpenCV; 30 fps, 150 frames, 5 s, 320×240–1280×720. Layout: train 800/800, val 200/200 (Fight/NonFight) | Developer (Windows) |
| 2026-11-15 | Fighting preprocessing: syntax check, 17 unit tests (sampling, letterbox/aspect, BGR→RGB, shape/dtype, labels, corrupt/empty/truncated video, manifest, resume incl. damaged .npy and lost manifest, --force, --limit, missing folder, dataset loader + errors), CLI run on 14 synthetic AVIs (4 resolutions, 1 truncated) + inspect_processed | All passed; truncated file resampled with WARNING; source checksums unchanged; 2.3 MB per clip | Claude, Linux sandbox (OpenCV 5.0.0) — **not on real RWF-2000** |
| 2026-11-14 | `prepare_dataset` on real cigarette dataset | Processed: train 1,080 (853 cigarette / 227 background), val 270 (236 / 34), test 318 (all cigarette). 0 invalid images, 0 invalid labels, 0 orphan labels, 0 test leakage, 0 warnings | Developer (Windows) |
| 2026-11-14 | YOLO training/inference scripts on synthetic noise data, CPU, imgsz 160 | Pipeline mechanics work: base download to `base/`, training, val re-check, copy to `finetuned/`, `best.json`, interrupt + `--resume` (continued from saved epoch), early stopping, image + video inference, COCO model rejected. Resuming a FINISHED run was found to start a new 100-epoch run → guard added. Metrics meaningless (noise data) | Claude, Linux sandbox — **no real training** |
| 2026-11-14 | Cigarette dataset contents | Developer reports `Cigarettes Detection.v13i.yolov8`: train 1,350 images / 1,350 labels, test 318 / 318, no valid/, data.yaml with 1 class `cigarette`. Source, licence and annotation validity **not yet verified** | Developer (Windows) |
| 2026-11-14 | `prepare_dataset.py` on a synthetic Roboflow-style dataset (bad class, polygon line, out-of-range and out-of-image boxes, missing label, undecodable image, duplicate stems, orphan label, train/test leakage) | All detected and excluded/reported; augmented copies kept in one split; rerun refused without `--overwrite`; source checksums unchanged; Ultralytics `check_det_dataset` accepted the data.yaml | Claude, Linux sandbox — **not on the real dataset** |
| 2026-11-14 | Restricted-zone unit tests (25: detection parsing, point-in-polygon incl. concave, inside/outside, allowed vs restricted time, overnight, weekdays, multiple zones, dwell, cooldown/duplicate suppression, invalid configs) | 25/25 passed | Claude, Linux sandbox (Python 3.13, Ultralytics 8.4.212) — not on developer machine |
| 2026-11-14 | Restricted-zone demo on a 75-frame synthetic clip made from Ultralytics' sample image, CPU | Detection, tracking, zone drawing, MP4, incidents.json and evidence frames produced; 2 potential incidents at simulated Mon 21:30, 0 at Mon 10:00 | Claude, Linux sandbox — **not real CCTV, not on developer machine** |
| 2026-11-13 | SmokingCNN training on RTX 4050 + `verify_checkpoint` | Developer reports training complete and checkpoint verified. **Training log / validation metrics not yet shared — to be recorded** | Developer (Windows) |
| 2026-11-13 | Stage 3 `train.py` on synthetic data (CPU): train, resume, early stopping, checkpoint reload with `weights_only=True`; ROC-AUC matched scikit-learn on random data | Worked as designed. **No real training run yet** | Claude, Linux sandbox |
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
| D20 | CNN trained locally on RTX 4050 (not Colab): Adam 1e-3, batch 32, AMP, ≤20 epochs, early stop patience 5 on val loss, ReduceLROnPlateau (×0.5, patience 2); best checkpoint by val loss |
| D21 | Final test-set evaluation postponed: all models (CNN, YOLO, …) built first, then evaluated together |
| D22 | Person detector: pretrained Ultralytics YOLO26n (COCO), float16 on CUDA; tracker ByteTrack (track_buffer 60); weights in `ai-service/models/person/` |
| D23 | Zone membership by bottom-centre of box (normalised); OpenCV point-in-polygon, edge = inside |
| D24 | Schedules evaluated in configured `campus_timezone`; precedence restricted window > allowed window > default; overnight windows belong to start day |
| D25 | Incidents: `restricted_zone_entry`, `pending_review`; min dwell 1 s; cooldown 300 s per (track, zone) |
| D26 | Cigarette YOLO data: original export untouched; processed copy in `ai-service/datasets/smoking_yolo/`; val = 20% of source train, split by source-photo group (Roboflow `.rf.` copies kept together), seed 42; train groups overlapping test are excluded; invalid files excluded and listed in `prepare_report.json` |
| D27 | Cigarette detector: Ultralytics YOLO26n (COCO-pretrained) fine-tuned for 1 class; imgsz 640, batch 8, AMP, ≤50 epochs, patience 15, seed 42, Ultralytics default optimiser/augmentation; best = highest val mAP50-95; test split not loaded |
| D28 | Fighting preprocessing: 16 uniformly spaced frames (first+last), sequential decode, letterbox to 224×224 (aspect kept, black pad), RGB uint8 .npy on D:; normalisation only at load time; official train/val split kept; no multiprocessing |
| D29 | Fighting model: CNN-LSTM — ImageNet ResNet18 (frozen, eval-mode BN, no_grad) → LSTM(hidden 256, 1 layer) → Dropout/Linear head → 1 logit; BCEWithLogitsLoss, AdamW 1e-3/wd 1e-2, batch 4, ≤20 epochs, early stop patience 5 + ReduceLROnPlateau on val loss, clip-level hflip, workers 0, AMP optional; best = lowest val loss |
| D30 | RWF-2000 has no test split: fighting results are reported as VALIDATION results only (optimistic) |
| D31 | Simulated camera sources (no physical CCTV): `library` Outside Library, `b_block` B Block, `cos` COS, `kravings` Kravings. Uploads go dashboard → Express → FastAPI only; max 500 MB; MP4/AVI/MOV/MKV; AI analysis of uploads not connected yet |
| D17 | Fusion thresholds tuned only on the CNN validation split; `dataset_smoking/testing` used once for final comparison |

Proposed, to confirm later: JWT in `sessionStorage` with 8 h expiry; evidence retention for dismissed incidents (`EVIDENCE_RETENTION_DAYS`).

## Known issues / risks

- **RWF-2000 access** may require a signed academic agreement and waiting time — verify early (Phase 4 prep).
- **VRAM budget** (~5.5 GB usable) limits how many models can run at once; models must be small and shared.
- **Datasets for smoking / cigarette** not yet identified; quality and licensing unknown.
- Run project commands from a **normal (non-admin)** PowerShell to avoid file-permission problems.
- **`dataset_smoking` source and license unknown** — needed for documentation and for near-duplicate/leakage checks (Stage 2).
- Image counts after scanning (pool 5,670; test 396) are lower than the folder counts originally reported (5,687; 400). No files were unreadable or duplicated, so the difference is probably files with extensions the scanner ignores (e.g. `.jfif`, `.gif`) — check the "Ignored non-image files" lines of `prepare_data` output.
- SmokingCNN receptive field is 46×46 px; likely weak on small/distant people in CCTV frames without person crops.
- Phase 2 deferred: no incident path exists yet between the AI service and the backend.
- Restricted-zone: recorded video uses simulated timestamps (`--start-time`); live CCTV integration not implemented. Track ID switches can cause duplicate incidents. Ultralytics/YOLO26 weights are AGPL-3.0.
- Cigarette dataset: annotations validated; **Roboflow source/licence text still to be recorded** in the YOLO README/datasets doc.
- RWF-2000 licence/terms of use not yet recorded.
- Cigarette detector confidence threshold (0.25) is the untuned Ultralytics default; tune on val only.

## Next step

Developer: test the dashboard upload panel end-to-end with the real FastAPI. Next integration step: connect stored uploads to the AI models (FastAPI side) and create incidents in the backend. Pending model results: YOLO, SmokingCNN metrics, restricted-zone tests, fighting CNN-LSTM training.
