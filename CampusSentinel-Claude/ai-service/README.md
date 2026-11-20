# CampusSentinel — AI Service

Python code for CampusSentinel's detection models and (later) the FastAPI inference service.

**Current scope: Phase 3, Stage 1** — smoking CNN architecture + checks, and the planned YOLO / fusion workflow (documentation only).
No model has been trained. No dataset has been downloaded. The FastAPI service has not been created yet.

## Layout

```
ai-service/
├── training/                 code that CREATES models (runs mainly in Google Colab)
│   ├── common/               shared helpers (seeding, device selection)
│   └── smoking/
│       ├── cnn/              SmokingCNN: model.py (architecture), inspect_model.py (Stage 1 checks)
│       ├── yolo/             cigarette detector — planned workflow (Stage 3+)
│       └── fusion/           CNN + YOLO hybrid evaluation — plan (Stage 6)
├── inference/                code that USES trained models locally (Stage 7)
│   └── smoking/
└── models/                   trained weight files ONLY — never source code
    └── smoking/
        ├── cnn/              our from-scratch CNN checkpoint (after training)
        └── yolo/             base/ = downloaded pretrained weights, finetuned/ = our best.pt
```

`training/` and `inference/` are Python packages, so commands are run **from the `ai-service` folder** with `python -m ...`.

## Setup (Command Prompt, Windows)

One virtual environment for the AI service, kept inside `ai-service\.venv`:

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

Install PyTorch **with CUDA** (the plain `pip install torch` on Windows gives a CPU-only build):

```cmd
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
```

- `cu134` = built for CUDA 13.4. Your driver (616.56) reports support up to CUDA 13.4, so this matches.
- If pip reports "No matching distribution", use `cu132` instead of `cu134` and tell Claude.
- No separate CUDA Toolkit install is needed; the wheel includes the CUDA runtime.

Check:

```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Expected shape of output: `2.15.0+cu134 True 13.4`.

Further packages (torchvision, ultralytics, numpy, etc.) are added in the stages that need them.

## Stage 1 check

```cmd
python -m training.smoking.cnn.inspect_model
python -m training.smoking.cnn.inspect_model --memory-probe
python -m training.smoking.cnn.inspect_model --memory-probe --amp
```

See `training/smoking/cnn/README.md` for what each check means.
