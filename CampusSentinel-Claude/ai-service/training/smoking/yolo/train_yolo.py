"""Fine-tune a pretrained YOLO26n on the processed cigarette dataset (train + val only).

Transfer learning: start from Ultralytics' COCO-pretrained yolo26n.pt and retrain it for
one class (0 = cigarette). The base weights are NOT ours; the fine-tuned result is.

The test split is never loaded: Ultralytics trains on `train` and validates on `val`
only. Model selection (best.pt) uses validation fitness (= val mAP@50-95 in 8.4.212).

Outputs
  ai-service/models/smoking/yolo/base/yolo26n.pt                 pretrained base (downloaded once)
  ai-service/runs/smoking_yolo/<run>/                            Ultralytics run: results.csv, plots,
                                                                 weights/best.pt, weights/last.pt, args.yaml
  ai-service/runs/smoking_yolo/<run>_val/                        final validation on val (confusion matrix, curves)
  ai-service/models/smoking/yolo/finetuned/best.pt               copy of the selected checkpoint
  ai-service/models/smoking/yolo/finetuned/best.json             provenance, config, validation metrics

Run from the ai-service folder:
    python -m training.smoking.yolo.train_yolo
    python -m training.smoking.yolo.train_yolo --batch 16                 (only if batch 8 used < ~3.5 GB)
    python -m training.smoking.yolo.train_yolo --resume                   (continue the last interrupted run)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DATA_YAML = AI_SERVICE_ROOT / "datasets" / "smoking_yolo" / "data.yaml"
BASE_WEIGHTS = AI_SERVICE_ROOT / "models" / "smoking" / "yolo" / "base" / "yolo26n.pt"
FINETUNED_DIR = AI_SERVICE_ROOT / "models" / "smoking" / "yolo" / "finetuned"
RUNS_DIR = AI_SERVICE_ROOT / "runs" / "smoking_yolo"
EXPECTED_NAMES = {0: "cigarette"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def fail(msg: str) -> int:
    print(f"ERROR: {msg}")
    return 1


def preflight(data_yaml: Path) -> dict:
    """Check the processed dataset WITHOUT touching the test images."""
    import yaml

    if not data_yaml.is_file():
        raise FileNotFoundError(f"{data_yaml} not found. Run: python -m training.smoking.yolo.prepare_dataset")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in (names or {}).items()}
    if names != EXPECTED_NAMES or int(cfg.get("nc", len(names))) != 1:
        raise ValueError(f"data.yaml must define exactly {EXPECTED_NAMES}, found nc={cfg.get('nc')} names={names}")
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    counts = {}
    for split in ("train", "val"):
        img_dir = root / cfg[split]
        lbl_dir = img_dir.parent / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            raise FileNotFoundError(f"{split} folder missing: {img_dir} / {lbl_dir}")
        n_img = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        n_lbl = sum(1 for p in lbl_dir.iterdir() if p.suffix == ".txt")
        if n_img == 0 or n_img != n_lbl:
            raise ValueError(f"{split}: {n_img} images vs {n_lbl} label files")
        counts[split] = n_img
    return {"root": root, "counts": counts}


def gpu_info() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return "0", f"{props.name}, {props.total_memory / 1024**3:.1f} GB, CUDA {torch.version.cuda}"
    return "cpu", "CPU only (training will be very slow)"


def is_unfinished(last_pt: Path) -> bool:
    """Ultralytics stores epoch = -1 in last.pt once a run has FINISHED. Resuming such a file
    would silently start a brand-new run with default settings, so we only resume unfinished runs."""
    import torch

    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)  # our own run file
    return ckpt.get("epoch", -1) != -1


def latest_last_pt() -> Path | None:
    """Most recent run whose last.pt is unfinished."""
    candidates = sorted(RUNS_DIR.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return next((p for p in candidates if is_unfinished(p)), None)


def is_oom(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO26n on the cigarette dataset (no test split).")
    ap.add_argument("--data", type=Path, default=DATA_YAML)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15, help="early stopping: epochs without val improvement")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers (use 0-2 if Windows errors)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="run folder name (default: yolo26n_<date>_<time>)")
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu to force CPU")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", nargs="?", const="latest", default=None,
                    help="resume: path to a run's weights/last.pt, or no value for the most recent run")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        return fail("Ultralytics is not installed. Run: pip install -r requirements.txt")

    device, gpu_desc = gpu_info()
    if args.device is not None:
        device = args.device
    print("=" * 78)
    print(f"Cigarette YOLO fine-tuning — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Ultralytics {ultralytics.__version__} | device {device} | {gpu_desc}")

    try:
        info = preflight(args.data)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return fail(str(exc))
    print(f"Dataset {args.data}")
    print(f"  train {info['counts']['train']} images | val {info['counts']['val']} images | test: NOT loaded")

    t0 = time.perf_counter()
    try:
        if args.resume:
            last = latest_last_pt() if args.resume == "latest" else Path(args.resume)
            if last is None or not last.is_file():
                return fail(f"no unfinished run to resume ({args.resume}); finished runs cannot be resumed. "
                            f"Runs are in {RUNS_DIR}")
            if not is_unfinished(last):
                return fail(f"{last} belongs to a run that already finished; nothing to resume.")
            print(f"Resuming from {last}")
            model = YOLO(str(last))
            model.train(resume=True)
        else:
            BASE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            print(f"Base weights (pretrained, Ultralytics COCO): {BASE_WEIGHTS}"
                  + ("" if BASE_WEIGHTS.is_file() else "  — downloading"))
            model = YOLO(str(BASE_WEIGHTS))  # downloads the official asset to this path if missing
            name = args.name or f"yolo26n_{datetime.now():%Y%m%d_%H%M%S}"
            print(f"Config: epochs≤{args.epochs} imgsz={args.imgsz} batch={args.batch} "
                  f"AMP={'off' if args.no_amp else 'on (if supported)'} patience={args.patience} seed={args.seed}")
            print(f"Run folder: {RUNS_DIR / name}")
            print("=" * 78)
            model.train(
                data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, amp=not args.no_amp, patience=args.patience, workers=args.workers,
                seed=args.seed, deterministic=True, project=str(RUNS_DIR), name=name, exist_ok=False,
                pretrained=True, plots=True, val=True, verbose=True,
            )
    except KeyboardInterrupt:
        print("\nInterrupted. Continue with: python -m training.smoking.yolo.train_yolo --resume")
        return 130
    except Exception as exc:  # noqa: BLE001 — report OOM clearly, re-raise anything else
        if is_oom(exc):
            print("\nERROR: GPU ran out of memory.")
            print("  • Close browsers/other GPU apps and retry, or")
            print(f"  • lower the batch: --batch {max(args.batch // 2, 1)}   (or --imgsz 512)")
            print("  • An interrupted run can be continued with --resume (same batch size).")
            return 2
        raise

    train_secs = time.perf_counter() - t0
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        return fail(f"training finished but {best} is missing")

    # Final check of the SELECTED checkpoint on the VALIDATION split only.
    print("\nValidating best.pt on the val split ...")
    best_model = YOLO(str(best))
    vm = best_model.val(data=str(args.data), split="val", imgsz=args.imgsz, batch=args.batch, device=device,
                        project=str(RUNS_DIR), name=f"{run_dir.name}_val", exist_ok=True, plots=True, verbose=False)
    metrics = {
        "precision": round(float(vm.box.mp), 4),
        "recall": round(float(vm.box.mr), 4),
        "mAP50": round(float(vm.box.map50), 4),
        "mAP50_95": round(float(vm.box.map), 4),
    }

    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, FINETUNED_DIR / "best.pt")
    tr_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(model.trainer.args).items()
               if k in ("epochs", "imgsz", "batch", "amp", "patience", "optimizer", "lr0", "seed", "workers", "device")}
    epochs_run = getattr(model.trainer, "epoch", None)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "YOLO26n detection",
        "origin": "fine-tuned by us from Ultralytics yolo26n.pt (COCO-pretrained); base weights are not ours",
        "base_weights": str(BASE_WEIGHTS.relative_to(AI_SERVICE_ROOT)).replace("\\", "/"),
        "classes": {"0": "cigarette"},
        "dataset": str(args.data.relative_to(AI_SERVICE_ROOT) if args.data.is_relative_to(AI_SERVICE_ROOT) else args.data).replace("\\", "/"),
        "dataset_counts": info["counts"],
        "selection": "Ultralytics best.pt = epoch with highest validation fitness (= val mAP50-95 in Ultralytics 8.4.212)",
        "train_args": tr_args,
        "epochs_completed": None if epochs_run is None else int(epochs_run) + 1,
        "training_seconds_this_session": round(train_secs, 1),
        "val_metrics": metrics,
        "run_dir": str(run_dir.relative_to(AI_SERVICE_ROOT)).replace("\\", "/"),
        "ultralytics_version": ultralytics.__version__,
        "test_split_used": False,
    }
    (FINETUNED_DIR / "best.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    m, s = divmod(int(train_secs), 60)
    print("\n" + "=" * 78)
    print(f"Training time this session: {m // 60}h {m % 60:02d}m {s:02d}s | epochs completed: {meta['epochs_completed']}")
    print(f"VALIDATION (val split) — precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f} | "
          f"mAP@50 {metrics['mAP50']:.4f} | mAP@50-95 {metrics['mAP50_95']:.4f}")
    print(f"Fine-tuned weights: {FINETUNED_DIR / 'best.pt'}")
    print(f"Run outputs:        {run_dir}  (results.csv, results.png, confusion matrices, PR curves)")
    print("Test split was NOT used.")
    return 0


if __name__ == "__main__":  # required on Windows for dataloader worker processes
    sys.exit(main())
