"""Train SmokingCNN from scratch on the train split, validating every epoch.

The TEST split is never loaded here.

Outputs
  ai-service/models/smoking/cnn/smoking_cnn_best.pt     best model (lowest val loss) — for inference
  ai-service/models/smoking/cnn/smoking_cnn_best.json   human-readable summary of that checkpoint
  ai-service/runs/smoking_cnn/last.pt                   full training state, for --resume
  ai-service/runs/smoking_cnn/history.csv               one row per epoch
  ai-service/runs/smoking_cnn/loss.png, accuracy.png    training curves
  ai-service/runs/smoking_cnn/train.log                 copy of the console output

Run from the ai-service folder:
    python -m training.smoking.cnn.train
    python -m training.smoking.cnn.train --resume runs\\smoking_cnn\\last.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.common.reproducibility import get_device, set_seed
from training.smoking.cnn.checkpoint import build_metadata, save_best
from training.smoking.cnn.dataset import build_dataloaders
from training.smoking.cnn.metrics import binary_metrics
from training.smoking.cnn.model import SmokingCNN
from training.smoking.cnn.prepare_data import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR

AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = AI_SERVICE_ROOT / "models" / "smoking" / "cnn"
DEFAULT_RUN_DIR = AI_SERVICE_ROOT / "runs" / "smoking_cnn"
BEST_NAME = "smoking_cnn_best.pt"

HISTORY_FIELDS = [
    "epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
    "val_precision", "val_recall", "val_f1", "val_roc_auc", "val_tp", "val_tn", "val_fp", "val_fn",
    "epoch_seconds", "best",
]


# --------------------------------------------------------------------------- logging
class Tee:
    """Print to console and append to a log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- epochs
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, use_amp) -> tuple[float, float]:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became {loss.item()} — training diverged. Try a lower --lr.")

        scaler.scale(loss).backward()   # AMP: scale loss to avoid float16 gradient underflow
        scaler.step(optimizer)
        scaler.update()

        n = labels.size(0)
        total_loss += loss.item() * n
        correct += ((logits.detach() > 0).float() == labels).sum().item()  # logit > 0 ⇔ prob > 0.5
        seen += n
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp) -> tuple[float, dict]:
    model.eval()
    total_loss, seen = 0.0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        seen += labels.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_probs).numpy().ravel()
    labels_np = torch.cat(all_labels).numpy().ravel()
    return total_loss / seen, binary_metrics(labels_np, probs)


# --------------------------------------------------------------------------- outputs
def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def plot_history(run_dir: Path, history: list[dict], best_epoch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no window needed; write PNG files
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    for key, ylabel, fname in (("loss", "BCE loss", "loss.png"), ("acc", "Accuracy", "accuracy.png")):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epochs, [h[f"train_{key}"] for h in history], marker="o", label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], marker="o", label="validation")
        if best_epoch:
            ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"SmokingCNN — {ylabel}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / fname, dpi=120)
        plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description="Train SmokingCNN (from scratch).")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_DIR / "split_seed42.json")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    p.add_argument("--min-delta", type=float, default=1e-4, help="min val-loss decrease that counts as improvement")
    p.add_argument("--lr-patience", type=int, default=2, help="ReduceLROnPlateau patience; 0 disables it")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--resume", type=Path, default=None, help="path to runs/smoking_cnn/last.pt")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(args.run_dir / "train.log")
    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp

    log("=" * 78)
    log(f"SmokingCNN training — started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch {torch.__version__} | device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " — WARNING: no GPU, this will be slow"))
    log(f"epochs≤{args.epochs} batch={args.batch_size} lr={args.lr} AMP={'on' if use_amp else 'off'} "
        f"early-stop patience={args.patience} seed={args.seed} workers={args.num_workers}")

    # ---- data (train + val only; test is NOT built) ----
    try:
        loaders = build_dataloaders(
            args.data_root, args.split_file, args.batch_size, args.num_workers, args.seed, include_test=False
        )
    except FileNotFoundError as exc:
        log(f"ERROR: {exc}")
        return 1
    train_loader, val_loader = loaders["train"], loaders["val"]
    log(f"train {len(train_loader.dataset)} images ({len(train_loader)} batches) | "
        f"val {len(val_loader.dataset)} images | test: not loaded")

    # ---- model, loss, optimiser ----
    model = SmokingCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)
        if args.lr_patience > 0 else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_config = {
        "epochs_max": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss", "amp": use_amp, "seed": args.seed, "patience": args.patience,
        "lr_scheduler": f"ReduceLROnPlateau(factor=0.5, patience={args.lr_patience})" if scheduler else "none",
        "augmentation": "RandomHorizontalFlip(0.5), RandomRotation(10)",
        "split_file": args.split_file.name, "split_md5": file_md5(args.split_file),
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
    }

    history: list[dict] = []
    start_epoch, best_loss, best_epoch, epochs_no_improve = 1, math.inf, 0, 0
    best_metrics: dict = {}

    # ---- resume ----
    if args.resume:
        if not args.resume.is_file():
            log(f"ERROR: resume checkpoint not found: {args.resume}")
            return 1
        state = torch.load(args.resume, map_location="cpu", weights_only=False)  # our own file
        if state["train_config"]["split_md5"] != train_config["split_md5"]:
            log("ERROR: split file changed since this checkpoint was saved — refusing to resume.")
            return 1
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        start_epoch = state["epoch"] + 1
        best_loss, best_epoch = state["best_loss"], state["best_epoch"]
        best_metrics, epochs_no_improve = state["best_metrics"], state["epochs_no_improve"]
        log(f"Resumed from {args.resume} at epoch {start_epoch} (best so far: epoch {best_epoch}, val loss {best_loss:.4f})")
        if epochs_no_improve >= args.patience:
            log("Early stopping had already triggered in that run; nothing to do.")
            return 0

    best_path = args.model_dir / BEST_NAME
    log(f"Best checkpoint → {best_path}")
    log("-" * 78)
    log(f"{'ep':>3} {'lr':>8} {'tr_loss':>8} {'tr_acc':>7} {'va_loss':>8} {'va_acc':>7} "
        f"{'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6} {'time':>8}")

    t_start = time.perf_counter()
    stop_reason = f"reached max epochs ({args.epochs})"
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.perf_counter()
            train_loader.generator.manual_seed(args.seed + epoch)  # shuffle order depends only on epoch
            lr_now = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, use_amp)
            val_loss, vm = evaluate(model, val_loader, loss_fn, device, use_amp)
            if scheduler:
                scheduler.step(val_loss)

            improved = val_loss < best_loss - args.min_delta
            if improved:
                best_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
                best_metrics = {"val_loss": val_loss, **vm}
                save_best(best_path, model, build_metadata(best_epoch, best_metrics, train_config))
                (args.model_dir / BEST_NAME.replace(".pt", ".json")).write_text(
                    json.dumps(build_metadata(best_epoch, best_metrics, train_config), indent=2), encoding="utf-8"
                )
            else:
                epochs_no_improve += 1

            secs = time.perf_counter() - t0
            auc = vm["roc_auc"] if vm["roc_auc"] is not None else float("nan")
            log(f"{epoch:>3} {lr_now:>8.1e} {train_loss:>8.4f} {train_acc:>7.4f} {val_loss:>8.4f} {vm['accuracy']:>7.4f} "
                f"{vm['precision']:>6.3f} {vm['recall']:>6.3f} {vm['f1']:>6.3f} {auc:>6.3f} "
                f"{fmt_duration(secs):>8}{'  * best' if improved else ''}")

            history.append({
                "epoch": epoch, "lr": lr_now, "train_loss": round(train_loss, 6), "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6), "val_acc": round(vm["accuracy"], 6),
                "val_precision": round(vm["precision"], 6), "val_recall": round(vm["recall"], 6),
                "val_f1": round(vm["f1"], 6), "val_roc_auc": None if vm["roc_auc"] is None else round(vm["roc_auc"], 6),
                "val_tp": vm["tp"], "val_tn": vm["tn"], "val_fp": vm["fp"], "val_fn": vm["fn"],
                "epoch_seconds": round(secs, 1), "best": improved,
            })
            write_history(args.run_dir / "history.csv", history)

            # Full state for resuming (optimizer, scaler, scheduler, counters).
            torch.save({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict(),
                "history": history, "best_loss": best_loss, "best_epoch": best_epoch,
                "best_metrics": best_metrics, "epochs_no_improve": epochs_no_improve, "train_config": train_config,
            }, args.run_dir / "last.pt")

            if epochs_no_improve >= args.patience:
                stop_reason = f"early stopping: no val-loss improvement for {args.patience} epochs"
                break

    except torch.cuda.OutOfMemoryError:
        log("\nERROR: GPU ran out of memory.")
        log("  • Close other GPU apps (browsers, games, video) and run again, or")
        log(f"  • use a smaller batch: --batch-size {max(args.batch_size // 2, 1)}")
        log("  • Training so far is kept in runs/smoking_cnn/last.pt (resume with --resume).")
        return 2
    except KeyboardInterrupt:
        log("\nInterrupted by user (Ctrl+C). Resume with: --resume runs\\smoking_cnn\\last.pt")
        stop_reason = "interrupted"

    total = time.perf_counter() - t_start
    if history:
        plot_history(args.run_dir, history, best_epoch)

    log("-" * 78)
    log(f"Stopped: {stop_reason}")
    log(f"Total training time (this session): {fmt_duration(total)}")
    if best_epoch:
        bm = best_metrics
        log(f"Best epoch {best_epoch}: val loss {bm['val_loss']:.4f} | acc {bm['accuracy']:.4f} | "
            f"precision {bm['precision']:.4f} | recall {bm['recall']:.4f} | F1 {bm['f1']:.4f} | "
            f"ROC-AUC {bm['roc_auc'] if bm['roc_auc'] is not None else float('nan'):.4f}")
        log(f"Val confusion (threshold 0.5): TP {bm['tp']}  FN {bm['fn']}  FP {bm['fp']}  TN {bm['tn']}")
        log(f"Saved: {best_path}")
    log(f"History: {args.run_dir / 'history.csv'} | plots: loss.png, accuracy.png")
    log("Test set was NOT used.")
    return 0


if __name__ == "__main__":  # required on Windows for DataLoader workers
    sys.exit(main())
