"""Train the CNN-LSTM fight classifier on the RWF-2000 train split, validating every epoch.

RWF-2000 has only train and val splits. All numbers produced here are TRAIN or
VALIDATION results; there is no test set. Because the best epoch, early stopping and
the LR schedule all use validation loss, validation metrics are optimistic.

Run from the ai-service folder:
    python -m training.fighting.train --data-root "D:\\CampusSentinel-Datasets\\fighting_processed"
    python -m training.fighting.train --data-root ... --amp                (mixed precision on CUDA)
    python -m training.fighting.train --data-root ... --resume             (continue from last.pt)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.fighting.checkpoint import save_checkpoint
from training.fighting.config import CHECKPOINT_DIR, RUNS_DIR, ModelConfig, TrainConfig, resolve_data_root
from training.fighting.dataset import FightingVideoDataset
from training.fighting.metrics import binary_metrics, classification_report, plot_confusion_matrix
from training.fighting.model import CNNLSTM, count_parameters

LOG = logging.getLogger("fighting.train")
HISTORY_FIELDS = ["epoch", "lr", "seconds"] + [f"{s}_{m}" for s in ("train", "val")
                                              for m in ("loss", "accuracy", "precision", "recall", "f1", "roc_auc")]


class ClipHorizontalFlip:
    """Flip ALL frames of a clip left-right with probability p (keeps motion consistent).
    A module-level class (not a lambda) so it also works with DataLoader workers on Windows."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:   # (T, C, H, W)
        return torch.flip(video, dims=[-1]) if torch.rand(1).item() < self.p else video


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, threshold, optimizer=None, scaler=None, amp=False,
              max_batches: int = 0) -> tuple[float, dict]:
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates (no grad)."""
    training = optimizer is not None
    model.train(training)
    total_loss, n_seen, probs, labels = 0.0, 0, [], []
    for i, (clips, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        clips = clips.to(device, non_blocking=True)
        target = y.to(device, non_blocking=True).float().unsqueeze(1)    # (B,) long → (B, 1) float
        with torch.set_grad_enabled(training), \
                torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(clips)
            loss = loss_fn(logits.float(), target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became {loss.item()} — training diverged; try a lower --lr")
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * y.size(0)
        n_seen += y.size(0)
        probs.append(torch.sigmoid(logits.detach().float()).squeeze(1).cpu())   # sigmoid → probabilities
        labels.append(y.cpu())
        del clips, target, logits, loss
    if n_seen == 0:
        raise RuntimeError("no batches were processed")
    p = torch.cat(probs).numpy()
    t = torch.cat(labels).numpy()
    return total_loss / n_seen, binary_metrics(t, p, threshold)


def plot_history(history: list[dict], run_dir: Path, best_epoch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    for key, label, fname in (("loss", "BCE loss", "loss.png"), ("accuracy", "Accuracy", "accuracy.png")):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(ep, [h[f"train_{key}"] for h in history], marker="o", label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], marker="o", label="validation")
        if best_epoch:
            ax.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"CNN-LSTM fighting — {label} (train vs validation)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / fname, dpi=120)
        plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    d, m = TrainConfig(), ModelConfig()
    ap = argparse.ArgumentParser(description="Train the CNN-LSTM fight classifier (train + validation only).")
    ap.add_argument("--data-root", help="folder with manifest.csv (or set CS_FIGHTING_DATA)")
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--backbone-lr", type=float, default=d.backbone_lr)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--lr-patience", type=int, default=d.lr_patience)
    ap.add_argument("--threshold", type=float, default=d.threshold)
    ap.add_argument("--workers", type=int, default=d.workers)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--no-hflip", action="store_true", help="disable training-time horizontal flip")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--hidden-size", type=int, default=m.hidden_size)
    ap.add_argument("--num-layers", type=int, default=m.num_layers)
    ap.add_argument("--lstm-dropout", type=float, default=m.lstm_dropout)
    ap.add_argument("--head-dropout", type=float, default=m.head_dropout)
    ap.add_argument("--train-layer4", action="store_true", help="also fine-tune ResNet18 layer4")
    ap.add_argument("--no-pretrained", action="store_true", help="random ResNet18 (offline smoke tests only)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--run-dir", type=Path, default=None, help="default: runs/fighting/train_<time>")
    ap.add_argument("--resume", action="store_true", help="continue from <checkpoint-dir>/last.pt")
    ap.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke tests only)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    try:
        return _main(argv)
    finally:
        # Release train.log (Windows keeps open files locked).
        for h in list(logging.getLogger().handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                logging.getLogger().removeHandler(h)


def _main(argv=None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_dir = args.run_dir or RUNS_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(run_dir / "train.log", encoding="utf-8")], force=True)

    try:
        data_root = resolve_data_root(args.data_root)
    except (ValueError, FileNotFoundError) as exc:
        LOG.error("%s", exc)
        return 1
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        LOG.warning("CUDA requested but not available; using CPU")
    amp = bool(args.amp and device.type == "cuda")
    if args.amp and not amp:
        LOG.warning("--amp ignored: CUDA not available")
    set_seed(args.seed)

    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, backbone_lr=args.backbone_lr,
                       weight_decay=args.weight_decay, patience=args.patience, lr_patience=args.lr_patience,
                       threshold=args.threshold, workers=args.workers, amp=amp, hflip=not args.no_hflip,
                       seed=args.seed, extra={"data_root": str(data_root), "max_batches": args.max_batches})
    mcfg = ModelConfig(hidden_size=args.hidden_size, num_layers=args.num_layers, lstm_dropout=args.lstm_dropout,
                       head_dropout=args.head_dropout, pretrained=not args.no_pretrained,
                       freeze_backbone=True, train_layer4=args.train_layer4)

    manifest = data_root / "manifest.csv"
    try:
        train_ds = FightingVideoDataset(manifest, "train", transform=ClipHorizontalFlip() if tcfg.hflip else None)
        val_ds = FightingVideoDataset(manifest, "val")
    except (ValueError, FileNotFoundError) as exc:
        LOG.error("dataset error: %s", exc)
        return 1
    pin = device.type == "cuda"
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.workers,
                              pin_memory=pin, generator=gen, persistent_workers=tcfg.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=tcfg.workers,
                            pin_memory=pin, persistent_workers=tcfg.workers > 0)

    try:
        model = CNNLSTM(mcfg).to(device)
    except Exception as exc:  # most likely: ImageNet weights could not be downloaded
        LOG.error("could not build the model (ResNet18 ImageNet weights need internet on first use): %s", exc)
        return 1
    total, trainable = count_parameters(model)
    groups = [{"params": model.temporal_parameters(), "lr": tcfg.lr}]
    if model.backbone_parameters():
        groups.append({"params": model.backbone_parameters(), "lr": tcfg.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=tcfg.lr_patience)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss()

    history, start_epoch, best_loss, best_epoch, no_improve = [], 1, float("inf"), 0, 0
    last_path, best_path = args.checkpoint_dir / "last.pt", args.checkpoint_dir / "best.pt"
    if args.resume:
        if not last_path.is_file():
            LOG.error("cannot resume: %s not found", last_path)
            return 1
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if ModelConfig.from_dict(state["model_config"]).hidden_size != mcfg.hidden_size:
            LOG.error("model settings differ from the checkpoint; resume with the same --hidden-size/--num-layers")
            return 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        history, start_epoch = state["history"], state["epoch"] + 1
        best_loss, best_epoch, no_improve = state["best_val_loss"], state["best_epoch"], state["epochs_no_improve"]
        LOG.info("Resumed from %s at epoch %d (best so far: epoch %d, val loss %.4f)", last_path, start_epoch,
                 best_epoch, best_loss)
        if no_improve >= tcfg.patience:
            LOG.info("Early stopping had already triggered; nothing to do.")
            return 0

    LOG.info("=" * 74)
    LOG.info("CNN-LSTM fighting training | device %s%s | AMP %s", device,
             f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "", "on" if amp else "off")
    LOG.info("train %d clips %s | val %d clips %s | NO test split (RWF-2000 has train/val only)",
             len(train_ds), train_ds.class_counts(), len(val_ds), val_ds.class_counts())
    LOG.info("parameters: %s total, %s trainable | backbone %s", f"{total:,}", f"{trainable:,}",
             "frozen" if model.fully_frozen else ("frozen except layer4" if mcfg.train_layer4 else "trainable"))
    LOG.info("config: %s", json.dumps({**tcfg.to_dict(), "model": mcfg.to_dict()}))
    LOG.info("run folder: %s | checkpoints: %s", run_dir, args.checkpoint_dir)
    if args.max_batches:
        LOG.warning("--max-batches %d: SMOKE TEST ONLY, metrics are not meaningful", args.max_batches)

    t_start, stop = time.perf_counter(), f"reached max epochs ({tcfg.epochs})"
    best_val_metrics = {}
    try:
        for epoch in range(start_epoch, tcfg.epochs + 1):
            t0 = time.perf_counter()
            gen.manual_seed(args.seed + epoch)            # shuffle order depends only on the epoch → resumable
            lr_now = optimizer.param_groups[0]["lr"]
            tr_loss, tr = run_epoch(model, train_loader, loss_fn, device, tcfg.threshold, optimizer, scaler, amp,
                                    args.max_batches)
            va_loss, va = run_epoch(model, val_loader, loss_fn, device, tcfg.threshold, amp=amp,
                                    max_batches=args.max_batches)
            scheduler.step(va_loss)
            secs = time.perf_counter() - t0

            row = {"epoch": epoch, "lr": lr_now, "seconds": round(secs, 1)}
            for split, loss, mm in (("train", tr_loss, tr), ("val", va_loss, va)):
                row[f"{split}_loss"] = round(loss, 6)
                for k in ("accuracy", "precision", "recall", "f1"):
                    row[f"{split}_{k}"] = round(mm[k], 6)
                row[f"{split}_roc_auc"] = None if mm["roc_auc"] is None else round(mm["roc_auc"], 6)
            history.append(row)

            improved = va_loss < best_loss - 1e-4
            if improved:
                best_loss, best_epoch, no_improve, best_val_metrics = va_loss, epoch, 0, {"loss": va_loss, **va}
                save_checkpoint(best_path, model, optimizer, epoch, best_loss, tcfg.to_dict(), best_val_metrics,
                                extra={"best_epoch": epoch})
                (args.checkpoint_dir / "best.json").write_text(json.dumps({
                    "note": "VALIDATION metrics (RWF-2000 has no test split); selected by lowest validation loss",
                    "best_epoch": epoch, "best_val_loss": round(va_loss, 6), "val_metrics": best_val_metrics,
                    "model_config": mcfg.to_dict(), "train_config": tcfg.to_dict(),
                    "class_mapping": {"0": "NonFight", "1": "Fight"}, "run_dir": str(run_dir)}, indent=2),
                    encoding="utf-8")
                plot_confusion_matrix(va, run_dir / "val_confusion_matrix_best.png",
                                      f"VALIDATION confusion matrix (best epoch {epoch}, thr {tcfg.threshold})")
                (run_dir / "val_classification_report_best.txt").write_text(
                    f"VALIDATION classification report — best epoch {epoch}\n\n{classification_report(va)}\n",
                    encoding="utf-8")
            else:
                no_improve += 1

            save_checkpoint(last_path, model, optimizer, epoch, best_loss, tcfg.to_dict(), va, extra={
                "best_epoch": best_epoch, "epochs_no_improve": no_improve, "history": history,
                "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict()})
            with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                w.writeheader()
                w.writerows(history)

            auc = va["roc_auc"] if va["roc_auc"] is not None else float("nan")
            LOG.info("ep %2d | lr %.1e | train loss %.4f acc %.3f | val loss %.4f acc %.3f P %.3f R %.3f F1 %.3f "
                     "AUC %.3f | %.0fs%s", epoch, lr_now, tr_loss, tr["accuracy"], va_loss, va["accuracy"],
                     va["precision"], va["recall"], va["f1"], auc, secs, "  * best" if improved else "")
            if no_improve >= tcfg.patience:
                stop = f"early stopping: no validation-loss improvement for {tcfg.patience} epochs"
                break
    except torch.cuda.OutOfMemoryError:
        LOG.error("GPU out of memory. Try: --batch-size %d, add --amp, or close other GPU apps. "
                  "Progress so far is in %s (use --resume).", max(1, tcfg.batch_size // 2), last_path)
        return 2
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Continue with --resume.")
        stop = "interrupted"

    if history:
        plot_history(history, run_dir, best_epoch)
    total_s = time.perf_counter() - t_start
    LOG.info("-" * 74)
    LOG.info("Stopped: %s | time this session: %dm %02ds", stop, total_s // 60, total_s % 60)
    if best_epoch and best_val_metrics:
        b = best_val_metrics
        LOG.info("BEST (VALIDATION) epoch %d: loss %.4f acc %.4f precision %.4f recall %.4f F1 %.4f ROC-AUC %s",
                 best_epoch, b["loss"], b["accuracy"], b["precision"], b["recall"], b["f1"],
                 "n/a" if b["roc_auc"] is None else f"{b['roc_auc']:.4f}")
    LOG.info("Checkpoints: %s | run outputs: %s", args.checkpoint_dir, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
