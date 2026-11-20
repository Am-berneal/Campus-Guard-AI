"""Evaluate a CNN-LSTM checkpoint on the RWF-2000 VALIDATION split.

RWF-2000 has no test split. These are VALIDATION results, and because the same split
selected the best epoch, they are optimistic. Never report them as test performance.

Outputs: ai-service/runs/fighting/eval_<time>/
    val_metrics.json, val_classification_report.txt, val_confusion_matrix.png, val_predictions.csv

Run from the ai-service folder:
    python -m training.fighting.evaluate --data-root "D:\\CampusSentinel-Datasets\\fighting_processed"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.fighting.checkpoint import load_checkpoint
from training.fighting.config import CHECKPOINT_DIR, RUNS_DIR, resolve_data_root
from training.fighting.dataset import FightingVideoDataset
from training.fighting.metrics import binary_metrics, classification_report, confusion_matrix, plot_confusion_matrix


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate on the VALIDATION split (RWF-2000 has no test split).")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    ap.add_argument("--data-root", help="folder with manifest.csv (or set CS_FIGHTING_DATA)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None, help="default: value stored in the checkpoint")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        data_root = resolve_data_root(args.data_root)
        device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
        model, meta = load_checkpoint(args.checkpoint, device)
        ds = FightingVideoDataset(data_root / "manifest.csv", "val")
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    threshold = args.threshold if args.threshold is not None else meta["train_config"].get("threshold", 0.5)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    probs, labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for clips, y in loader:
            logits = model(clips.to(device, non_blocking=True)).float()
            total_loss += loss_fn(logits, y.to(device).float().unsqueeze(1)).item()
            probs.append(torch.sigmoid(logits).squeeze(1).cpu())
            labels.append(y)
    p, t = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    m = binary_metrics(t, p, threshold)
    m["loss"] = total_loss / len(ds)

    out = args.out_dir or RUNS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "split": "validation",
        "note": "RWF-2000 provides no test split. These are VALIDATION results, optimistic because "
                "validation loss also selected the checkpoint. Not test performance.",
        "checkpoint": str(args.checkpoint), "best_epoch": meta.get("best_epoch", meta.get("epoch")),
        "clips": len(ds), "class_counts": ds.class_counts(), "threshold": threshold,
        "metrics": m, "confusion_matrix_rows_true_cols_pred": {"labels": ["NonFight", "Fight"],
                                                              "matrix": confusion_matrix(m)},
    }
    (out / "val_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = classification_report(m)
    (out / "val_classification_report.txt").write_text(f"VALIDATION classification report\n\n{report}\n", encoding="utf-8")
    plot_confusion_matrix(m, out / "val_confusion_matrix.png", f"VALIDATION confusion matrix (thr {threshold})")
    with (out / "val_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["npy_path", "label", "fight_probability", "predicted"])
        for (path, label), prob in zip(ds.items, p):
            w.writerow([path, label, round(float(prob), 5), int(prob >= threshold)])

    print("VALIDATION RESULTS (RWF-2000 val split — not a test set)")
    print(f"  clips {len(ds)} {ds.class_counts()} | threshold {threshold}")
    auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
    print(f"  loss {m['loss']:.4f} | accuracy {m['accuracy']:.4f} | precision {m['precision']:.4f} | "
          f"recall {m['recall']:.4f} | F1 {m['f1']:.4f} | ROC-AUC {auc}")
    print("\n" + report)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
