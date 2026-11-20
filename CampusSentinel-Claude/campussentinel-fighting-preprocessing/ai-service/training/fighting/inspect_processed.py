"""Check preprocessed clips and save frame previews.

Checks: manifest present; every 'ok' row has its .npy; no stray .npy outside the manifest;
sampled clips have the right dtype/shape/value range; labels match class folders.
Previews (4x4 grid of the 16 frames) go to <processed root>/_previews/.

Run from the ai-service folder:
    python -m training.fighting.inspect_processed
    python -m training.fighting.inspect_processed --samples 3
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from training.fighting.dataset import LABELS
from training.fighting.prepare_data import DEFAULT_OUTPUT


def save_grid(clip: np.ndarray, dst: Path, cols: int = 4, tile: int = 160) -> None:
    rows = int(np.ceil(len(clip) / cols))
    grid = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, frame in enumerate(clip):
        r, c = divmod(i, cols)
        small = cv2.resize(frame, (tile, tile), interpolation=cv2.INTER_AREA)
        grid[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = small
        cv2.putText(grid, str(i), (c * tile + 4, r * tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))  # stored RGB → OpenCV writes BGR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--samples", type=int, default=2, help="clips to check + preview per split/class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = args.processed
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        print(f"ERROR: {manifest} not found")
        return 1
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok_rows = [r for r in rows if r["status"] == "ok"]
    problems = 0

    print(f"Manifest: {len(rows)} rows ({len(ok_rows)} ok, {len(rows) - len(ok_rows)} failed)")
    counts = Counter((r["split"], r["class_name"]) for r in ok_rows)
    for split in ("train", "val"):
        for cls in ("Fight", "NonFight"):
            print(f"  {split}/{cls:<9} {counts[(split, cls)]:>5} ok clips")
    for r in rows:
        if r["status"] != "ok":
            print(f"  FAILED {r['source_path']}: {r['error']}")

    missing = [r["npy_path"] for r in ok_rows if not (root / r["npy_path"]).is_file()]
    listed = {r["npy_path"] for r in rows}
    on_disk = {p.relative_to(root).as_posix() for p in root.glob("*/*/*.npy")}
    stray = sorted(on_disk - listed)
    bad_labels = [r["npy_path"] for r in rows if LABELS.get(r["class_name"]) != int(r["label"])
                  or r["npy_path"].split("/")[1] != r["class_name"]]
    print(f"\nMissing .npy for ok rows: {len(missing)} | .npy not in manifest: {len(stray)} | "
          f"label/folder mismatches: {len(bad_labels)}")
    for m in missing[:10]:
        print(f"  MISSING {m}")
    for s in stray[:10]:
        print(f"  STRAY {s}")
    problems += len(missing) + len(bad_labels)

    rng = random.Random(args.seed)
    print("\nSampled clips")
    for split in ("train", "val"):
        for cls in ("Fight", "NonFight"):
            group = [r for r in ok_rows if r["split"] == split and r["class_name"] == cls
                     and (root / r["npy_path"]).is_file()]
            for r in rng.sample(group, min(args.samples, len(group))):
                arr = np.load(root / r["npy_path"])
                expected = tuple(int(v) for v in r["output_shape"].split("x"))
                ok = arr.dtype == np.uint8 and arr.shape == expected
                problems += not ok
                print(f"  [{'OK' if ok else 'BAD'}] {r['npy_path']}  {arr.dtype} {arr.shape} "
                      f"min {arr.min()} max {arr.max()} mean {arr.mean():.1f} | label {r['label']} | "
                      f"src {r['width']}x{r['height']}, {r['frame_count']} frames @ {r['fps']} fps | "
                      f"indices {r['frame_indices']}")
                save_grid(arr, root / "_previews" / split / cls / (Path(r["npy_path"]).stem + ".jpg"))

    size_mb = sum((root / r["npy_path"]).stat().st_size for r in ok_rows if (root / r["npy_path"]).is_file()) / 1024**2
    print(f"\nDisk usage of clips: {size_mb:.1f} MB")
    print(f"Previews: {root / '_previews'}")
    print(f"Result: {'ALL CHECKS PASSED' if problems == 0 else f'{problems} PROBLEM(S) FOUND'}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
