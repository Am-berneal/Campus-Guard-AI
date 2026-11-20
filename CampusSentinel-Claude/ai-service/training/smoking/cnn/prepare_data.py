"""Check dataset_smoking and create a reproducible stratified train/validation split.

READ-ONLY with respect to the dataset: images are opened for decoding and hashing,
nothing is moved, renamed, edited or deleted. The split is written as a JSON list
of relative paths.

What it does
  1. Finds image files in smoking/, not_smoking/, testing/smoking/, testing/not_smoking/.
  2. Fully decodes every image (catches truncated/corrupt files, not only bad headers).
  3. SHA-256 hashes every file to find EXACT duplicates (byte-identical files).
  4. Builds the training pool, excluding:
       - unreadable images,
       - images identical to any TEST image          (would leak test data into training),
       - images present in BOTH classes              (contradictory labels),
       - extra copies of duplicates within one class (keep one, so a copy can't land in train AND val).
  5. Stratified 80/20 split per class with a fixed seed.
  6. Writes splits/split_seed<seed>.json (paths relative to the dataset root, '/' separators,
     so the same file works on Windows and in Colab).

The test images are never part of train/val. They are listed in the JSON for the
final evaluation only.

Run from the ai-service folder:
    python -m training.smoking.cnn.prepare_data --check-only
    python -m training.smoking.cnn.prepare_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

CLASSES = ("not_smoking", "smoking")  # label index: not_smoking=0, smoking=1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset_smoking"
DEFAULT_SPLIT_DIR = Path(__file__).resolve().parent / "splits"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_image(path: Path) -> tuple[bool, str]:
    """Return (ok, info). Fully decodes the image, the same way training will."""
    try:
        with Image.open(path) as img:
            img.load()
            w, h = img.size
            mode = img.mode
        if w < 16 or h < 16:
            return False, f"too small ({w}x{h})"
        return True, mode
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def scan_folder(data_root: Path, rel_folder: str, label: int) -> tuple[list[dict], list[dict], list[str]]:
    """Return (valid records, invalid records, ignored non-image files) for one class folder."""
    folder = data_root / rel_folder
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")

    valid, invalid, ignored = [], [], []
    files = sorted(p for p in folder.iterdir() if p.is_file())  # sorted → deterministic order
    for i, path in enumerate(files, 1):
        rel = path.relative_to(data_root).as_posix()
        if path.suffix.lower() not in IMAGE_EXTS:
            ignored.append(rel)
            continue
        ok, info = check_image(path)
        record = {"path": rel, "label": label}
        if ok:
            record["sha256"] = sha256_of(path)
            record["mode"] = info
            valid.append(record)
        else:
            record["error"] = info
            invalid.append(record)
        if i % 500 == 0:
            print(f"    {rel_folder}: {i}/{len(files)} files checked")
    return valid, invalid, ignored


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dataset_smoking and create a train/val split.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--check-only", action="store_true", help="report only; do not write a split file")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data_root = args.data_root.resolve()
    if not 0.0 < args.val_fraction < 0.5:
        print("ERROR: --val-fraction must be between 0 and 0.5")
        return 1
    print(f"Dataset root: {data_root}  (read-only)\n")

    # ---------------------------------------------------------------- 1–2. readability
    print("Step 1/3  Decoding every image ...")
    groups = {}
    invalid_all, ignored_all = [], []
    try:
        for label, cls in enumerate(CLASSES):
            for rel_folder, group in ((cls, f"pool/{cls}"), (f"testing/{cls}", f"test/{cls}")):
                valid, invalid, ignored = scan_folder(data_root, rel_folder, label)
                groups[group] = valid
                invalid_all += invalid
                ignored_all += ignored
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\n  {'Folder':<24}{'Valid':>8}")
    for group, records in groups.items():
        print(f"  {group:<24}{len(records):>8}")
    print(f"  {'Unreadable images':<24}{len(invalid_all):>8}")
    print(f"  {'Ignored non-image files':<24}{len(ignored_all):>8}")
    for r in invalid_all:
        print(f"    UNREADABLE {r['path']}  ({r['error']})")
    for rel in ignored_all[:20]:
        print(f"    IGNORED    {rel}")
    modes = Counter(r["mode"] for recs in groups.values() for r in recs)
    print(f"  Colour modes: {dict(modes)}  (all converted to RGB when loaded)")

    # ---------------------------------------------------------------- 3. exact duplicates
    print("\nStep 2/3  Exact-duplicate check (SHA-256) ...")
    test_records = groups["test/not_smoking"] + groups["test/smoking"]
    pool_records = groups["pool/not_smoking"] + groups["pool/smoking"]
    test_hashes = {r["sha256"] for r in test_records}

    test_dup_groups = defaultdict(list)
    for r in test_records:
        test_dup_groups[r["sha256"]].append(r["path"])
    test_internal = [paths for paths in test_dup_groups.values() if len(paths) > 1]

    by_hash = defaultdict(list)
    for r in pool_records:
        by_hash[r["sha256"]].append(r)

    excluded = []
    pool_kept = {0: [], 1: []}
    n_test_overlap = n_conflict = n_within = 0
    for digest, records in by_hash.items():
        records = sorted(records, key=lambda r: r["path"])
        if digest in test_hashes:
            n_test_overlap += len(records)
            excluded += [{"path": r["path"], "reason": "identical to a test image"} for r in records]
            continue
        if len({r["label"] for r in records}) > 1:
            n_conflict += len(records)
            excluded += [{"path": r["path"], "reason": "same image in both classes"} for r in records]
            continue
        keep, extra = records[0], records[1:]
        n_within += len(extra)
        excluded += [{"path": r["path"], "reason": f"duplicate of {keep['path']}"} for r in extra]
        pool_kept[keep["label"]].append(keep)

    print(f"  Training-pool images identical to a TEST image : {n_test_overlap:>5}  → excluded")
    print(f"  Images present in BOTH classes                 : {n_conflict:>5}  → excluded")
    print(f"  Extra copies within the same class             : {n_within:>5}  → excluded (one copy kept)")
    print(f"  Duplicate groups inside the test set           : {len(test_internal):>5}  → reported only (test set untouched)")
    for r in excluded[:15]:
        print(f"    EXCLUDED {r['path']}  ({r['reason']})")
    if len(excluded) > 15:
        print(f"    ... {len(excluded) - 15} more (full list in the split file)")

    # ---------------------------------------------------------------- 4. stratified split
    print(f"\nStep 3/3  Stratified split (val fraction {args.val_fraction}, seed {args.seed})")
    rng = random.Random(args.seed)
    train, val = [], []
    for label in (0, 1):
        items = sorted(pool_kept[label], key=lambda r: r["path"])  # fixed order before shuffling
        rng.shuffle(items)
        n_val = round(len(items) * args.val_fraction)
        val += items[:n_val]
        train += items[n_val:]

    def counts(records):
        c = Counter(r["label"] for r in records)
        return c[0], c[1]

    print(f"\n  {'Split':<8}{'not_smoking':>13}{'smoking':>10}{'total':>8}{'% smoking':>11}")
    for name, recs in (("train", train), ("val", val), ("test", test_records)):
        n0, n1 = counts(recs)
        total = n0 + n1
        pct = 100 * n1 / total if total else 0.0
        print(f"  {name:<8}{n0:>13}{n1:>10}{total:>8}{pct:>10.1f}%")

    if args.check_only:
        print("\n--check-only: no split file written.")
        return 0

    def slim(records):
        return [{"path": r["path"], "label": r["label"]} for r in sorted(records, key=lambda r: r["path"])]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"split_seed{args.seed}.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root_name": data_root.name,
        "classes": list(CLASSES),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "counts": {k: dict(zip(CLASSES, counts(v))) for k, v in (("train", train), ("val", val), ("test", test_records))},
        "train": slim(train),
        "val": slim(val),
        "test": slim(test_records),
        "excluded": excluded,
        "unreadable": [{"path": r["path"], "error": r["error"]} for r in invalid_all],
    }
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nSplit written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
