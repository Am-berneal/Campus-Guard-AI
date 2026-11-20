"""Validate the cigarette YOLO dataset and build a processed train/val/test copy.

The ORIGINAL dataset is only read: nothing in it is created, modified, renamed, moved or deleted.

Steps
  1. Show the source data.yaml and any Roboflow README (source, licence).
  2. Validate every image/label pair in train/ and test/:
       - image decodes fully (Pillow)
       - label file exists, has the same file stem, no two images share a stem
       - each line: `class x_center y_center width height` (5 numbers, YOLO detection format)
       - class ID must be 0 (cigarette)
       - coordinates normalised to 0..1, width/height > 0, box inside the image (small tolerance)
     Empty label files are valid "background" images (no cigarette) and are kept.
  3. Group train images by SOURCE image. Roboflow exports name augmented copies
     `<original>_jpg.rf.<hash>.jpg`; copies of one photo must stay in the same split,
     otherwise validation contains near-identical copies of training images.
  4. Leakage check: train images that are byte-identical to a test image, or come from the
     same Roboflow source name, are excluded from train/val and reported.
  5. Deterministic group split of the valid train images: 80% train / 20% val, seed 42.
  6. Copy (not move) valid files into the processed folder, write data.yaml + report JSON.

Run from the ai-service folder:
    python -m training.smoking.yolo.prepare_dataset --dry-run      (validate and report only)
    python -m training.smoking.yolo.prepare_dataset                (validate + create processed copy)
    python -m training.smoking.yolo.prepare_dataset --overwrite    (rebuild an existing processed copy)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[4]
AI_SERVICE_ROOT = PROJECT_ROOT / "ai-service"
DEFAULT_SOURCE = PROJECT_ROOT / "Cigarettes Detection.v13i.yolov8"
DEFAULT_OUTPUT = AI_SERVICE_ROOT / "datasets" / "smoking_yolo"
MARKER = ".campussentinel_prepared"          # identifies folders this script created (safe --overwrite)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ["cigarette"]
EDGE_TOLERANCE = 0.005                        # boxes may exceed the image by 0.5% (rounding in exports)
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]+$")


# --------------------------------------------------------------------------- helpers
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_key(stem: str) -> str:
    """'IMG_12_jpg.rf.ab12cd' → 'IMG_12_jpg' (all augmented copies of one photo share it)."""
    return ROBOFLOW_SUFFIX.sub("", stem)


def decode_image(path: Path) -> tuple[int, int] | str:
    try:
        with Image.open(path) as im:
            im.load()
            return im.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        return f"{type(exc).__name__}: {exc}"


def validate_label(path: Path) -> tuple[list[tuple[float, float, float, float]], list[str], list[str]]:
    """Return (boxes, errors, warnings). Any error makes the whole image invalid."""
    boxes, errors, warnings = [], [], []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"label unreadable: {exc}"], []
    seen = set()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            kind = "segmentation polygon" if len(parts) > 5 and len(parts) % 2 == 1 else "malformed"
            errors.append(f"line {n}: {kind} ({len(parts)} values, expected 5)")
            continue
        if not re.fullmatch(r"\d+", parts[0]):
            errors.append(f"line {n}: class id {parts[0]!r} is not an integer")
            continue
        if int(parts[0]) != 0:
            errors.append(f"line {n}: class id {parts[0]} (only 0 = cigarette allowed)")
            continue
        try:
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"line {n}: non-numeric coordinate")
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            errors.append(f"line {n}: NaN/Inf coordinate")
        elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            errors.append(f"line {n}: coordinate outside 0..1 ({x}, {y}, {w}, {h})")
        elif w <= 0 or h <= 0:
            errors.append(f"line {n}: zero-size box (w={w}, h={h})")
        elif (x - w / 2 < -EDGE_TOLERANCE or x + w / 2 > 1 + EDGE_TOLERANCE
              or y - h / 2 < -EDGE_TOLERANCE or y + h / 2 > 1 + EDGE_TOLERANCE):
            errors.append(f"line {n}: box extends outside the image")
        else:
            key = (round(x, 6), round(y, 6), round(w, 6), round(h, 6))
            if key in seen:
                warnings.append(f"line {n}: duplicate box")
            seen.add(key)
            boxes.append((x, y, w, h))
    return boxes, errors, warnings


def scan_split(src: Path, split: str) -> dict:
    """Validate <src>/<split>/images + labels. Returns dict with valid items and problems."""
    img_dir, lbl_dir = src / split / "images", src / split / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"expected {img_dir} and {lbl_dir}")

    all_files = sorted(p for p in img_dir.iterdir() if p.is_file())
    images = [p for p in all_files if p.suffix.lower() in IMAGE_EXTS]
    other = [p.name for p in all_files if p.suffix.lower() not in IMAGE_EXTS]
    labels = {p.stem.lower(): p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

    by_stem = defaultdict(list)
    for p in images:
        by_stem[p.stem.lower()].append(p)  # case-insensitive: Windows treats A.jpg and a.jpg as one name

    report = {"split": split, "images_found": len(images), "label_files_found": len(labels),
              "non_image_files": other, "excluded": [], "warnings": [], "valid": []}
    for i, (stem, paths) in enumerate(sorted(by_stem.items()), 1):
        if len(paths) > 1:
            for p in paths:
                report["excluded"].append({"file": p.name, "reason": f"duplicate stem with {[q.name for q in paths]}"})
            continue
        img = paths[0]
        lbl = labels.get(stem)
        if lbl is None:
            report["excluded"].append({"file": img.name, "reason": "missing label file"})
            continue
        size = decode_image(img)
        if isinstance(size, str):
            report["excluded"].append({"file": img.name, "reason": f"image cannot be decoded: {size}"})
            continue
        boxes, errors, warnings = validate_label(lbl)
        if errors:
            report["excluded"].append({"file": img.name, "reason": "invalid label: " + "; ".join(errors[:3])
                                       + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")})
            continue
        if warnings:
            report["warnings"].append({"file": lbl.name, "warning": "; ".join(warnings[:3])})
        report["valid"].append({"image": img, "label": lbl, "boxes": boxes, "size": size,
                                "sha256": sha256_of(img), "group": source_key(img.stem)})
        if i % 300 == 0:
            print(f"    {split}: {i}/{len(by_stem)} checked")

    image_stems = set(by_stem)
    report["orphan_labels"] = sorted(labels[s].name for s in labels.keys() - image_stems)
    return report


def summarise(items: list[dict]) -> dict:
    boxes = [b for it in items for b in it["boxes"]]
    px_areas = sorted(b[2] * it["size"][0] * b[3] * it["size"][1] for it in items for b in it["boxes"])
    return {
        "images": len(items),
        "images_with_cigarette": sum(1 for it in items if it["boxes"]),
        "background_images": sum(1 for it in items if not it["boxes"]),
        "boxes": len(boxes),
        "class_distribution": {"0 (cigarette)": len(boxes)},
        "max_boxes_in_one_image": max((len(it["boxes"]) for it in items), default=0),
        "median_box_area_px": round(px_areas[len(px_areas) // 2]) if px_areas else None,
        "small_boxes_lt_32x32px": sum(1 for a in px_areas if a < 32 * 32),
    }


def print_split_report(rep: dict) -> None:
    s = summarise(rep["valid"])
    print(f"\n  [{rep['split']}] images {rep['images_found']}, label files {rep['label_files_found']}")
    print(f"      valid {s['images']}  (with cigarette {s['images_with_cigarette']}, empty/background "
          f"{s['background_images']})  boxes {s['boxes']}  class 0: {s['boxes']}")
    print(f"      median box area {s['median_box_area_px']} px²; boxes smaller than 32×32 px: "
          f"{s['small_boxes_lt_32x32px']}; max boxes per image {s['max_boxes_in_one_image']}")
    print(f"      excluded {len(rep['excluded'])} | orphan labels (no image) {len(rep['orphan_labels'])} | "
          f"non-image files {len(rep['non_image_files'])} | warnings {len(rep['warnings'])}")
    for e in rep["excluded"][:25]:
        print(f"        EXCLUDED {e['file']}: {e['reason']}")
    if len(rep["excluded"]) > 25:
        print(f"        ... {len(rep['excluded']) - 25} more (see prepare_report.json)")
    for o in rep["orphan_labels"][:10]:
        print(f"        ORPHAN LABEL {o}")
    for f in rep["non_image_files"][:10]:
        print(f"        NON-IMAGE FILE {f}")
    for w in rep["warnings"][:5]:
        print(f"        WARNING {w['file']}: {w['warning']}")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and split the cigarette YOLO dataset (read-only on the source).")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="validate and report; copy nothing")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing processed dataset")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    src, out = args.source.resolve(), args.output.resolve()
    if not src.is_dir():
        print(f"ERROR: source dataset not found: {src}")
        return 1
    if out == src or src in out.parents or out in src.parents:
        print("ERROR: output folder must be separate from the source dataset.")
        return 1

    # ---- 1. source description ----
    print(f"Source (read-only): {src}")
    yaml_path = src / "data.yaml"
    source_yaml = yaml_path.read_text(encoding="utf-8", errors="replace") if yaml_path.is_file() else None
    print("\n--- source data.yaml ---")
    print(source_yaml.strip() if source_yaml else "(missing)")
    readmes = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md"))
    for r in readmes:
        print(f"\n--- {r.name} ---")
        print(r.read_text(encoding="utf-8", errors="replace").strip()[:2500])
    if source_yaml and "cigarette" not in source_yaml.lower():
        print("\nWARNING: 'cigarette' not found in source data.yaml names. Check the class mapping before training.")
    extra_dirs = sorted(p.name for p in src.iterdir() if p.is_dir() and p.name not in ("train", "test"))
    if extra_dirs:
        print(f"\nNOTE: other folders in source (not used): {extra_dirs}")

    # ---- 2. validation ----
    print("\nValidating images and labels ...")
    try:
        train_rep, test_rep = scan_split(src, "train"), scan_split(src, "test")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    print_split_report(train_rep)
    print_split_report(test_rep)

    # ---- 3/4. grouping + leakage against test ----
    test_hashes = {it["sha256"] for it in test_rep["valid"]}
    test_groups = {it["group"] for it in test_rep["valid"]}
    all_groups = defaultdict(list)
    for it in train_rep["valid"]:
        all_groups[it["group"]].append(it)
    groups, leaked = {}, []
    for key, members in all_groups.items():
        # If ANY copy of a source photo is in the test set, drop ALL its copies from train/val.
        if key in test_groups:
            reason = f"same source image as a test image ({key})"
        elif any(m["sha256"] in test_hashes for m in members):
            reason = "this source group contains a file byte-identical to a test image"
        else:
            groups[key] = members
            continue
        leaked += [{"file": m["image"].name, "reason": reason} for m in members]
    multi = sum(1 for g in groups.values() if len(g) > 1)
    print(f"\n  Train source groups: {len(groups)} ({multi} with >1 image, e.g. Roboflow augmented copies)")
    print(f"  Train images excluded for test leakage: {len(leaked)}")
    for e in leaked[:15]:
        print(f"        LEAKAGE {e['file']}: {e['reason']}")

    # ---- 5. deterministic group split ----
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    total = sum(len(groups[k]) for k in keys)
    target_val = round(total * args.val_fraction)
    val_items, train_items, n_val = [], [], 0
    for k in keys:
        if n_val < target_val:
            val_items += groups[k]
            n_val += len(groups[k])
        else:
            train_items += groups[k]
    test_items = test_rep["valid"]

    print(f"\n  {'split':<6}{'images':>8}{'w/ cig':>8}{'bg':>6}{'boxes':>8}")
    splits = {"train": train_items, "val": val_items, "test": test_items}
    summaries = {}
    for name, items in splits.items():
        s = summaries[name] = summarise(items)
        print(f"  {name:<6}{s['images']:>8}{s['images_with_cigarette']:>8}{s['background_images']:>6}{s['boxes']:>8}")
    print(f"  (val target {args.val_fraction:.0%} of {total} = {target_val}; whole groups kept together, seed {args.seed})")

    if args.dry_run:
        print("\n--dry-run: nothing copied or written.")
        return 0

    # ---- 6. write processed dataset ----
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            print(f"\nERROR: {out} already exists and is not empty. Use --overwrite to rebuild it.")
            return 1
        if not (out / MARKER).is_file():
            print(f"\nERROR: {out} was not created by this script (no {MARKER}); refusing to delete it.")
            return 1
        shutil.rmtree(out)
        print(f"\nRemoved previous processed dataset: {out}")

    for name, items in splits.items():
        img_dst, lbl_dst = out / name / "images", out / name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for it in items:
            for s_path, d_dir in ((it["image"], img_dst), (it["label"], lbl_dst)):
                dst = d_dir / s_path.name
                if dst.exists():  # cannot happen after stem checks; never overwrite silently
                    print(f"ERROR: name collision at {dst}; aborting.")
                    return 1
                shutil.copy2(s_path, dst)

    out_posix = out.as_posix()
    (out / "data.yaml").write_text(
        "# Generated by ai-service/training/smoking/yolo/prepare_dataset.py — do not edit by hand.\n"
        f"# Source (unchanged): {src.as_posix()}\n"
        f"# train/val: seed-{args.seed} group split of the source train folder; test: source test folder.\n"
        f"path: {out_posix}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: cigarette\n",
        encoding="utf-8",
    )

    def listing(items):
        return sorted(it["image"].name for it in items)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": src.as_posix(), "output": out_posix, "seed": args.seed, "val_fraction": args.val_fraction,
        "source_data_yaml": source_yaml,
        "summary": summaries,
        "excluded": {"train": train_rep["excluded"], "test": test_rep["excluded"], "train_test_leakage": leaked},
        "orphan_labels": {"train": train_rep["orphan_labels"], "test": test_rep["orphan_labels"]},
        "non_image_files": {"train": train_rep["non_image_files"], "test": test_rep["non_image_files"]},
        "warnings": {"train": train_rep["warnings"], "test": test_rep["warnings"]},
        "files": {name: listing(items) for name, items in splits.items()},
    }
    (out / "prepare_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / MARKER).write_text("created by prepare_dataset.py\n", encoding="utf-8")

    print(f"\nProcessed dataset written: {out}")
    print(f"  data.yaml: {out / 'data.yaml'}")
    print(f"  report:    {out / 'prepare_report.json'}")
    print("Source dataset was not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
