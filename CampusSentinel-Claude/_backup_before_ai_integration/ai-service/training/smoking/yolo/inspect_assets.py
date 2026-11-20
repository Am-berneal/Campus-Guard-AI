"""READ-ONLY inspection: is there a usable cigarette object-detection dataset / model?

Nothing is created, moved, modified or deleted. The script only lists and reads files.

Reports
  1. Environment: Python, PyTorch/CUDA, Ultralytics (installed or not, version).
  2. Existing YOLO code, docs and weight files in the project.
  3. Dataset config YAMLs (files with `names` + `train`/`val` keys).
  4. Every `labels` folder: label-file count, matching images, class-ID histogram,
     malformed lines, out-of-range coordinates, empty label files, images without labels.
  5. Other annotation formats: COCO JSON, Pascal VOC XML.

Run from the ai-service folder:
    python -m training.smoking.yolo.inspect_assets
    python -m training.smoking.yolo.inspect_assets --search "C:\\Users\\hp\\Downloads"
    python -m training.smoking.yolo.inspect_assets --read-model-names   (loads .pt files via Ultralytics;
                                                                        only use on files you trust)
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SKIP_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__", ".idea", ".vscode"}
CLASSIFICATION_DATASET = "dataset_smoking"  # SmokingCNN data: image-level labels only, no boxes
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
MAX_JSON_MB = 60


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def walk(root: Path):
    """os.walk that skips virtualenvs/node_modules and does not descend into the CNN dataset."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if Path(dirpath).name == CLASSIFICATION_DATASET:
            dirnames[:] = []
            continue
        yield Path(dirpath), dirnames, filenames


# --------------------------------------------------------------------------- 1. environment
def report_environment() -> None:
    section("1. Environment")
    print(f"  Python      {platform.python_version()} ({sys.executable})")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  PyTorch     {torch.__version__}  CUDA available: {cuda}"
              + (f"  ({torch.cuda.get_device_name(0)})" if cuda else ""))
    except ImportError:
        print("  PyTorch     NOT INSTALLED")
    for pkg in ("ultralytics", "torchvision", "opencv-python", "PyYAML", "matplotlib"):
        try:
            print(f"  {pkg:<12}{md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:<12}NOT INSTALLED")


# --------------------------------------------------------------------------- 2. existing assets
def report_existing(roots: list[Path]) -> None:
    section("2. Existing YOLO code, docs and weight files")
    ai = PROJECT_ROOT / "ai-service"
    for rel in ("training/smoking/yolo", "models/smoking", "docs", "runs"):
        folder = ai / rel
        if not folder.exists():
            print(f"  ai-service/{rel}: (does not exist)")
            continue
        files = [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        print(f"  ai-service/{rel}: {len(files)} file(s)")
        for p in files[:40]:
            print(f"      {p.relative_to(ai)}  ({human(p.stat().st_size)})")
    top_docs = PROJECT_ROOT / "docs"
    if top_docs.exists():
        print(f"  docs/ (project root): {sorted(p.name for p in top_docs.iterdir())}")

    print("\n  Weight files (*.pt, *.onnx) found anywhere under the search roots:")
    found = 0
    for root in roots:
        for dirpath, _, filenames in walk(root):
            for f in filenames:
                if f.lower().endswith((".pt", ".onnx")):
                    p = dirpath / f
                    print(f"      {p}  ({human(p.stat().st_size)})")
                    found += 1
    if not found:
        print("      none")


def report_model_names(roots: list[Path]) -> None:
    section("2b. Class names stored in .pt files (loaded with Ultralytics)")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  Ultralytics not installed — skipped.")
        return
    for root in roots:
        for dirpath, _, filenames in walk(root):
            for f in filenames:
                if not f.lower().endswith(".pt") or "smoking_cnn" in f:
                    continue
                p = dirpath / f
                try:
                    m = YOLO(str(p))
                    print(f"  {p}\n      task={m.task}  names={m.names}")
                except Exception as exc:  # not a YOLO checkpoint, or incompatible
                    print(f"  {p}\n      not loadable as YOLO: {type(exc).__name__}: {str(exc)[:120]}")


# --------------------------------------------------------------------------- 3. YAML configs
def report_yaml(roots: list[Path]) -> None:
    section("3. Dataset configuration YAMLs")
    try:
        import yaml
    except ImportError:
        yaml = None
        print("  (PyYAML not installed: showing raw text of candidate files)")
    found = 0
    for root in roots:
        for dirpath, _, filenames in walk(root):
            for f in filenames:
                if not f.lower().endswith((".yaml", ".yml")):
                    continue
                p = dirpath / f
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if "names" not in text or not any(k in text for k in ("train:", "val:")):
                    continue
                found += 1
                print(f"\n  {p}")
                if yaml:
                    try:
                        cfg = yaml.safe_load(text) or {}
                        for key in ("path", "train", "val", "test", "nc", "names"):
                            if key in cfg:
                                print(f"      {key}: {cfg[key]}")
                        continue
                    except yaml.YAMLError as exc:
                        print(f"      YAML parse error: {exc}")
                for line in text.splitlines()[:25]:
                    print(f"      | {line}")
    if not found:
        print("  none found")


# --------------------------------------------------------------------------- 4. YOLO label folders
def inspect_labels_dir(labels_dir: Path) -> None:
    txts = sorted(labels_dir.glob("*.txt"))
    images_dir = labels_dir.parent / "images" if labels_dir.name == "labels" else None
    # Common layouts: <split>/labels + <split>/images  OR  labels/<split> + images/<split>
    if labels_dir.parent.name == "labels":
        images_dir = labels_dir.parent.parent / "images" / labels_dir.name
    images = []
    if images_dir and images_dir.is_dir():
        images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    class_counts, bad_lines, out_of_range, empty_files, boxes = Counter(), 0, 0, 0, 0
    examples = []
    for t in txts:
        lines = [ln.strip() for ln in t.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        if not lines:
            empty_files += 1
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:  # >5 values would be a segmentation polygon, not a box
                bad_lines += 1
                if len(examples) < 3:
                    examples.append(f"{t.name}: '{ln[:80]}' ({len(parts)} values)")
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                bad_lines += 1
                continue
            class_counts[cls] += 1
            boxes += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                out_of_range += 1

    label_stems = {t.stem for t in txts}
    image_stems = {p.stem for p in images}
    print(f"\n  {labels_dir}")
    print(f"      label files: {len(txts)}   images: {len(images)}"
          + (f" (in {images_dir})" if images_dir and images_dir.is_dir() else "   (no matching images folder found)"))
    print(f"      boxes: {boxes}   class-ID histogram: {dict(sorted(class_counts.items()))}")
    print(f"      empty label files (background images): {empty_files}")
    print(f"      malformed lines: {bad_lines}   out-of-range coordinates: {out_of_range}")
    if images:
        print(f"      images without label file: {len(image_stems - label_stems)}   "
              f"label files without image: {len(label_stems - image_stems)}")
    for ex in examples:
        print(f"      e.g. {ex}")
    if txts:
        sample = txts[0].read_text(encoding="utf-8", errors="replace").strip().splitlines()[:2]
        print(f"      sample ({txts[0].name}): {sample}")


def report_labels(roots: list[Path]) -> None:
    section("4. YOLO-format label folders")
    found = 0
    for root in roots:
        for dirpath, dirnames, filenames in walk(root):
            in_labels_tree = dirpath.name == "labels" or dirpath.parent.name == "labels"
            if in_labels_tree and any(f.endswith(".txt") for f in filenames):
                found += 1
                inspect_labels_dir(dirpath)
    if not found:
        print("  none found")


# --------------------------------------------------------------------------- 5. other formats
def report_other_formats(roots: list[Path]) -> None:
    section("5. Other annotation formats (COCO JSON / Pascal VOC XML)")
    coco, voc_dirs = 0, Counter()
    for root in roots:
        for dirpath, _, filenames in walk(root):
            for f in filenames:
                p = dirpath / f
                if f.lower().endswith(".json") and "node_modules" not in p.parts:
                    if p.stat().st_size > MAX_JSON_MB * 1024 * 1024:
                        continue
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except (OSError, ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
                        coco += 1
                        cats = [c.get("name") for c in data["categories"]]
                        print(f"  COCO: {p}\n      images {len(data['images'])}, "
                              f"annotations {len(data['annotations'])}, categories {cats}")
                elif f.lower().endswith(".xml"):
                    try:
                        head = p.read_text(encoding="utf-8", errors="replace")[:4000]
                    except OSError:
                        continue
                    if "<bndbox>" in head:
                        voc_dirs[dirpath] += 1
    for d, n in voc_dirs.items():
        print(f"  Pascal VOC: {d}  ({n} XML files with bounding boxes)")
    if not coco and not voc_dirs:
        print("  none found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only inspection of cigarette detection assets.")
    parser.add_argument("--search", type=Path, action="append", default=[],
                        help="extra folder to search (repeatable), e.g. your Downloads folder")
    parser.add_argument("--read-model-names", action="store_true",
                        help="load .pt files with Ultralytics to print their class names (trusted files only)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    roots = [PROJECT_ROOT] + [p.resolve() for p in args.search if p.exists()]
    for p in args.search:
        if not p.exists():
            print(f"WARNING: search folder does not exist: {p}")
    print(f"Read-only inspection. Search roots: {[str(r) for r in roots]}")
    print(f"Skipped: {sorted(SKIP_DIRS)}; '{CLASSIFICATION_DATASET}' (classification data, no boxes) not scanned.")

    report_environment()
    report_existing(roots)
    if args.read_model_names:
        report_model_names(roots)
    report_yaml(roots)
    report_labels(roots)
    report_other_formats(roots)
    section("Done — nothing was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
