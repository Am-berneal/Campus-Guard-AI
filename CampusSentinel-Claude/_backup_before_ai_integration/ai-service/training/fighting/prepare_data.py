"""RWF-2000 preprocessing: sample 16 frames per video → letterboxed 224×224 RGB uint8 .npy.

The ORIGINAL dataset is only read. Nothing in it is modified, moved or copied.

For each video (one at a time, no multiprocessing):
  1. Open with OpenCV; read FPS, reported frame count, width, height.
  2. Choose T frame indices uniformly over the whole clip (first and last frame included).
  3. Read the video sequentially: grab() every frame (cheap), retrieve() only selected ones.
     Sequential reading avoids inaccurate seeking in AVI files. If the file turns out
     shorter than its header claims, indices are recomputed from the real count.
  4. Each selected frame: resize keeping aspect ratio → pad (black) to 224×224 → BGR→RGB.
  5. Save (T, 224, 224, 3) uint8 atomically (temp file + rename), so an interrupted run
     never leaves a half-written file that looks valid.

Resume: a video is skipped when its .npy already exists and has the right shape/dtype
(use --force to regenerate). manifest.csv is merged with the previous one and re-written
regularly during the run, so progress survives interruption.

Run from the ai-service folder:
    python -m training.fighting.prepare_data --limit 2
    python -m training.fighting.prepare_data
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("fighting.prepare")

DEFAULT_SOURCE = Path(r"D:\CampusSentinel-Datasets\fighting")
DEFAULT_OUTPUT = Path(r"D:\CampusSentinel-Datasets\fighting_processed")
SPLITS = ("train", "val")
CLASSES = {"NonFight": 0, "Fight": 1}          # class folder → numeric label
VIDEO_EXTS = {".avi"}
MANIFEST_FIELDS = [
    "source_path", "npy_path", "split", "class_name", "label", "frame_count", "fps", "width", "height",
    "frame_indices", "output_shape", "status", "error",
]
SAVE_MANIFEST_EVERY = 25


class VideoError(Exception):
    """A video that cannot be processed (unreadable, empty, corrupt frames)."""


# --------------------------------------------------------------------------- pure functions (unit-tested)
def uniform_indices(frame_count: int, num_frames: int) -> list[int]:
    """`num_frames` evenly spaced indices over [0, frame_count-1], first and last included.

    150 frames, 16 samples → 0, 10, 20, …, 139, 149 (spacing ≈ 9.93, rounded).
    Shorter videos than num_frames repeat indices so the output length is constant.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return [0]
    return [int(round(i * (frame_count - 1) / (num_frames - 1))) for i in range(num_frames)]


def letterbox(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Resize so the LONGER side equals `size` (aspect ratio kept), centre on a black square."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR frame")
    h, w = frame_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR   # AREA avoids aliasing when shrinking
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[top:top + new_h, left:left + new_w] = resized
    return out


def preprocess_frame(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Letterbox, then BGR → RGB. Returns uint8 (size, size, 3)."""
    return cv2.cvtColor(letterbox(frame_bgr, size), cv2.COLOR_BGR2RGB)


@dataclass
class VideoInfo:
    frame_count: int      # frames actually decoded (may differ from the header)
    fps: float
    width: int
    height: int
    indices: list[int]
    note: str = ""        # e.g. header/actual frame-count mismatch


def _read_selected(path: Path, indices: list[int], size: int) -> tuple[dict[int, np.ndarray], int]:
    """Read sequentially; return {index: processed frame} and the number of frames decoded."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("OpenCV cannot open the file")
    wanted, frames, i = set(indices), {}, 0
    try:
        # Read to the end: this also tells us the TRUE frame count (grab() skips decoding).
        while cap.grab():
            if i in wanted:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise VideoError(f"frame {i} could not be decoded")
                frames[i] = preprocess_frame(frame, size)
            i += 1
    finally:
        cap.release()
    return frames, i


def process_video(path: Path, num_frames: int, size: int) -> tuple[np.ndarray, VideoInfo]:
    """Return (T, size, size, 3) uint8 RGB array and video metadata. Raises VideoError."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("OpenCV cannot open the file")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise VideoError(f"invalid frame size {width}x{height}")

    indices = uniform_indices(reported, num_frames) if reported > 0 else [0]
    frames, actual = _read_selected(path, indices, size)
    if actual == 0:
        raise VideoError("no frames could be decoded")
    if actual != reported or len(frames) != len(set(indices)):
        # Header was wrong (common with damaged AVIs): resample using the real length.
        note = f"header reports {reported} frames but {actual} decoded; indices resampled from {actual}"
        LOG.warning("%s: %s", path.name, note)
        indices = uniform_indices(actual, num_frames)
        frames, actual2 = _read_selected(path, indices, size)
        if len(frames) != len(set(indices)):
            raise VideoError(f"only {len(frames)} of {len(set(indices))} selected frames decoded (video has {actual2})")
        actual = actual2

    else:
        note = ""
    clip = np.stack([frames[i] for i in indices]).astype(np.uint8, copy=False)
    return clip, VideoInfo(actual, round(fps, 3), width, height, indices, note)


def save_atomic(arr: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, dst)  # atomic on the same drive


def is_valid_npy(path: Path, shape: tuple[int, ...]) -> bool:
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == shape and arr.dtype == np.uint8
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- manifest
def read_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["source_path"]: row for row in csv.DictReader(f)}


def write_manifest(path: Path, rows: dict[str, dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    ordered = sorted(rows.values(), key=lambda r: (r["split"], r["class_name"], r["npy_path"]))
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(tmp, path)


def probe_metadata(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 3),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


# --------------------------------------------------------------------------- main pipeline
def run(source: Path, output: Path, num_frames: int = 16, size: int = 224,
        limit: int = 0, force: bool = False) -> dict:
    """Process the dataset. Returns the report dict (also written to preprocessing_report.json)."""
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset not found: {source}")
    if output == source or source in output.parents:
        raise ValueError("output folder must not be inside the source dataset")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    rows = read_manifest(manifest_path)
    shape = (num_frames, size, size, 3)
    shape_str = "x".join(map(str, shape))

    counts = Counter()        # (split, class, outcome) → n
    problems, warnings, ignored, missing_dirs = [], [], [], []
    t_start, done = time.perf_counter(), 0

    for split in SPLITS:
        for class_name, label in CLASSES.items():
            class_dir = source / split / class_name
            if not class_dir.is_dir():
                LOG.error("Missing directory: %s", class_dir)
                missing_dirs.append(str(class_dir))
                continue
            files = sorted(p for p in class_dir.iterdir() if p.is_file())
            videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
            ignored += [str(p) for p in files if p.suffix.lower() not in VIDEO_EXTS]
            stems = Counter(p.stem.lower() for p in videos)
            if limit:
                videos = videos[:limit]
            LOG.info("%s/%s: %d videos%s", split, class_name, len(videos), f" (limit {limit})" if limit else "")

            for n, video in enumerate(videos, 1):
                rel_npy = f"{split}/{class_name}/{video.stem}.npy"
                dst = output / rel_npy
                key = str(video)
                base = {"source_path": key, "npy_path": rel_npy, "split": split, "class_name": class_name,
                        "label": label, "output_shape": shape_str}
                if stems[video.stem.lower()] > 1:
                    msg = "another video in this folder has the same name (different extension/case)"
                    rows[key] = {**base, "frame_count": "", "fps": "", "width": "", "height": "",
                                 "frame_indices": "", "status": "failed", "error": msg}
                    counts[(split, class_name, "failed")] += 1
                    problems.append({"video": key, "error": msg})
                    LOG.error("[%s/%s %d/%d] %s FAILED: %s", split, class_name, n, len(videos), video.name, msg)
                    continue

                prev = rows.get(key)
                if not force and dst.is_file() and is_valid_npy(dst, shape):
                    if not (prev and prev.get("status") == "ok" and prev.get("output_shape") == shape_str):
                        fc, fps, w, h = probe_metadata(video)  # manifest row lost: rebuild metadata
                        prev = {**base, "frame_count": fc, "fps": fps, "width": w, "height": h,
                                "frame_indices": " ".join(map(str, uniform_indices(fc, num_frames))) if fc else "",
                                "status": "ok", "error": ""}
                    rows[key] = prev
                    counts[(split, class_name, "skipped")] += 1
                    continue

                t0 = time.perf_counter()
                try:
                    clip, info = process_video(video, num_frames, size)
                    if clip.shape != shape or clip.dtype != np.uint8:
                        raise VideoError(f"unexpected output {clip.shape} {clip.dtype}")
                    save_atomic(clip, dst)
                    rows[key] = {**base, "frame_count": info.frame_count, "fps": info.fps, "width": info.width,
                                 "height": info.height, "frame_indices": " ".join(map(str, info.indices)),
                                 "status": "ok", "error": f"WARNING: {info.note}" if info.note else ""}
                    counts[(split, class_name, "processed")] += 1
                    if info.note:
                        warnings.append({"video": key, "warning": info.note})
                    if n % 20 == 0 or n == len(videos):
                        LOG.info("[%s/%s %d/%d] %s ok (%dx%d, %d frames, %.2fs)", split, class_name, n,
                                 len(videos), video.name, info.width, info.height, info.frame_count,
                                 time.perf_counter() - t0)
                except (VideoError, cv2.error, ValueError, OSError) as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    if dst.is_file() and force:
                        LOG.warning("keeping previous %s (regeneration failed)", dst.name)
                    rows[key] = {**base, "frame_count": "", "fps": "", "width": "", "height": "",
                                 "frame_indices": "", "status": "failed", "error": msg}
                    counts[(split, class_name, "failed")] += 1
                    problems.append({"video": key, "error": msg})
                    LOG.error("[%s/%s %d/%d] %s FAILED: %s", split, class_name, n, len(videos), video.name, msg)

                done += 1
                if done % SAVE_MANIFEST_EVERY == 0:
                    write_manifest(manifest_path, rows)

    write_manifest(manifest_path, rows)
    elapsed = time.perf_counter() - t_start

    per_class = {}
    for split in SPLITS:
        for class_name in CLASSES:
            per_class[f"{split}/{class_name}"] = {
                k: counts[(split, class_name, k)] for k in ("processed", "skipped", "failed")}
    status_totals = Counter(r["status"] for r in rows.values())
    disk_bytes = sum((output / r["npy_path"]).stat().st_size for r in rows.values()
                     if r["status"] == "ok" and (output / r["npy_path"]).is_file())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source), "output": str(output),
        "config": {"num_frames": num_frames, "size": size, "color": "RGB", "dtype": "uint8",
                   "sampling": "uniform over whole video, first and last frame included",
                   "resize": "aspect ratio preserved, centred, black padding", "labels": CLASSES,
                   "limit_per_class": limit, "force": force},
        "this_run": {"per_class": per_class, "seconds": round(elapsed, 1)},
        "manifest_totals": {"rows": len(rows), "ok": status_totals.get("ok", 0),
                            "failed": status_totals.get("failed", 0)},
        "disk_usage_mb": round(disk_bytes / 1024**2, 1),
        "missing_directories": missing_dirs,
        "ignored_non_video_files": ignored,
        "failures_this_run": problems,
        "warnings_this_run": warnings,
    }
    (output / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess RWF-2000 videos into 16x224x224x3 uint8 .npy clips.")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N videos per class (0 = all)")
    ap.add_argument("--force", action="store_true", help="regenerate .npy files even if valid ones exist")
    ap.add_argument("--verbose", action="store_true", help="log every video")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.frames < 1 or args.size < 16:
        LOG.error("--frames must be >= 1 and --size >= 16")
        return 1

    try:
        report = run(args.source, args.output, args.frames, args.size, args.limit, args.force)
    except (FileNotFoundError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1

    print("\n" + "=" * 70)
    print(f"{'split/class':<16}{'processed':>11}{'skipped':>9}{'failed':>8}")
    for key, c in report["this_run"]["per_class"].items():
        print(f"{key:<16}{c['processed']:>11}{c['skipped']:>9}{c['failed']:>8}")
    mt = report["manifest_totals"]
    print(f"\nManifest: {mt['rows']} rows ({mt['ok']} ok, {mt['failed']} failed) | "
          f"disk {report['disk_usage_mb']} MB | {report['this_run']['seconds']} s")
    if report["missing_directories"]:
        print(f"MISSING DIRECTORIES: {report['missing_directories']}")
    if report["ignored_non_video_files"]:
        print(f"Ignored non-AVI files: {len(report['ignored_non_video_files'])} (listed in report)")
    for w in report["warnings_this_run"][:20]:
        print(f"WARNING {w['video']}: {w['warning']}")
    for p in report["failures_this_run"][:20]:
        print(f"FAILED {p['video']}: {p['error']}")
    print(f"Output: {report['output']}")
    failed = mt["failed"] > 0 or bool(report["missing_directories"])
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
