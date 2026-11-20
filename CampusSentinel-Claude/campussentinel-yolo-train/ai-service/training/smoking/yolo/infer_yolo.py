"""Run the fine-tuned cigarette detector on an image or a video and save annotated output.

Run from the ai-service folder:
    python -m training.smoking.yolo.infer_yolo --image path\\to\\photo.jpg
    python -m training.smoking.yolo.infer_yolo --video path\\to\\clip.mp4 --show

Outputs in ai-service/runs/smoking_yolo/inference/<name>_<time>/
    image: <name>_annotated.jpg + detections.json
    video: annotated.mp4 + detections.json (frames that contain detections)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from training.smoking.yolo.cigarette_detector import AI_SERVICE_ROOT, DEFAULT_WEIGHTS, CigaretteDetector

OUT_ROOT = AI_SERVICE_ROOT / "runs" / "smoking_yolo" / "inference"


def run_image(det: CigaretteDetector, path: Path, out_dir: Path) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cannot read image {path}")
    detections, annotated = det.detect_and_draw(frame)
    cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"), annotated)
    (out_dir / "detections.json").write_text(json.dumps(
        {"source": str(path), "frame_size": list(frame.shape[1::-1]), "detections": detections}, indent=2),
        encoding="utf-8")
    print(f"{len(detections)} cigarette detection(s)")
    for d in detections:
        print(f"  {d['class_name']} {d['confidence']:.2f} box {d['bbox_xyxy']}")


def run_video(det: CigaretteDetector, path: Path, out_dir: Path, show: bool, max_frames: int) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames_with, idx, total, t0 = [], 0, 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections, annotated = det.detect_and_draw(frame)
            if detections:
                frames_with.append({"frame_index": idx, "time_s": round(idx / fps, 3), "detections": detections})
                total += len(detections)
            writer.write(annotated)
            if show:
                cv2.imshow("Cigarette detector (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
            if max_frames and idx >= max_frames:
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    secs = time.perf_counter() - t0
    (out_dir / "detections.json").write_text(json.dumps(
        {"source": str(path), "fps": fps, "frames_processed": idx, "frames": frames_with}, indent=2), encoding="utf-8")
    print(f"{idx} frames in {secs:.1f}s ({idx / max(secs, 1e-6):.1f} fps) | "
          f"{len(frames_with)} frames with detections, {total} boxes")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cigarette detection on an image or video.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--video", type=Path)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (untuned default)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = args.image or args.video
    if not path.is_file():
        print(f"ERROR: file not found: {path}")
        return 1
    try:
        det = CigaretteDetector(args.weights, args.conf, args.iou, args.imgsz, args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    out_dir = OUT_ROOT / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model {args.weights} on {det.device} | conf {args.conf}")
    try:
        if args.image:
            run_image(det, path, out_dir)
        else:
            run_video(det, path, out_dir, args.show, args.max_frames)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        det.close()
    print(f"Saved to {out_dir}")
    print("Note: a detected cigarette is not proof that someone is smoking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
