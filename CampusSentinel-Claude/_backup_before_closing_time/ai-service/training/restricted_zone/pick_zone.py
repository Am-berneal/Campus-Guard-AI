"""Click a polygon on a video's first frame (or an image) and print normalised coordinates.

Left-click: add point   |   u: undo   |   r: reset   |   Enter: print & finish   |   q / Esc: quit

Run from the ai-service folder:
    python -m training.restricted_zone.pick_zone --source path\\to\\clip.mp4
Paste the printed `polygon:` line into your zones YAML.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def first_frame(source: str):
    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return cv2.imread(source)
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video or image file")
    args = p.parse_args()
    frame = first_frame(args.source)
    if frame is None:
        print(f"ERROR: could not read a frame from {args.source}")
        return 1
    h, w = frame.shape[:2]
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    win = "Click zone corners - Enter to finish"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        for pt in points:
            cv2.circle(view, pt, 4, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(view, [np.array(points, dtype=np.int32)], len(points) > 2, (0, 0, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 10):  # Enter
            break
        elif key in (ord("q"), 27):
            points.clear()
            break
    cv2.destroyAllWindows()

    if len(points) < 3:
        print("No polygon (need at least 3 points).")
        return 1
    norm = [[round(x / w, 3), round(y / h, 3)] for x, y in points]
    print(f"Frame size {w}x{h}. Paste into the zone entry:")
    print(f"    polygon: {norm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
