"""Small OpenCV helpers shared by the detectors: probe a video, stream selected frames, save evidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

LOG = logging.getLogger("inference.video")


class VideoReadError(Exception):
    """The video cannot be opened or decoded."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int      # from the container header (may be wrong for damaged files)
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict:
        return {"fps": round(self.fps, 3), "frame_count": self.frame_count, "width": self.width,
                "height": self.height, "duration_seconds": round(self.duration_s, 2)}


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError("OpenCV cannot open the video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info = VideoInfo(fps if 1 <= fps <= 240 else 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if fps < 1 or fps > 240:
            LOG.warning("%s: implausible FPS %.2f, assuming 25", path.name, fps)
    finally:
        cap.release()
    if info.width <= 0 or info.height <= 0:
        raise VideoReadError("video has no readable frames")
    return info


def iter_frames(path: Path, wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) for the wanted indices, reading sequentially.

    grab() skips decoding of unwanted frames; sequential reading avoids unreliable
    seeking in AVI/MP4 files. Only one frame is held in memory at a time.
    """
    if not wanted:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError("OpenCV cannot open the video")
    last = max(wanted)
    i = 0
    try:
        while i <= last and cap.grab():
            if i in wanted:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield i, frame
                else:
                    LOG.warning("%s: frame %d could not be decoded", path.name, i)
            i += 1
    finally:
        cap.release()


def save_jpeg(image: np.ndarray, dst: Path, max_width: int = 960) -> Path:
    """Save a (down-scaled) evidence image."""
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"could not write {dst}")
    return dst


def label(image: np.ndarray, text: str, color=(0, 0, 255)) -> np.ndarray:
    """Draw a caption bar at the top of an evidence image."""
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def group_times(times: list[float], gap_s: float) -> list[list[int]]:
    """Group sorted timestamps into runs where neighbours are at most gap_s apart (returns index lists)."""
    groups: list[list[int]] = []
    for i, t in enumerate(times):
        if groups and t - times[groups[-1][-1]] <= gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups
