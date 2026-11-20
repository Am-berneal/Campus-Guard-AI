"""Polygon zones and point-in-polygon checks.

Polygon coordinates are NORMALISED: x = fraction of frame width (0 left → 1 right),
y = fraction of frame height (0 top → 1 bottom). A zone drawn once therefore works at
any video resolution from the same camera. If the camera moves or zooms, redraw the zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from training.restricted_zone.schedule import Schedule


@dataclass(frozen=True)
class Zone:
    zone_id: str
    name: str
    status: str                      # "restricted" → schedule applies; "allowed" → never a violation
    polygon: tuple[tuple[float, float], ...]
    schedule: Schedule
    enabled: bool = True

    def __post_init__(self) -> None:
        # OpenCV wants an (N, 1, 2) float32 contour. Cached on the frozen instance.
        object.__setattr__(self, "_contour", np.asarray(self.polygon, dtype=np.float32).reshape(-1, 1, 2))

    def contains(self, point_norm: tuple[float, float]) -> bool:
        """True if the normalised point is inside the polygon or on its edge."""
        return cv2.pointPolygonTest(self._contour, (float(point_norm[0]), float(point_norm[1])), False) >= 0

    def is_restricted_at(self, when: datetime, tz: ZoneInfo) -> bool:
        if not self.enabled or self.status == "allowed":
            return False
        return self.schedule.status_at(when, tz) == "restricted"

    def polygon_pixels(self, width: int, height: int) -> np.ndarray:
        return np.array([[round(x * width), round(y * height)] for x, y in self.polygon], dtype=np.int32)


def bottom_center(bbox_xyxy: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float]:
    """Bottom-centre of a pixel box, normalised to 0..1.

    The bottom of a person's box is roughly where their feet touch the ground. Using it
    (not the box centre) avoids counting someone standing in front of a zone as inside it.
    """
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")
    x1, _y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2 / width, y2 / height)
